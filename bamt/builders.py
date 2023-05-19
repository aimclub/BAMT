import itertools
from datetime import timedelta

from pgmpy.base import DAG
from pgmpy.estimators import HillClimbSearch
from bamt.redef_HC import hc as hc_method

from bamt.nodes.discrete_node import DiscreteNode
from bamt.nodes.gaussian_node import GaussianNode
from bamt.nodes.conditional_logit_node import ConditionalLogitNode
from bamt.nodes.logit_node import LogitNode
from bamt.nodes.mixture_gaussian_node import MixtureGaussianNode
from bamt.nodes.conditional_mixture_gaussian_node import ConditionalMixtureGaussianNode
from bamt.nodes.conditional_gaussian_node import ConditionalGaussianNode

from bamt.log import logger_builder
from pandas import DataFrame
from bamt.utils import GraphUtils as gru
from bamt.utils import EvoUtils as evo

from golem.core.adapter import DirectAdapter
from golem.core.dag.verification_rules import has_no_cycle, has_no_self_cycled_nodes
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.objective import Objective, ObjectiveEvaluate
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams, GraphOptimizer
from golem.core.optimisers.genetic.operators.selection import SelectionTypesEnum

from typing import Dict, List, Optional, Tuple, Callable, TypedDict, Union, Sequence


class ParamDict(TypedDict, total=False):
    init_edges: Optional[Sequence[str]]
    init_nodes: Optional[List[str]]
    remove_init_edges: bool
    white_list: Optional[Tuple[str, str]]
    bl_add: Optional[List[str]]


class StructureBuilder(object):
    """
    Base Class for Structure Builder.
    It can restrict nodes defined by RESTRICTIONS
    """

    def __init__(self, descriptor: Dict[str, Dict[str, str]]):
        """
        :param descriptor: a dict with types and signs of nodes
        Attributes:
        black_list: a list with restricted connections;
        """
        self.skeleton = {'V': [],
                         'E': []}
        self.descriptor = descriptor

        self.has_logit = bool

        self.black_list = None

    def restrict(self, data: DataFrame,
                 init_nodes: Optional[List[str]],
                 bl_add: Optional[List[str]]):
        """
        :param data: data to deal with
        :param init_nodes: nodes to begin with (thus they have no parents)
        :param bl_add: additional vertices
        """
        node_type = self.descriptor['types']
        blacklist = []
        datacol = data.columns.to_list()

        if not self.has_logit:
            # Has_logit flag allows BN building edges between cont and disc
            RESTRICTIONS = [('cont', 'disc'), ('cont', 'disc_num')]
            for x, y in itertools.product(datacol, repeat=2):
                if x != y:
                    if (node_type[x], node_type[y]) in RESTRICTIONS:
                        blacklist.append((x, y))
        else:
            self.black_list = []
        if init_nodes:
            blacklist += [(x, y)
                          for x in datacol for y in init_nodes if x != y]
        if bl_add:
            blacklist = blacklist + bl_add
        self.black_list = blacklist

    def get_family(self):
        """
        A function that updates a skeleton;
        """
        if not self.skeleton['V']:
            logger_builder.error("Vertex list is None")
            return None
        if not self.skeleton['E']:
            logger_builder.error("Edges list is None")
            return None
        for node_instance in self.skeleton['V']:
            node = node_instance.name
            children = []
            parents = []
            for edge in self.skeleton['E']:
                if node in edge:
                    if edge.index(node) == 0:
                        children.append(edge[1])
                    if edge.index(node) == 1:
                        parents.append(edge[0])

            disc_parents = []
            cont_parents = []
            for parent in parents:
                if self.descriptor['types'][parent] in ['disc', 'disc_num']:
                    disc_parents.append(parent)
                else:
                    cont_parents.append(parent)

            id = self.skeleton['V'].index(node_instance)
            self.skeleton['V'][id].disc_parents = disc_parents
            self.skeleton['V'][id].cont_parents = cont_parents
            self.skeleton['V'][id].children = children

        ordered = gru.toporder(self.skeleton['V'], self.skeleton['E'])
        notOrdered = [node.name for node in self.skeleton['V']]
        mask = [notOrdered.index(name) for name in ordered]
        self.skeleton['V'] = [self.skeleton['V'][i] for i in mask]


class VerticesDefiner(StructureBuilder):
    """
    Main class for defining vertices
    """

    def __init__(self, descriptor: Dict[str, Dict[str, str]],
                 regressor: Optional[object]):
        """
        Automatically creates a list of nodes
        """
        super(VerticesDefiner, self).__init__(descriptor=descriptor)
        # Notice that vertices are used only by Builders
        self.vertices = []

        node = None
        # LEVEL 1: Define a general type of node: Discrete or Gaussian
        for vertex, type in self.descriptor['types'].items():
            if type in ['disc_num', 'disc']:
                node = DiscreteNode(name=vertex)
            elif type == 'cont':
                node = GaussianNode(name=vertex, regressor=regressor)
            else:
                msg = f"""First stage of automatic vertex detection failed on {vertex} due TypeError ({type}).
                Set vertex manually (by calling set_nodes()) or investigate the error."""
                logger_builder.error(msg)
                continue

            self.vertices.append(node)

    def overwrite_vertex(
            self,
            has_logit: bool,
            use_mixture: bool,
            classifier: Optional[Callable],
            regressor: Optional[Callable]):
        """
        Level 2: Redefined nodes according structure (parents)
        :param classifier: an object to pass into logit, condLogit nodes
        :param regressor: an object to pass into gaussianish nodes
        :param has_logit allows edges from cont to disc nodes
        :param use_mixture allows using Mixture
        """
        for node_instance in self.vertices:
            node = node_instance
            if has_logit:
                if 'Discrete' in node_instance.type:
                    if node_instance.cont_parents:
                        if not node_instance.disc_parents:
                            node = LogitNode(
                                name=node_instance.name, classifier=classifier)

                        elif node_instance.disc_parents:
                            node = ConditionalLogitNode(
                                name=node_instance.name, classifier=classifier)

            if use_mixture:
                if 'Gaussian' in node_instance.type:
                    if not node_instance.disc_parents:
                        node = MixtureGaussianNode(
                            name=node_instance.name)
                    elif node_instance.disc_parents:
                        node = ConditionalMixtureGaussianNode(
                            name=node_instance.name)
                    else:
                        continue
            else:
                if 'Gaussian' in node_instance.type:
                    if node_instance.disc_parents:
                        node = ConditionalGaussianNode(
                            name=node_instance.name, regressor=regressor)
                    else:
                        continue

            if node_instance == node:
                continue

            id = self.skeleton['V'].index(node_instance)
            node.disc_parents = node_instance.disc_parents
            node.cont_parents = node_instance.cont_parents
            node.children = node_instance.children
            self.skeleton['V'][id] = node


class EdgesDefiner(StructureBuilder):
    def __init__(self, descriptor: Dict[str, Dict[str, str]]):
        super(EdgesDefiner, self).__init__(descriptor)


class HillClimbDefiner(VerticesDefiner, EdgesDefiner):
    """
    Object to define structure and pass it into skeleton
    """

    def __init__(self, data: DataFrame, descriptor: Dict[str, Dict[str, str]],
                 scoring_function: Union[Tuple[str, Callable], Tuple[str]],
                 regressor: Optional[object] = None):
        """
        :param scoring_function: a tuple with following format (Name, scoring_function)
        """

        self.scoring_function = scoring_function
        self.optimizer = HillClimbSearch(data)
        self.params = {'init_edges': None,
                       'init_nodes': None,
                       'remove_init_edges': True,
                       'white_list': None,
                       'bl_add': None}
        super(HillClimbDefiner, self).__init__(descriptor, regressor=regressor)

    def apply_K2(self,
                 data: DataFrame,
                 init_edges: Optional[List[Tuple[str,
                 str]]],
                 progress_bar: bool,
                 remove_init_edges: bool,
                 white_list: Optional[List[Tuple[str,
                 str]]]):
        """
        :param init_edges: list of tuples, a graph to start learning with
        :param remove_init_edges: allows changes in a model defined by user
        :param data: user's data
        :param progress_bar: verbose regime
        :param white_list: list of allowed edges
        """
        if not all([i in ['disc', 'disc_num']
                    for i in gru.nodes_types(data).values()]):
            logger_builder.error(
                f"K2 deals only with discrete data. Continuous data: {[col for col, type in gru.nodes_types(data).items() if type not in ['disc', 'disc_num']]}")
            return None

        if len(self.scoring_function) != 2:
            from pgmpy.estimators import K2Score
            scoring_function = K2Score
        else:
            scoring_function = self.scoring_function[1]

        if not init_edges:
            best_model = self.optimizer.estimate(
                scoring_method=scoring_function(data),
                black_list=self.black_list,
                white_list=white_list,
                show_progress=progress_bar
            )
        else:

            if remove_init_edges:
                startdag = DAG()
                nodes = [str(v) for v in self.vertices]
                startdag.add_nodes_from(nodes=nodes)
                startdag.add_edges_from(ebunch=init_edges)
                best_model = self.optimizer.estimate(
                    black_list=self.black_list,
                    white_list=white_list,
                    start_dag=startdag,
                    show_progress=False)
            else:
                best_model = self.optimizer.estimate(
                    black_list=self.black_list,
                    white_list=white_list,
                    fixed_edges=init_edges,
                    show_progress=False)

        structure = [list(x) for x in list(best_model.edges())]
        self.skeleton['E'] = structure

    def apply_group1(self,
                     data: DataFrame,
                     progress_bar: bool,
                     init_edges: Optional[List[Tuple[str,
                     str]]],
                     remove_init_edges: bool,
                     white_list: Optional[List[Tuple[str,
                     str]]]):
        """
        This method implements the group of scoring functions.
        Group:
        "MI" - Mutual Information,
        "LL" - Log Likelihood,
        "BIC" - Bayess Information Criteria,
        "AIC" - Akaike information Criteria.
        """
        column_name_dict = dict([(n.name, i)
                                 for i, n in enumerate(self.vertices)])
        blacklist_new = []
        for pair in self.black_list:
            blacklist_new.append(
                (column_name_dict[pair[0]], column_name_dict[pair[1]]))
        if white_list:
            white_list_old = white_list[:]
            white_list = []
            for pair in white_list_old:
                white_list.append(
                    (column_name_dict[pair[0]], column_name_dict[pair[1]]))
        if init_edges:
            init_edges_old = init_edges[:]
            init_edges = []
            for pair in init_edges_old:
                init_edges.append(
                    (column_name_dict[pair[0]], column_name_dict[pair[1]]))

        bn = hc_method(
            data,
            metric=self.scoring_function[0],
            restriction=white_list,
            init_edges=init_edges,
            remove_geo_edges=remove_init_edges,
            black_list=blacklist_new,
            debug=progress_bar)
        structure = []
        nodes = sorted(list(bn.nodes()))
        for rv in nodes:
            for pa in bn.F[rv]['parents']:
                structure.append([list(column_name_dict.keys())[list(column_name_dict.values()).index(
                    pa)], list(column_name_dict.keys())[list(column_name_dict.values()).index(rv)]])
        self.skeleton['E'] = structure


class HCStructureBuilder(HillClimbDefiner):
    """
    Final object with build method
    """

    def __init__(self, data: DataFrame,
                 descriptor: Dict[str, Dict[str, str]],
                 scoring_function: Tuple[str, Callable],
                 regressor: Optional[object],
                 has_logit: bool, use_mixture: bool):
        """
        :param data: train data
        :param descriptor: map for data
        """

        super(
            HCStructureBuilder,
            self).__init__(
            descriptor=descriptor,
            data=data,
            scoring_function=scoring_function,
            regressor=regressor)
        self.use_mixture = use_mixture
        self.has_logit = has_logit

    def build(self, data: DataFrame,
              progress_bar: bool,
              classifier: Optional[object],
              regressor: Optional[object],
              params: Optional[ParamDict] = None):
        if params:
            for param, value in params.items():
                self.params[param] = value

        init_nodes = self.params.pop('init_nodes')
        bl_add = self.params.pop('bl_add')

        # Level 1
        self.skeleton['V'] = self.vertices

        self.restrict(data, init_nodes, bl_add)
        if self.scoring_function[0] == 'K2':
            self.apply_K2(data=data, progress_bar=progress_bar, **self.params)
        elif self.scoring_function[0] in ['MI', 'LL', 'BIC', 'AIC']:
            self.apply_group1(
                data=data,
                progress_bar=progress_bar,
                **self.params)

        # Level 2

        self.get_family()
        self.overwrite_vertex(has_logit=self.has_logit,
                              use_mixture=self.use_mixture,
                              classifier=classifier,
                              regressor=regressor)


class EvoStructureBuilder(VerticesDefiner, EdgesDefiner):
    def __init__(self, data, descriptor, scoring_function, has_logit, use_mixture, regressor):
        super().__init__(data, descriptor, scoring_function, has_logit, use_mixture, regressor)
        self.data = data
        self.descriptor = descriptor
        self.scoring_function = scoring_function
        self.has_logit = has_logit
        self.use_mixture = use_mixture
        self.regressor = regressor
        self.params = {'init_edges': None,
                       'init_nodes': None,
                       'remove_init_edges': True,
                       'white_list': None,
                       'bl_add': None}
        self.default_pop_size = 15
        self.default_crossover_prob = 0.9
        self.default_mutation_prob = 0.8
        self.default_crossovers = [CrossoverTypesEnum.exchange_edges,
                                   CrossoverTypesEnum.exchange_parents_one,
                                   CrossoverTypesEnum.exchange_parents_both]
        self.default_mutations = [evo.custom_mutation_add, evo.custom_mutation_delete, evo.custom_mutation_reverse]
        self.default_max_arity = 10
        self.default_max_depth = 10
        self.default_timeout = 180




    def build(self,
              data: DataFrame,
              custom_metric: Callable = evo.K2_metric,
              **kwargs):
        # Get the list of node names
        nodes_types = data.columns.to_list()

        # Create the initial population
        initial = [evo.CustomGraphModel(nodes=[evo.CustomGraphNode(node_type) for node_type in nodes_types])]

        # Define the requirements for the evolutionary algorithm
        requirements = GraphRequirements(
            max_arity=kwargs.get('max_arity', self.default_max_arity),
            max_depth=kwargs.get('timeout', self.default_max_depth),
            timeout=timedelta(seconds=kwargs.get('timeout', self.default_timeout)))

        # Set the parameters for the evolutionary algorithm
        optimizer_parameters = GPAlgorithmParameters(
            pop_size=kwargs.get('pop_size', self.default_pop_size),
            crossover_prob=kwargs.get('crossover_prob', self.default_crossover_prob),
            mutation_prob=kwargs.get('mutation_prob', self.default_mutation_prob),
            genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
            mutation_types=kwargs.get('custom_mutations', self.default_mutations),
            crossover_types=kwargs.get('custom_crossovers', self.default_mutation_prob),
            selection_types=[SelectionTypesEnum.tournament])

        # Set the adapter for the conversion between the graph and the data structures used by the optimizer
        adapter = DirectAdapter(base_graph_class=evo.CustomGraphModel, base_node_class=evo.CustomGraphNode)

        # Set the constraints for the graph
        constraints = [has_no_self_cycled_nodes, has_no_cycle, evo.has_no_duplicates]
        
        graph_generation_params = GraphGenerationParams(
            adapter=adapter,
            rules_for_constraint=constraints,
            available_node_types=nodes_types)

        # Define the objective function to optimize
        objective = Objective({'custom': custom_metric})

        # Initialize the optimizer
        self.optimizer = EvoGraphOptimizer(
            objective=objective,
            initial_graphs=initial,
            requirements=requirements,
            graph_generation_params=graph_generation_params,
            graph_optimizer_params=optimizer_parameters)

        # Define the function to evaluate the objective function
        objective_eval = ObjectiveEvaluate(objective, data=data, visualisation=False)

        # Run the optimization
        optimized_graphs = self.optimizer.optimise(objective_eval)

        # Get the best graph
        best_graph = adapter.restore(optimized_graphs[0])

        # Convert the best graph to the format used by the Bayesian Network
        self.skeleton = self._convert_to_bn_format(best_graph)
    
    def _convert_to_bn_format(self, graph):
        # Convert the graph to the format used by the Bayesian Network
        # This is a placeholder and should be replaced with the actual conversion code
        return {'V': [], 'E': []}
