import itertools

from pgmpy.base import DAG
from pgmpy.estimators import HillClimbSearch
from bamt.redef_HC import hc as hc_method

from bamt.nodes import *
from bamt.log import logger_builder
from pandas import DataFrame
from bamt.utils import GraphUtils as gru

from typing import Dict, List, Optional, Tuple, Callable, TypedDict, Union, Sequence


class ParamDict(TypedDict, total=False):
    init_edges: Optional[Sequence[str, str]]
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
                node = discrete_node.DiscreteNode(name=vertex)
            elif type == 'cont':
                Node = nodes.GaussianNode(name=vertex, regressor=regressor)
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
            Node = node_instance
            if has_logit:
                if 'Discrete' in node_instance.type:
                    if node_instance.cont_parents:
                        if not node_instance.disc_parents:
                            Node = logit_node.LogitNode(
                                name=node_instance.name, classifier=classifier)

                        elif node_instance.disc_parents:
                            Node = conditional_logit_node.ConditionalLogitNode(
                                name=node_instance.name, classifier=classifier)

            if use_mixture:
                if 'Gaussian' in node_instance.type:
                    if not node_instance.disc_parents:
                        Node = mixture_gaussian_node.MixtureGaussianNode(
                            name=node_instance.name)
                    elif node_instance.disc_parents:
                        Node = conditional_mixture_gaussian_node.ConditionalMixtureGaussianNode(
                            name=node_instance.name)
                    else:
                        continue
            else:
                if 'Gaussian' in node_instance.type:
                    if node_instance.disc_parents:
                        Node = nodes.ConditionalGaussianNode(
                            name=node_instance.name, regressor=regressor)
                    else:
                        continue

            if node_instance == Node:
                continue

            id = self.skeleton['V'].index(node_instance)
            Node.disc_parents = node_instance.disc_parents
            Node.cont_parents = node_instance.cont_parents
            Node.children = node_instance.children
            self.skeleton['V'][id] = Node


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
        import bamt.utils.GraphUtils as gru
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
        # (score == "MI") | (score == "LL") | (score == "BIC") | (score == "AIC")
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
