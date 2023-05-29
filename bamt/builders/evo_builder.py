from datetime import timedelta

from pandas import DataFrame

from bamt.builders.builders_base import BaseDefiner
from bamt.utils import EvoUtils as evo

from golem.core.adapter import DirectAdapter
from golem.core.dag.verification_rules import has_no_cycle, has_no_self_cycled_nodes
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.objective import Objective, ObjectiveEvaluate
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams
from golem.core.optimisers.genetic.operators.selection import SelectionTypesEnum

from typing import Dict, Optional


class EvoDefiner(BaseDefiner):
    """
    Object that might take additional methods to decompose structure builder class
    """
    def __init__(self, data: DataFrame, descriptor: Dict[str, Dict[str, str]],
                 regressor: Optional[object] = None):

        super().__init__(data, descriptor, regressor)


class EvoStructureBuilder(EvoDefiner):
    """
    This class uses an evolutionary algorithm based on GOLEM to generate a Directed Acyclic Graph (DAG) that represents
    the structure of a Bayesian Network.

    Attributes:
        data (DataFrame): Input data used to build the structure.
        descriptor (dict): Descriptor describing node types and signs.
        regressor (object): A regression model for continuous nodes.
        has_logit (bool): Indicates whether a logit link function should be used.
        use_mixture (bool): Indicates whether a mixture model should be used.
    """
    def __init__(
            self,
            data: DataFrame,
            descriptor: Dict[str, Dict[str, str]],
            regressor: Optional[object],
            has_logit: bool,
            use_mixture: bool):
        super(
            EvoStructureBuilder,
            self).__init__(
            data=data,
            descriptor=descriptor,
            regressor=regressor)
        self.data = data
        self.descriptor = descriptor
        self.has_logit = has_logit
        self.use_mixture = use_mixture
        self.regressor = regressor
        self.params = {'init_edges': None,
                       'init_nodes': None,
                       'remove_init_edges': True,
                       'white_list': None,
                       'bl_add': None}
        self.default_n_jobs = -1
        self.default_pop_size = 15
        self.default_crossover_prob = 0.9
        self.default_mutation_prob = 0.8
        self.default_max_arity = 100
        self.default_max_depth = 100
        self.default_timeout = 180
        self.objective_metric = evo.K2_metric
        self.default_crossovers = [CrossoverTypesEnum.exchange_edges,
                                   CrossoverTypesEnum.exchange_parents_one,
                                   CrossoverTypesEnum.exchange_parents_both]
        self.default_mutations = [
            evo.custom_mutation_add,
            evo.custom_mutation_delete,
            evo.custom_mutation_reverse]
        self.default_selection = [SelectionTypesEnum.tournament]
        self.default_constraints = [
            has_no_self_cycled_nodes,
            has_no_cycle,
            evo.has_no_duplicates]

    def build(self,
              data: DataFrame,
              classifier: Optional[object],
              regressor: Optional[object],
              **kwargs):
        """
           Builds the structure of a Bayesian network from the given data using an evolutionary algorithm.

           Args:
               data (DataFrame): The data from which to build the structure.
               classifier (Optional[object]): A classification model for discrete nodes.
               regressor (Optional[object]): A regression model for continuous nodes.

           Additional optional parameters can be provided as keyword arguments (kwargs) to customize the evolutionary
           algorithm used to generate the structure. These include parameters to control the size of the population,
           the probabilities of crossover and mutation, constraints on the structure of the graph, and many others.

           The resulting structure is stored in the `skeleton` attribute of the `EvoStructureBuilder` object.
        """
        # Get the list of node names
        nodes_types = data.columns.to_list()

        # Create the initial population
        initial = [
            evo.CustomGraphModel(
                nodes=kwargs.get(
                    'init_nodes', [
                        evo.CustomGraphNode(node_type) for node_type in nodes_types]))]

        # Define the requirements for the evolutionary algorithm
        requirements = GraphRequirements(
            max_arity=kwargs.get(
                'max_arity', self.default_max_arity), max_depth=kwargs.get(
                'max_depth', self.default_max_depth),
            timeout=timedelta(
                minutes=kwargs.get(
                    'timeout', self.default_timeout)),
            n_jobs=kwargs.get('n_jobs', self.default_n_jobs))

        # Set the parameters for the evolutionary algorithm
        optimizer_parameters = GPAlgorithmParameters(
            pop_size=kwargs.get('pop_size', self.default_pop_size),
            crossover_prob=kwargs.get('crossover_prob', self.default_crossover_prob),
            mutation_prob=kwargs.get('mutation_prob', self.default_mutation_prob),
            genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
            mutation_types=kwargs.get('custom_mutations', self.default_mutations),
            crossover_types=kwargs.get('custom_crossovers', self.default_crossovers),
            selection_types=kwargs.get('selection_type', self.default_selection))

        # Set the adapter for the conversion between the graph and the data
        # structures used by the optimizer
        adapter = DirectAdapter(
            base_graph_class=evo.CustomGraphModel,
            base_node_class=evo.CustomGraphNode)

        # Set the constraints for the graph

        if kwargs.get('blacklist', None) is not None:
            self.default_constraints.append(evo.has_no_blacklist_edges)
        if kwargs.get('whitelist', None) is not None:
            self.default_constraints.append(evo.has_only_whitelist_edges)

        constraints = kwargs.get('custom_constraints', [])

        constraints.append(self.default_constraints)

        if kwargs.get('blacklist', None) is not None:
            constraints.append(evo.has_no_blacklist_edges)
        if kwargs.get('whitelist', None) is not None:
            constraints.append(evo.has_only_whitelist_edges)

        graph_generation_params = GraphGenerationParams(
            adapter=adapter,
            rules_for_constraint=constraints,
            available_node_types=nodes_types)

        # Define the objective function to optimize
        objective = Objective({'custom': kwargs.get(
            'custom_metric', self.objective_metric)})

        # Initialize the optimizer
        optimizer = EvoGraphOptimizer(
            objective=objective,
            initial_graphs=initial,
            requirements=requirements,
            graph_generation_params=graph_generation_params,
            graph_optimizer_params=optimizer_parameters)

        # Define the function to evaluate the objective function
        objective_eval = ObjectiveEvaluate(
            objective, data=data)

        # Run the optimization
        optimized_graph = optimizer.optimise(objective_eval)[0]

        # Get the best graph
        best_graph_edge_list = optimized_graph.operator.get_edges()
        best_graph_edge_list = self._convert_to_strings(best_graph_edge_list)

        # Convert the best graph to the format used by the Bayesian Network
        self.skeleton['V'] = self.vertices
        self.skeleton['E'] = best_graph_edge_list

        self.get_family()
        self.overwrite_vertex(has_logit=self.has_logit,
                              use_mixture=self.use_mixture,
                              classifier=classifier,
                              regressor=regressor)

    @staticmethod
    def _convert_to_strings(nested_list):
        return [[str(item) for item in inner_list]
                for inner_list in nested_list]
