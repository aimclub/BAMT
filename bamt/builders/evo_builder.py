from datetime import timedelta

from pandas import DataFrame

from bamt.builders.builders_base import VerticesDefiner, EdgesDefiner
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

from typing import Callable


class EvoStructureBuilder(VerticesDefiner, EdgesDefiner):
    def __init__(
            self,
            data,
            descriptor,
            scoring_function,
            has_logit,
            use_mixture,
            regressor):
        super().__init__(
            data=data,
            descriptor=descriptor,
            scoring_function=scoring_function,
            regressor=regressor)
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
        self.default_max_arity = 10
        self.default_max_depth = 10
        self.default_timeout = 180
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
        self.objective_metric = evo.K2_metric

    def build(self,
              data: DataFrame,
              **kwargs):
        # Get the list of node names
        nodes_types = data.columns.to_list()

        # Create the initial population
        initial = [
            evo.CustomGraphModel(
                nodes=[
                    evo.CustomGraphNode(node_type) for node_type in nodes_types])]

        # Define the requirements for the evolutionary algorithm
        requirements = GraphRequirements(
            max_arity=kwargs.get(
                'max_arity', self.default_max_arity), max_depth=kwargs.get(
                'timeout', self.default_max_depth), timeout=timedelta(
                seconds=kwargs.get(
                    'timeout', self.default_timeout)))

        # Set the parameters for the evolutionary algorithm
        optimizer_parameters = GPAlgorithmParameters(
            pop_size=kwargs.get('pop_size', self.default_pop_size),
            crossover_prob=kwargs.get('crossover_prob', self.default_crossover_prob),
            mutation_prob=kwargs.get('mutation_prob', self.default_mutation_prob),
            genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
            mutation_types=kwargs.get('custom_mutations', self.default_mutations),
            crossover_types=kwargs.get('custom_crossovers', self.default_mutation_prob),
            selection_types=kwargs.get('selection_type', self.default_selection))

        # Set the adapter for the conversion between the graph and the data
        # structures used by the optimizer
        adapter = DirectAdapter(
            base_graph_class=evo.CustomGraphModel,
            base_node_class=evo.CustomGraphNode)

        # Set the constraints for the graph
        constraints = kwargs.get(
            'custom_constraints',
            self.default_constraints)

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
            objective, data=data, visualisation=False)

        # Run the optimization
        optimized_graph = optimizer.optimise(objective_eval)[0]

        # Get the best graph
        best_graph = adapter.restore(optimized_graph)

        # Convert the best graph to the format used by the Bayesian Network
        self.skeleton = self._convert_to_bn_format(best_graph)

        self.get_family()
        self.overwrite_vertex(has_logit=self.has_logit,
                              use_mixture=self.use_mixture,
                              classifier=classifier,
                              regressor=regressor)

    def _convert_to_bn_format(self, graph):
        # Convert the graph to the format used by the Bayesian Network
        # This is a placeholder and should be replaced with the actual
        # conversion code
        return {'V': [self.vertices], 'E': [graph]}
