from datetime import timedelta

from pandas import DataFrame
from sklearn import preprocessing
import bamt.preprocessors as pp

from bamt.builders.builders_base import BaseDefiner
from bamt.log import logger_builder
from bamt.utils.composite_utils import CompositeGeneticOperators
from bamt.utils.composite_utils.CompositeModel import CompositeModel, CompositeNode
from bamt.nodes.composite_discrete_node import CompositeDiscreteNode
from bamt.nodes.composite_continuous_node import CompositeContinuousNode
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

from typing import Dict, Optional, List, Tuple


class CompositeDefiner(BaseDefiner):
    """
    Object that might take additional methods to decompose structure builder class
    """

    def __init__(
        self,
        data: DataFrame,
        descriptor: Dict[str, Dict[str, str]],
        regressor: Optional[object] = None,
    ):
        super().__init__(data, descriptor, regressor)

        # Notice that vertices are used only by Builders
        self.vertices = []

        # LEVEL 1: Define a general type of node: Discrete or Gaussian
        for vertex, type in self.descriptor["types"].items():
            if type in ["disc_num", "disc"]:
                node = CompositeDiscreteNode(name=vertex)
            elif type == "cont":
                node = CompositeContinuousNode(name=vertex, regressor=regressor)
            else:
                msg = f"""First stage of automatic vertex detection failed on {vertex} due TypeError ({type}).
                Set vertex manually (by calling set_nodes()) or investigate the error."""
                logger_builder.error(msg)
                continue

            self.vertices.append(node)


class CompositeStructureBuilder(CompositeDefiner):
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
    ):
        super(CompositeStructureBuilder, self).__init__(
            data=data, descriptor=descriptor, regressor=regressor
        )
        self.data = data
        self.descriptor = descriptor
        self.regressor = regressor
        self.params = {
            "init_edges": None,
            "init_nodes": None,
            "remove_init_edges": True,
            "white_list": None,
            "bl_add": None,
        }
        self.default_n_jobs = -1
        self.default_pop_size = 15
        self.default_crossover_prob = 0.9
        self.default_mutation_prob = 0.8
        self.default_max_arity = 100
        self.default_max_depth = 100
        self.default_timeout = 180
        self.objective_metric = CompositeGeneticOperators.composite_metric
        self.default_crossovers = [
            CrossoverTypesEnum.exchange_edges,
            CrossoverTypesEnum.exchange_parents_one,
            CrossoverTypesEnum.exchange_parents_both,
            CompositeGeneticOperators.custom_crossover_all_model,
        ]
        self.default_mutations = [
            CompositeGeneticOperators.custom_mutation_add_structure,
            CompositeGeneticOperators.custom_mutation_delete_structure,
            CompositeGeneticOperators.custom_mutation_reverse_structure,
            CompositeGeneticOperators.custom_mutation_add_model,
        ]
        self.default_selection = [SelectionTypesEnum.tournament]
        self.default_constraints = [
            has_no_self_cycled_nodes,
            has_no_cycle,
            evo.has_no_duplicates,
        ]
        self.default_history_directory = None

    def build(
        self,
        data: DataFrame,
        classifier: Optional[object],
        regressor: Optional[object],
        **kwargs,
    ):
        """
        Calls the search method to execute all the evolutionary computations.

        Args:
            data (DataFrame): The data from which to build the structure.
            classifier (Optional[object]): A classification model for discrete nodes.
            regressor (Optional[object]): A regression model for continuous nodes.
        """
        best_graph_edge_list = self.search(data, **kwargs)

        # Convert the best graph to the format used by the Bayesian Network
        self.skeleton["V"] = self.vertices
        self.skeleton["E"] = best_graph_edge_list

        self.get_family()

    def search(self, data: DataFrame, **kwargs) -> List[Tuple[str, str]]:
        """
        Executes all the evolutionary computations and returns the best graph's edge list.

        Args:
            data (DataFrame): The data from which to build the structure.

        Returns:
            best_graph_edge_list (List[Tuple[str, str]]): The edge list of the best graph found by the search.
        """
        # Get the list of node names
        vertices = list(data.columns)

        encoder = preprocessing.LabelEncoder()
        p = pp.Preprocessor([("encoder", encoder)])
        discretized_data, _ = p.apply(data)

        # Create the initial population
        initial = [
            CompositeModel(
                nodes=[
                    CompositeNode(
                        nodes_from=None,
                        content={
                            "name": vertex,
                            "type": p.nodes_types[vertex],
                            "parent_model": None,
                        },
                    )
                    for vertex in vertices
                ]
            )
        ]

        objective = Objective({"custom": CompositeGeneticOperators.composite_metric})
        objective_eval = ObjectiveEvaluate(objective, data=discretized_data)

        # Define the requirements for the evolutionary algorithm
        requirements = GraphRequirements(
            max_arity=kwargs.get("max_arity", self.default_max_arity),
            max_depth=kwargs.get("max_depth", self.default_max_depth),
            timeout=timedelta(minutes=kwargs.get("timeout", self.default_timeout)),
            history_dir=kwargs.get("history_dir", None),
            early_stopping_iterations=kwargs.get("early_stopping_iterations", None),
            n_jobs=kwargs.get("n_jobs", self.default_n_jobs),
        )

        # Set the parameters for the evolutionary algorithm
        optimizer_parameters = GPAlgorithmParameters(
            pop_size=kwargs.get("pop_size", self.default_pop_size),
            crossover_prob=kwargs.get("crossover_prob", self.default_crossover_prob),
            mutation_prob=kwargs.get("mutation_prob", self.default_mutation_prob),
            genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
            mutation_types=kwargs.get("custom_mutations", self.default_mutations),
            crossover_types=kwargs.get("custom_crossovers", self.default_crossovers),
            selection_types=kwargs.get("selection_type", self.default_selection),
        )

        # Set the adapter for the conversion between the graph and the data
        # structures used by the optimizer
        adapter = DirectAdapter(
            base_graph_class=CompositeModel, base_node_class=CompositeNode
        )

        # Set the constraints for the graph

        constraints = kwargs.get("custom_constraints", [])

        constraints.extend(self.default_constraints)

        if kwargs.get("blacklist", None) is not None:
            constraints.append(evo.has_no_blacklist_edges)
        if kwargs.get("whitelist", None) is not None:
            constraints.append(evo.has_only_whitelist_edges)

        graph_generation_params = GraphGenerationParams(
            adapter=adapter, rules_for_constraint=constraints
        )

        # Define the objective function to optimize
        objective = Objective(
            {"custom": kwargs.get("custom_metric", self.objective_metric)}
        )

        # Initialize the optimizer
        optimizer = EvoGraphOptimizer(
            objective=objective,
            initial_graphs=initial,
            requirements=requirements,
            graph_generation_params=graph_generation_params,
            graph_optimizer_params=optimizer_parameters,
        )

        # Define the function to evaluate the objective function
        objective_eval = ObjectiveEvaluate(objective, data=discretized_data)

        # Run the optimization
        optimized_graph = optimizer.optimise(objective_eval)[0]

        # Get the best graph
        best_graph_edge_list = optimized_graph.operator.get_edges()
        best_graph_edge_list = self._convert_to_strings(best_graph_edge_list)

        return best_graph_edge_list

    @staticmethod
    def _convert_to_strings(nested_list):
        return [[str(item) for item in inner_list] for inner_list in nested_list]
