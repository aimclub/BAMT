"""
GOLEM-based evolutionary optimizer for Bayesian Network structure learning.

This module implements structure learning using the GOLEM evolutionary framework,
which uses genetic algorithms to search the space of DAG structures.
"""

from datetime import timedelta
from typing import List, Optional, Tuple

import networkx as nx
import pandas as pd
from golem.core.adapter import DirectAdapter
from golem.core.dag.verification_rules import has_no_cycle, has_no_self_cycled_nodes
from golem.core.log import Log
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.genetic.operators.selection import SelectionTypesEnum
from golem.core.optimisers.objective import Objective, ObjectiveEvaluate
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams

from bamt.score_functions import ScoreFunction
from bamt.utils import evo_utils as evo

from .score_dag_optimizer import ScoreDAGOptimizer


class GOLEMOptimizer(ScoreDAGOptimizer):
    """
    GOLEM-based evolutionary structure optimizer.

    Uses genetic algorithms to search for optimal DAG structures. The algorithm:
    1. Starts with initial population of graphs
    2. Applies genetic operators (crossover, mutation, selection)
    3. Evaluates fitness using scoring function
    4. Evolves population over generations

    Supports blacklist/whitelist constraints and various customization options.

    Attributes:
        score_function: The scoring function to optimize
        n_jobs: Number of parallel jobs (default: -1 for all CPUs)
        pop_size: Population size (default: 15)
        crossover_prob: Crossover probability (default: 0.9)
        mutation_prob: Mutation probability (default: 0.8)
        max_arity: Maximum number of parents per node (default: 100)
        max_depth: Maximum graph depth (default: 100)
        timeout: Timeout in minutes (default: 180)
        num_of_generations: Number of generations (default: 50)
        early_stopping_iterations: Early stopping patience (default: 50)
        verbose: Whether to print optimization progress
    """

    def __init__(
        self,
        score_function: Optional[ScoreFunction] = None,
        n_jobs: int = -1,
        pop_size: int = 15,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.8,
        max_arity: int = 100,
        max_depth: int = 100,
        timeout: int = 180,
        num_of_generations: int = 50,
        early_stopping_iterations: int = 50,
        verbose: bool = True,
    ):
        """
        Initialize GOLEM optimizer.

        Args:
            score_function: Scoring function (uses K2 by default)
            n_jobs: Number of parallel jobs
            pop_size: Population size
            crossover_prob: Crossover probability
            mutation_prob: Mutation probability
            max_arity: Maximum parents per node
            max_depth: Maximum graph depth
            timeout: Timeout in minutes
            num_of_generations: Number of generations
            early_stopping_iterations: Early stopping patience
            verbose: Print optimization progress
        """
        super().__init__()
        self.score_function = score_function
        self.n_jobs = n_jobs
        self.pop_size = pop_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.max_arity = max_arity
        self.max_depth = max_depth
        self.timeout = timeout
        self.num_of_generations = num_of_generations
        self.early_stopping_iterations = early_stopping_iterations
        self.verbose = verbose

        # Default genetic operators
        self.default_crossovers = [
            CrossoverTypesEnum.exchange_edges,
            CrossoverTypesEnum.exchange_parents_one,
            CrossoverTypesEnum.exchange_parents_both,
        ]
        self.default_mutations = [
            evo.custom_mutation_add,
            evo.custom_mutation_delete,
            evo.custom_mutation_reverse,
        ]
        self.default_selection = [SelectionTypesEnum.tournament]
        self.default_constraints = [
            has_no_self_cycled_nodes,
            has_no_cycle,
            evo.has_no_duplicates,
        ]

    def optimize(
        self,
        data: pd.DataFrame,
        node_names: Optional[List[str]] = None,
        **kwargs
    ) -> nx.DiGraph:
        """
        Find the optimal DAG structure using GOLEM evolutionary algorithm.

        Args:
            data: The dataset to learn structure from
            node_names: Names of nodes (defaults to column names)
            **kwargs: Additional parameters:
                - init_nodes: Initial nodes for starting graph
                - blacklist: List of forbidden edges
                - whitelist: List of allowed edges
                - custom_mutations: Custom mutation operators
                - custom_crossovers: Custom crossover operators
                - custom_constraints: Custom constraint functions
                - custom_metric: Custom scoring metric
                - max_arity: Override max parents
                - max_depth: Override max depth
                - num_of_generations: Override generations
                - timeout: Override timeout
                - early_stopping_iterations: Override early stopping
                - n_jobs: Override parallel jobs
                - pop_size: Override population size
                - crossover_prob: Override crossover probability
                - mutation_prob: Override mutation probability

        Returns:
            nx.DiGraph: The learned DAG structure
        """
        if node_names is None:
            node_names = list(data.columns)

        # Create the initial population
        initial = [
            evo.CustomGraphModel(
                nodes=kwargs.get(
                    "init_nodes",
                    [evo.CustomGraphNode(node_type) for node_type in node_names],
                )
            )
        ]

        # Define the requirements for the evolutionary algorithm
        requirements = GraphRequirements(
            max_arity=kwargs.get("max_arity", self.max_arity),
            max_depth=kwargs.get("max_depth", self.max_depth),
            num_of_generations=kwargs.get("num_of_generations", self.num_of_generations),
            timeout=timedelta(minutes=kwargs.get("timeout", self.timeout)),
            early_stopping_iterations=kwargs.get(
                "early_stopping_iterations", self.early_stopping_iterations
            ),
            n_jobs=kwargs.get("n_jobs", self.n_jobs),
        )

        # Set the parameters for the evolutionary algorithm
        optimizer_parameters = GPAlgorithmParameters(
            pop_size=kwargs.get("pop_size", self.pop_size),
            crossover_prob=kwargs.get("crossover_prob", self.crossover_prob),
            mutation_prob=kwargs.get("mutation_prob", self.mutation_prob),
            genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
            mutation_types=kwargs.get("custom_mutations", self.default_mutations),
            crossover_types=kwargs.get("custom_crossovers", self.default_crossovers),
            selection_types=kwargs.get("selection_type", self.default_selection),
        )

        # Set the adapter for the conversion between the graph and the data
        # structures used by the optimizer
        adapter = DirectAdapter(
            base_graph_class=evo.CustomGraphModel, base_node_class=evo.CustomGraphNode
        )

        # Set the constraints for the graph
        constraints = kwargs.get("custom_constraints", [])
        constraints.extend(self.default_constraints)

        if kwargs.get("blacklist", None) is not None:
            constraints.append(evo.has_no_blacklist_edges)
        if kwargs.get("whitelist", None) is not None:
            constraints.append(evo.has_only_whitelist_edges)

        graph_generation_params = GraphGenerationParams(
            adapter=adapter,
            rules_for_constraint=constraints,
            available_node_types=node_names,
        )

        # Define the objective function to optimize
        # Use K2 metric by default if no custom metric provided
        metric = kwargs.get("custom_metric", evo.K2_metric)
        objective = Objective({"custom": metric})

        # Initialize the optimizer
        optimizer = EvoGraphOptimizer(
            objective=objective,
            initial_graphs=initial,
            requirements=requirements,
            graph_generation_params=graph_generation_params,
            graph_optimizer_params=optimizer_parameters,
        )

        # Define the function to evaluate the objective function
        objective_eval = ObjectiveEvaluate(objective, data=data)

        if not kwargs.get("verbose", self.verbose):
            Log().reset_logging_level(logging_level=50)

        # Run the optimization
        optimized_graph = optimizer.optimise(objective_eval)[0]

        # Get the best graph edge list
        best_graph_edge_list = optimized_graph.operator.get_edges()
        best_graph_edge_list = self._convert_to_strings(best_graph_edge_list)

        # Create NetworkX DiGraph
        graph = nx.DiGraph()
        graph.add_nodes_from(node_names)
        graph.add_edges_from(best_graph_edge_list)

        return graph

    @staticmethod
    def _convert_to_strings(nested_list: List[Tuple]) -> List[Tuple[str, str]]:
        """Convert nested list of edges to string tuples."""
        return [tuple([str(item) for item in inner_list]) for inner_list in nested_list]
