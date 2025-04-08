from datetime import timedelta
from typing import Dict, Optional, List, Tuple, Callable

import pandas as pd
from sklearn import preprocessing

import bamt.preprocessors as pp
from bamt.builders.builders_base import BaseDefiner
from bamt.nodes.composite_continuous_node import CompositeContinuousNode
from bamt.nodes.composite_discrete_node import CompositeDiscreteNode
from bamt.nodes.discrete_node import DiscreteNode
from bamt.nodes.gaussian_node import GaussianNode
from bamt.builders.evo_builders.learning_operators import (
    k2_metric,
    has_no_duplicates,
    has_no_blacklist_edges,
    has_only_whitelist_edges,
    custom_mutation_add,
    custom_mutation_delete,
    custom_mutation_reverse,
    CompositeNode as DeapCompositeNode,
    CompositeGraph,
)
from bamt.builders.evo_builders.deap_optimizer import GraphEvolutionOptimizer
from bamt.builders.evo_builders.deap_ml_models import MLModels
from bamt.log import logger_builder


class EvoDefiner(BaseDefiner):
    """
    Object that might take additional methods to decompose structure builder class
    """

    def __init__(
        self,
        data: pd.DataFrame,
        descriptor: Dict[str, Dict[str, str]],
        regressor: Optional[object] = None,
    ):
        super().__init__(data, descriptor, regressor)


class EvoStructureBuilder(EvoDefiner):
    """
    This class uses an evolutionary algorithm based on DEAP to generate a Directed Acyclic Graph (DAG)
    that represents the structure of a Bayesian Network.

    Attributes:
        data (DataFrame): Input data used to build the structure.
        descriptor (dict): Descriptor describing node types and signs.
        regressor (object): A regression model for continuous nodes.
        has_logit (bool): Indicates whether a logit link function should be used.
        use_mixture (bool): Indicates whether a mixture model should be used.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        descriptor: Dict[str, Dict[str, str]],
        regressor: Optional[object],
        has_logit: bool,
        use_mixture: bool,
    ):
        super(EvoStructureBuilder, self).__init__(
            data=data, descriptor=descriptor, regressor=regressor
        )
        self.data = data
        self.descriptor = descriptor
        self.has_logit = has_logit
        self.use_mixture = use_mixture
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
        self.default_num_of_generations = 50
        self.default_early_stopping_iterations = 50
        self.logging_level = 50
        self.objective_metric = k2_metric
        self.verbose = True

    def build(
        self,
        data: pd.DataFrame,
        classifier: Optional[object],
        regressor: Optional[object],
        **kwargs
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
        self.overwrite_vertex(
            has_logit=self.has_logit,
            use_mixture=self.use_mixture,
            classifier=classifier,
            regressor=regressor,
        )

    def search(self, data: pd.DataFrame, **kwargs) -> List[Tuple[str, str]]:
        """
        Executes all the evolutionary computations and returns the best graph's edge list.

        Args:
            data (DataFrame): The data from which to build the structure.

        Returns:
            best_graph_edge_list (List[Tuple[str, str]]): The edge list of the best graph found by the search.
        """
        # Get the list of node names
        nodes_types = data.columns.to_list()

        # Create the initial graph model
        initial_graph = CompositeGraph()
        for node_name in nodes_types:
            node = DeapCompositeNode(
                name=node_name,
                node_type=self.descriptor["types"].get(node_name, "cont"),
            )
            initial_graph.add_node(node)

        # Define constraints for the evolutionary algorithm
        constraints = [has_no_duplicates]

        # Add blacklist/whitelist constraints if provided
        blacklist = kwargs.get("blacklist", None)
        whitelist = kwargs.get("whitelist", None)

        if blacklist:
            constraints.append(lambda g: has_no_blacklist_edges(g, blacklist))
        if whitelist:
            constraints.append(lambda g: has_only_whitelist_edges(g, whitelist))

        # Set up the optimizer
        optimizer = GraphEvolutionOptimizer(
            objective_function=self.objective_metric,
            constraints=constraints,
            data=data,
            population_size=kwargs.get("pop_size", self.default_pop_size),
            generations=kwargs.get(
                "num_of_generations", self.default_num_of_generations
            ),
            tournament_size=3,  # Default tournament size
            crossover_probability=kwargs.get(
                "crossover_prob", self.default_crossover_prob
            ),
            mutation_probability=kwargs.get(
                "mutation_prob", self.default_mutation_prob
            ),
            n_jobs=kwargs.get("n_jobs", self.default_n_jobs),
            early_stopping_rounds=kwargs.get(
                "early_stopping_iterations", self.default_early_stopping_iterations
            ),
            timeout_minutes=kwargs.get("timeout", self.default_timeout),
            maximize=False,  # We want to minimize the objective (negative K2 score)
            verbose=kwargs.get("verbose", self.verbose),
        )

        # Run the optimization
        results = optimizer.optimize([initial_graph])
        best_graph, best_score = results[0]  # Get the best result

        # Get the best graph's edge list
        best_graph_edge_list = best_graph.get_edges()

        return best_graph_edge_list

    @staticmethod
    def _convert_to_strings(nested_list):
        return [tuple([str(item) for item in inner_list]) for inner_list in nested_list]
