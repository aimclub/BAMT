from datetime import timedelta
import random
from typing import Dict, Optional, List, Tuple, Callable

from pandas import DataFrame
from sklearn import preprocessing

import bamt.preprocessors as pp
from bamt.builders.builders_base import VerticesDefiner, EdgesDefiner
from bamt.builders.evo_builders.deap_ml_models import MLModels
from bamt.builders.evo_builders.deap_optimizer import GraphEvolutionOptimizer
from bamt.log import logger_builder
from bamt.nodes.composite_continuous_node import CompositeContinuousNode
from bamt.nodes.composite_discrete_node import CompositeDiscreteNode
from bamt.nodes.discrete_node import DiscreteNode
from bamt.nodes.gaussian_node import GaussianNode
from bamt.builders.evo_builders.learning_operators import (
    composite_metric,
    CompositeGraph,
    CompositeNode as DeapCompositeNode,
)


class CompositeDefiner(VerticesDefiner, EdgesDefiner):
    """
    Object that might take additional methods to decompose structure builder class
    """

    def __init__(
        self,
        descriptor: Dict[str, Dict[str, str]],
        regressor: Optional[object] = None,
    ):
        super().__init__(descriptor, regressor)

        # Notice that vertices are used only by Builders
        self.vertices = []

        # LEVEL 1: Define a general type of node: Discrete or Сontinuous
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
    This class uses an evolutionary algorithm based on DEAP to generate a Directed Acyclic Graph (DAG) that represents
    the structure of a Composite Bayesian Network.

    Attributes:
        data (DataFrame): Input data used to build the structure.
        descriptor (dict): Descriptor describing node types and signs.
    """

    def __init__(
        self,
        data: DataFrame,
        descriptor: Dict[str, Dict[str, str]],
        regressor: Optional[object],
    ):
        super(CompositeStructureBuilder, self).__init__(
            descriptor=descriptor, regressor=regressor
        )
        self.data = data
        self.parent_models_dict = {}
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
        self.default_num_of_generations = 50
        self.default_early_stopping_iterations = 50
        self.objective_metric = composite_metric
        self.verbose = True
        self.logging_level = 50

    def overwrite_vertex(
        self,
        regressor: Optional[Callable],
    ):
        for node_instance in self.vertices:
            node = node_instance
            if (
                len(node_instance.cont_parents + node_instance.disc_parents) < 1
                and type(node_instance).__name__ == "CompositeContinuousNode"
            ):
                node = GaussianNode(name=node_instance.name, regressor=regressor)
            elif (
                len(node_instance.cont_parents + node_instance.disc_parents) < 1
                and type(node_instance).__name__ == "CompositeDiscreteNode"
            ):
                node = DiscreteNode(name=node_instance.name)
            else:
                continue
            id_node = self.skeleton["V"].index(node_instance)
            node.disc_parents = node_instance.disc_parents
            node.cont_parents = node_instance.cont_parents
            node.children = node_instance.children
            self.skeleton["V"][id_node] = node

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
        best_graph_edge_list, parent_models = self.search(data, **kwargs)

        # Convert the best graph to the format used by the Bayesian Network
        self.skeleton["V"] = self.vertices
        self.skeleton["E"] = best_graph_edge_list
        self.parent_models_dict = parent_models

        self.get_family()
        self.overwrite_vertex(regressor=regressor)

    def search(self, data: DataFrame, **kwargs) -> [List[Tuple[str, str]], Dict]:
        """
        Executes all the evolutionary computations and returns the best graph's edge list.

        Args:
            data (DataFrame): The data from which to build the structure.

        Returns:
            best_graph_edge_list (List[Tuple[str, str]]): The edge list of the best graph found by the search.
            parent_models (Dict): Dictionary mapping node names to their parent models
        """
        # Get the list of node names
        vertices = list(data.columns)

        encoder = preprocessing.LabelEncoder()
        p = pp.Preprocessor([("encoder", encoder)])
        preprocessed_data, _ = p.apply(data)

        # Create the initial composite graph
        initial_graph = CompositeGraph()
        for vertex in vertices:
            node_type = p.nodes_types[vertex]
            node = DeapCompositeNode(name=vertex, node_type=node_type)
            initial_graph.add_node(node)

        # Define mutations and constraints for the evolutionary algorithm
        def custom_mutation_add_model(graph):
            """Add a model to a random node with parents."""
            try:
                nodes_with_parents = [node for node in graph.nodes if node.parents]
                if nodes_with_parents:
                    node = random.choice(nodes_with_parents)
                    ml_models = MLModels()
                    node.content["parent_model"] = ml_models.get_model_by_children_type(
                        node
                    )
            except Exception as e:
                print(f"Error in custom_mutation_add_model: {e}")
            return graph

        # Set up the optimizer
        optimizer = GraphEvolutionOptimizer(
            objective_function=composite_metric,
            constraints=[],  # No specific constraints for composite model
            data=preprocessed_data,
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
            maximize=False,  # Minimize negative score
            verbose=kwargs.get("verbose", self.verbose),
        )

        # Run the optimization
        results = optimizer.optimize([initial_graph])
        best_graph, _ = results[0]  # Get the best result

        # Extract parent models from the best graph
        parent_models = {}
        for node in best_graph.nodes:
            parent_models[node.name] = node.content.get("parent_model")

        # Get the best graph's edge list
        best_graph_edge_list = best_graph.get_edges()

        return best_graph_edge_list, parent_models

    @staticmethod
    def _convert_to_strings(nested_list):
        return [[str(item) for item in inner_list] for inner_list in nested_list]

    @staticmethod
    def _get_parent_models(graph):
        parent_models = {}
        for node in graph.nodes:
            parent_models[node.content["name"]] = node.content.get("parent_model")
        return parent_models
