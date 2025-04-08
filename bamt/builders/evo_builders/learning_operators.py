import random
from typing import List, Tuple, Dict, Any, Callable
import pandas as pd
import numpy as np
from pgmpy.estimators import K2
from pgmpy.models import BayesianNetwork
import networkx as nx
from sklearn.metrics import root_mean_squared_error
from scipy.stats import norm

from bamt.builders.evo_builders.deap_graph import Graph, Node


def k2_metric(graph: Graph, data: pd.DataFrame) -> float:
    """
    Compute K2 score for a graph structure using pgmpy.
    Higher score is better, but we return negative for minimization.
    """
    try:
        # Convert graph to edge list format expected by pgmpy
        edges = []
        for node in graph.nodes:
            for child in node.children:
                edges.append((node.name, child.name))

        # Create Bayesian Network model
        model = BayesianNetwork(edges)

        # Add nodes (columns) to the model
        model.add_nodes_from(data.columns)

        # Compute K2 score
        k2 = K2(data)
        score = k2.score(model)

        return -score  # Negate for minimization
    except Exception as e:
        print(f"Error computing K2 score: {e}")
        return float("inf")  # Return very poor score for invalid structures


def has_no_duplicates(graph: Graph) -> bool:
    """Verify that there are no duplicate node names in the graph."""
    node_names = [node.name for node in graph.nodes]
    return len(node_names) == len(set(node_names))


def has_no_blacklist_edges(graph: Graph, blacklist: List[Tuple[str, str]]) -> bool:
    """Verify that the graph has no edges in the blacklist."""
    for node in graph.nodes:
        for child in node.children:
            if (node.name, child.name) in blacklist:
                return False
    return True


def has_only_whitelist_edges(graph: Graph, whitelist: List[Tuple[str, str]]) -> bool:
    """Verify that all edges in the graph are in the whitelist."""
    for node in graph.nodes:
        for child in node.children:
            if (node.name, child.name) not in whitelist:
                return False
    return True


# Custom mutation operators for graph evolution


def custom_mutation_add(graph: Graph) -> Graph:
    """Add a random edge to the graph if it doesn't create a cycle."""
    if len(graph.nodes) < 2:
        return graph  # Not enough nodes to add an edge

    # Try multiple times to find a valid edge to add
    for _ in range(100):
        # Select random source and target nodes
        source = random.choice(graph.nodes)
        target = random.choice([n for n in graph.nodes if n != source])

        # Check if edge already exists
        if target in source.children:
            continue

        # Create temporary graph to check for cycles
        temp_graph = graph.copy()
        temp_graph.add_edge(source, target)

        # If adding this edge doesn't create a cycle, add it to the original graph
        if not temp_graph.has_cycle():
            graph.add_edge(source, target)
            break

    return graph


def custom_mutation_delete(graph: Graph) -> Graph:
    """Delete a random edge from the graph."""
    # Get all edges in the graph
    edges = [(node, child) for node in graph.nodes for child in node.children]
    if not edges:
        return graph  # No edges to delete

    # Select a random edge and remove it
    source, target = random.choice(edges)
    graph.remove_edge(source, target)

    return graph


def custom_mutation_reverse(graph: Graph) -> Graph:
    """Reverse a random edge in the graph if it doesn't create a cycle."""
    # Get all edges in the graph
    edges = [(node, child) for node in graph.nodes for child in node.children]
    if not edges:
        return graph  # No edges to reverse

    # Try multiple times to find a valid edge to reverse
    for _ in range(100):
        # Select a random edge
        if not edges:
            break
        source, target = random.choice(edges)
        edges.remove((source, target))

        # Create temporary graph to check for cycles
        temp_graph = graph.copy()
        temp_graph.remove_edge(source, target)
        temp_graph.add_edge(target, source)

        # If reversing this edge doesn't create a cycle, do it in the original graph
        if not temp_graph.has_cycle():
            graph.remove_edge(source, target)
            graph.add_edge(target, source)
            break

    return graph


# Composite model utilities


class CompositeNode(Node):
    """Extension of Node with support for model association."""

    def __init__(self, name: str, node_type: str = None, parent_model: Any = None):
        super().__init__(name)
        self.content = {"name": name, "type": node_type, "parent_model": parent_model}


class CompositeGraph(Graph):
    """Extension of Graph for composite models."""

    def get_nodes_by_type(self, node_type: str) -> List[CompositeNode]:
        """Get all nodes of a specific type."""
        return [
            node
            for node in self.nodes
            if "type" in node.content and node.content["type"] == node_type
        ]


def composite_metric(graph: CompositeGraph, data: pd.DataFrame) -> float:
    """
    Calculate a composite metric for evaluating the graph structure.
    This is a more complex metric that considers both structure and fitted models.
    """
    # Split data for training and testing
    from sklearn.model_selection import train_test_split

    data_train, data_test = train_test_split(data, train_size=0.8, random_state=42)

    score = 0
    len_data = len(data_train)

    for node in graph.nodes:
        node_name = node.content["name"]
        node_type = node.content["type"]

        if node_name not in data_train.columns:
            continue

        data_of_node_train = data_train[node_name]
        data_of_node_test = data_test[node_name]

        # Different handling based on node type and whether it has parents
        if not node.parents:
            if node_type == "cont":
                # For continuous nodes with no parents, use Gaussian distribution
                mu, sigma = data_of_node_train.mean(), data_of_node_train.std()
                score += norm.logpdf(data_of_node_test, loc=mu, scale=sigma).sum()
            else:
                # For discrete nodes with no parents, use frequency-based probability
                count = data_of_node_train.value_counts()
                frequency = np.log(count / len_data)
                score += data_of_node_test.map(frequency).fillna(np.log(1e-7)).sum()
        else:
            # For nodes with parents, use the associated model
            parent_model = node.content.get("parent_model")
            if not parent_model:
                continue

            # Get parent node names
            parent_names = [parent.name for parent in node.parents]

            # Extract features and target
            features_train = data_train[parent_names].to_numpy()
            target_train = data_of_node_train.to_numpy()

            if len(set(target_train)) <= 1:
                continue  # Skip if target has only one value

            try:
                # Fit the model
                fitted_model = parent_model.fit(features_train, target_train)

                features_test = data_test[parent_names].to_numpy()
                target_test = data_of_node_test.to_numpy()

                if node_type == "cont":
                    # For continuous target, calculate RMSE and log-likelihood
                    predictions = fitted_model.predict(features_test)
                    rmse = (
                        root_mean_squared_error(target_test, predictions, squared=False)
                        + 1e-7
                    )
                    score += norm.logpdf(target_test, loc=predictions, scale=rmse).sum()
                else:
                    # For discrete target, calculate log-probability
                    predict_proba = fitted_model.predict_proba(features_test)
                    unique_values = data_train[node_name].unique()
                    value_to_index = {
                        val: idx for idx, val in enumerate(sorted(unique_values))
                    }

                    indices = [value_to_index.get(val, 0) for val in target_test]
                    probas = np.array(
                        [predict_proba[i, idx] for i, idx in enumerate(indices)]
                    )
                    probas = np.maximum(probas, 1e-7)  # Avoid log(0)
                    score += np.log(probas).sum()
            except Exception as e:
                print(f"Error fitting model for node {node_name}: {e}")
                # Penalize errors
                score -= 1000

    return -score  # Negative for minimization
