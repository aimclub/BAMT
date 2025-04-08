from typing import Callable, List
import networkx as nx

from bamt.builders.evo_builders.deap_graph import Graph, Node


def has_no_self_cycle(graph: Graph) -> bool:
    """Verify that the graph has no self-cycles."""
    for node in graph.nodes:
        if node in node.children:
            return False
    return True


def has_no_cycle(graph: Graph) -> bool:
    """Verify that the graph has no cycles."""
    G = graph.to_networkx()
    return nx.is_directed_acyclic_graph(G)


def node_type_constraint(graph: Graph, allowed_types: List[str]) -> bool:
    """Verify that all nodes in the graph have an allowed type."""
    for node in graph.nodes:
        if "type" in node.content and node.content["type"] not in allowed_types:
            return False
    return True


def max_parents_constraint(graph: Graph, max_parents: int) -> bool:
    """Verify that no node has more parents than allowed."""
    for node in graph.nodes:
        if len(node.parents) > max_parents:
            return False
    return True


def has_no_duplicates(graph: Graph) -> bool:
    """Verify that there are no duplicate node names."""
    node_names = [node.name for node in graph.nodes]
    return len(node_names) == len(set(node_names))
