import itertools
from copy import deepcopy
from typing import Any, List, Tuple

from golem.core.dag.graph_node import descriptive_id_recursive_nodes
from golem.core.dag.graph_utils import distance_to_primary_level


def equivalent_subtree(graph_first: Any, graph_second: Any, with_primary_nodes: bool = False) \
        -> List[Tuple[Any, Any]]:
    """Finds the similar subtrees in two given trees.
    With `with_primary_nodes` primary nodes are considered too.
    Due to a lot of common subgraphs consisted only of single primary nodes, these nodes can be
    not considered with `with_primary_nodes=False`."""

    pairs_list = []
    all_nodes = graph_first.nodes + graph_second.nodes
    all_descriptive_ids = [set(descriptive_id_recursive_nodes(node)) for node in graph_first.nodes] \
                          + [set(descriptive_id_recursive_nodes(node)) for node in graph_second.nodes]
    all_recursive_ids = dict(zip(all_nodes, all_descriptive_ids))
    for node_first in graph_first.nodes:
        for node_second in graph_second.nodes:
            if (node_first, node_second) in pairs_list:
                continue
            equivalent_pairs = structural_equivalent_nodes(node_first=node_first, node_second=node_second,
                                                           recursive_ids=all_recursive_ids)
            pairs_list.extend(equivalent_pairs)

    pairs_list = list(set(pairs_list))
    if with_primary_nodes:
        return pairs_list
    # remove nodes with no children
    result = []
    for pair in pairs_list:
        if len(pair[0].nodes_from) != 0:
            result.append(pair)
    return result


def replace_subtrees(graph_first: Any, graph_second: Any, node_from_first: Any, node_from_second: Any,
                     layer_in_first: int, layer_in_second: int, max_depth: int):
    node_from_graph_first_copy = deepcopy(node_from_first)

    summary_depth = layer_in_first + distance_to_primary_level(node_from_second) + 1
    if summary_depth <= max_depth and summary_depth != 0:
        graph_first.update_subtree(node_from_first, node_from_second)

    summary_depth = layer_in_second + distance_to_primary_level(node_from_first) + 1
    if summary_depth <= max_depth and summary_depth != 0:
        graph_second.update_subtree(node_from_second, node_from_graph_first_copy)


def num_of_parents_in_crossover(num_of_final_inds: int) -> int:
    return num_of_final_inds if not num_of_final_inds % 2 else num_of_final_inds + 1


def filter_duplicates(archive, population) -> List[Any]:
    filtered_archive = []
    for ind in archive.items:
        has_duplicate_in_pop = False
        for pop_ind in population:
            if ind.fitness == pop_ind.fitness:
                has_duplicate_in_pop = True
                break
        if not has_duplicate_in_pop:
            filtered_archive.append(ind)
    return filtered_archive


def structural_equivalent_nodes(node_first: Any, node_second: Any, recursive_ids: dict = None) -> List[Tuple[Any, Any]]:
    """ Returns the list of nodes from which subtrees are structurally equivalent.
    :param node_first: node from first graph from which to start the search.
    :param node_second: node from second graph from which to start the search.
    :param recursive_ids: dict with recursive descriptive id of node with nodes as keys.
    Descriptive ids can be obtained with `descriptive_id_recursive_nodes`.
    """

    nodes = []
    is_same_type = type(node_first) == type(node_second)
    # check if both nodes are primary or secondary
    if hasattr(node_first, 'is_primary') and hasattr(node_second, 'is_primary'):
        is_same_graph_node_type = node_first.is_primary == node_second.is_primary
        is_same_type = is_same_type and is_same_graph_node_type

    for node1_child, node2_child in itertools.product(node_first.nodes_from, node_second.nodes_from):
        nodes_set = structural_equivalent_nodes(node_first=node1_child, node_second=node2_child,
                                                recursive_ids=recursive_ids)
        nodes.extend(nodes_set)
    if is_same_type and len(node_first.nodes_from) == len(node_second.nodes_from) \
            and are_subtrees_the_same(match_set=nodes,
                                      node_first=node_first, node_second=node_second,
                                      recursive_ids=recursive_ids):
        nodes.append((node_first, node_second))
    return nodes


def are_subtrees_the_same(match_set: List[Tuple[Any, Any]],
                          node_first: Any, node_second: Any, recursive_ids: dict = None) -> bool:
    """ Returns `True` if subtrees of specified root nodes are the same, otherwise returns `False`.
    :param match_set: pairs of nodes to checks subtrees from.
    :param node_first: first node from which to compare subtree
    :param node_second: second node from which to compare subtree
    :param recursive_ids: dict with recursive descriptive id of node with nodes as keys.
    Descriptive ids can be obtained with `descriptive_id_recursive_nodes`."""
    matched = []
    if not recursive_ids:
        first_recursive_id = set(descriptive_id_recursive_nodes(node_first))
        second_recursive_id = set(descriptive_id_recursive_nodes(node_second))
    else:
        first_recursive_id = recursive_ids[node_first]
        second_recursive_id = recursive_ids[node_second]

    # 1. Number of exact children must be the same
    # 2. All children from one node must have a match from other node children
    # 3. Protection from cycles when lengths of descriptive ids are the same due to cycles
    if len(node_first.nodes_from) != len(node_second.nodes_from) or \
            len(match_set) == 0 and len(node_first.nodes_from) != 0 or \
            len(first_recursive_id) != len(second_recursive_id):
        return False

    for node, node2 in itertools.product(node_first.nodes_from, node_second.nodes_from):
        if (node, node2) or (node2, node) in match_set:
            matched.append((node, node2))
    if len(matched) >= len(node_first.nodes_from):
        return True
    return False
