from copy import copy
from typing import List
from collections import defaultdict


def nodes_from_edges(edges: list):
    """
    Retrieves all nodes from the list of edges.
            Arguments
    ----------
    *edges* : list

    Returns
    -------
    *nodes* : list

    Effects
    -------
    None
    """
    return set().union(edges)


def edges_to_dict(edges: list):
    """
    Transfers the list of edges to the dictionary of parents.
            Arguments
    ----------
    *edges* : list

    Returns
    -------
    *parents_dict* : dict

    Effects
    -------
    None
    """
    nodes = nodes_from_edges(edges)
    parents_dict = defaultdict(list, {node: [] for node in nodes})
    parents_dict.update(
        {child: parents_dict[child] + [parent] for parent, child in edges}
    )
    return parents_dict
