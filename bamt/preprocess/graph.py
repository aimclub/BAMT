import os,sys,inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0,parentdir)
from copy import copy
import math
from typing import List
import numpy as np
import pandas as pd


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
    all_nodes = []
    for e in edges:
        all_nodes.extend(e)
    return list(set(all_nodes))

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
    parents_dict = {var: [] for var in nodes}
    for e in edges:
        parents_dict[e[1]].append(e[0])
    return parents_dict