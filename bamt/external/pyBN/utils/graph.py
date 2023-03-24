"""
Collection of Graph Algorithms.

Any networkx dependencies should exist here only, but
there are currently a few exceptions which will
eventually be corrected.

"""

import networkx as nx


def would_cause_cycle(e, u, v, reverse=False):
    """
    Test if adding the edge u -> v to the BayesNet
    object would create a DIRECTED (i.e. illegal) cycle.
    """
    G = nx.DiGraph(e)
    if reverse:
        G.remove_edge(v, u)
    G.add_edge(u, v)
    try:
        nx.find_cycle(G, source=u)
        return True
    except BaseException:
        return False


def topsort(edge_dict, root=None):
    """
    List of nodes in topological sort order from edge dict
    where key = rv and value = list of rv's children
    """
    queue = []
    if root is not None:
        queue = [root]
    else:
        for rv in edge_dict.keys():
            prior = True
            for p in edge_dict.keys():
                if rv in edge_dict[p]:
                    prior = False
            if prior:
                queue.append(rv)

    visited = []
    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.append(vertex)
            for nbr in edge_dict[vertex]:
                queue.append(nbr)
            # queue.extend(edge_dict[vertex]) # add all vertex's children
    return visited
