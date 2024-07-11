from typing import Sequence

import networkx as nx
import numpy as np

from golem.metrics.graph_features import degree_stats
from libs.netcomp import _eigs, normalized_laplacian_eig
from libs.netcomp import laplacian_matrix


def nxgraph_stats(graph: nx.Graph):
    degrees = nx.degree_histogram(graph)
    degrees_norm = np.divide(degrees, np.linalg.norm(degrees)).round(2)
    stats = dict(
        num_nodes=graph.number_of_nodes(),
        num_edges=graph.number_of_edges(),
        avg_clustering=np.round(nx.average_clustering(graph), 3),
        degrees_hist=degrees,
        degrees_hist_norm=degrees_norm,
    )
    return stats


def degree_distance_kernel(target_graph: nx.DiGraph, graph: nx.DiGraph) -> float:
    return degree_stats([graph], [target_graph])


def degree_distance(target_graph: nx.DiGraph,
                    graph: nx.DiGraph,
                    normalized: bool = False) -> float:
    """This is a heuristic metric for graphs where central
    nodes are more important than peripheral ones. The "heavier"
    the nodes (i.e. the higher their degree), the more significant
    the difference in number of such nodes between two graphs."""
    # Compute histogram of node degrees
    degrees_t = np.array(nx.degree_histogram(target_graph), dtype=float)
    degrees_g = np.array(nx.degree_histogram(graph), dtype=float)
    return degree_dist_weighted_compute(degrees_t, degrees_g, normalized)


def degree_dist_weighted_compute(degrees_t: Sequence[float],
                                 degrees_g: Sequence[float],
                                 normalized: bool = False) -> float:
    degrees_t = np.asarray(degrees_t)
    degrees_g = np.asarray(degrees_g)

    # Extend arrays to the same length with zeros
    common_len = max(len(degrees_t), len(degrees_g))
    degrees_t.resize(common_len, refcheck=False)
    degrees_g.resize(common_len, refcheck=False)

    # Compute weights as normalized degrees
    weights = np.arange(1, common_len + 1).astype(float)
    weights /= np.sum(weights)

    # Normalize
    if normalized:
        degrees_t /= np.sum(degrees_t)
        degrees_g /= np.sum(degrees_g)

    # Compute distance between node degrees weighted by degree
    dist = np.linalg.norm(weights * (degrees_t - degrees_g))
    return dist


def size_diff(target_graph: nx.DiGraph, graph: nx.DiGraph) -> float:
    nodes_diff = abs(target_graph.number_of_nodes() - graph.number_of_nodes())
    edges_diff = abs(target_graph.number_of_edges() - graph.number_of_edges())
    return nodes_diff + np.sqrt(edges_diff)


def spectral_dist(target_graph: nx.DiGraph, graph: nx.DiGraph,
                  k: int = 20, kind: str = 'laplacian',
                  size_diff_penalty: float = 0.2,
                  match_size: bool = False,
                  ) -> float:
    target_adj = nx.adjacency_matrix(target_graph)
    adj = nx.adjacency_matrix(graph)

    # compute spectral distance
    value = lambda_dist(target_adj, adj, kind=kind, k=k, match_size=match_size)

    if size_diff_penalty > 1e-5:
        value += size_diff_penalty * size_diff(target_graph, graph)
    return value


def spectral_dists_all(target_graph: nx.DiGraph, graph: nx.DiGraph,
                       k: int = 20, match_size: bool = True) -> dict:
    target_adj = nx.adjacency_matrix(target_graph)
    adj = nx.adjacency_matrix(graph)

    print(f'computing metrics for {k} spectral values between {target_adj.shape} & {adj.shape}')

    vals = {}
    for kind in ('adjacency', 'laplacian_norm', 'laplacian'):
        value = lambda_dist(target_adj, adj, kind=kind, k=k, match_size=match_size)
        vals[kind] = np.round(value, 3)
    vals['nodes_diff'] = size_diff(target_graph, graph)
    return vals


def min_max(a, b):
    return (a, b) if a <= b else (b, a)


def lambda_dist(A1, A2, k=None, p=2, kind='laplacian', match_size=True):
    """The lambda distance between graphs, which is defined as

        d(G1,G2) = norm(L_1 - L_2)

    where L_1 is a vector of the top k eigenvalues of the appropriate matrix
    associated with G1, and L2 is defined similarly.

    Parameters
    ----------
    A1, A2 : NumPy matrices
        Adjacency matrices of graphs to be compared

    k : Integer
        The number of eigenvalues to be compared

    p : non-zero Float
        The p-norm is used to compare the resulting vector of eigenvalues.

    kind : String , in {'laplacian','laplacian_norm','adjacency'}
        The matrix for which eigenvalues will be calculated.

    Returns
    -------
    dist : float
        The distance between the two graphs

    Notes
    -----
    The norm can be any p-norm; by default we use p=2. If p<0 is used, the
    result is not a mathematical norm, but may still be interesting and/or
    useful.

    If k is provided, then we use the k SMALLEST eigenvalues for the Laplacian
    distances, and we use the k LARGEST eigenvalues for the adjacency
    distance. This is because the corresponding order flips, as L = D-A.

    References
    ----------

    See Also
    --------
    netcomp.linalg._eigs
    normalized_laplacian_eigs

    """
    # check sizes & determine number of eigenvalues (k)
    nmin, nmax = min_max(A1.shape[0], A2.shape[0])
    if match_size:
        shape = (nmax, nmax)
        A1.resize(shape)
        A2.resize(shape)
    else:
        k = min(k, nmin)

    if kind == 'laplacian':
        # form matrices
        L1, L2 = [laplacian_matrix(A) for A in [A1, A2]]
        # get eigenvalues, ignore eigenvectors
        evals1, evals2 = [_eigs(L)[0] for L in [L1, L2]]
    elif kind == 'laplacian_norm':
        # use our function to graph evals of normalized laplacian
        evals1, evals2 = [normalized_laplacian_eig(A)[0] for A in [A1, A2]]
    elif kind == 'adjacency':
        evals1, evals2 = [_eigs(A)[0] for A in [A1, A2]]
        # reverse, so that we are sorted from large to small, since we care
        # about the k LARGEST eigenvalues for the adjacency distance
        evals1, evals2 = [evals[::-1] for evals in [evals1, evals2]]
    else:
        raise AttributeError(f"Invalid type {kind}, choose from 'laplacian', "
                             f"'laplacian_norm', and 'adjacency'.")
    dist = np.linalg.norm(evals1[:k] - evals2[:k], ord=p)
    return dist
