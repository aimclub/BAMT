from typing import Callable, Sequence, Tuple

import networkx as nx
import numpy as np

from golem.metrics import mmd
from golem.metrics.mmd import compute_mmd


def compute_all_stats(graph_prediction: Sequence[nx.Graph],
                      graph_target: Sequence[nx.Graph]) -> Tuple[float, float]:
    mmd_degree = degree_stats(graph_prediction, graph_target)
    mmd_clustering = clustering_stats(graph_prediction, graph_target)

    return mmd_degree, mmd_clustering


def degree_stats(graph_prediction: Sequence[nx.Graph],
                 graph_target: Sequence[nx.Graph]) -> float:
    return mmd_stats(nx.degree_histogram, graph_prediction, graph_target, normalize=True)


def clustering_stats(graph_prediction: Sequence[nx.Graph],
                     graph_target: Sequence[nx.Graph]) -> float:
    bins = 100
    return mmd_stats(clustering_stats_graph, graph_prediction, graph_target,
                     sigma=0.1, distance_scaling=bins, normalize=False)


def mmd_stats(stat_function: Callable[[nx.Graph], np.ndarray],
              graph_prediction: Sequence[nx.Graph],
              graph_target: Sequence[nx.Graph],
              **kwargs):
    sample_predict = list(map(stat_function, graph_prediction))
    sample_target = list(map(stat_function, graph_target))
    return compute_mmd(sample_target, sample_predict, **kwargs)


def mmd_stats_impl(stat_function: Callable[[nx.Graph], np.ndarray],
                   graph_prediction: Sequence[nx.Graph],
                   graph_target: Sequence[nx.Graph],
                   kernel: Callable = mmd.gaussian_emd,
                   sigma: float = 1.0,
                   distance_scaling: float = 1.0,
                   normalize: bool = False) -> float:

    sample_predict = list(map(stat_function, graph_prediction))
    sample_target = list(map(stat_function, graph_target))

    mmd_dist = compute_mmd(sample_target, sample_predict,
                           normalize=normalize, kernel=kernel,
                           sigma=sigma, distance_scaling=distance_scaling)
    return mmd_dist


def clustering_stats_graph(graph: nx.Graph, bins: int = 100) -> np.ndarray:
    clustering_coeffs = list(nx.clustering(graph).values())
    hist, _ = np.histogram(clustering_coeffs, bins=bins, range=(0.0, 1.0), density=True)
    return hist
