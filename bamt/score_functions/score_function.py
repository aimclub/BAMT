from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd


class ScoreFunction(ABC):
    """
    Abstract base class for scoring functions used in structure learning.

    Scoring functions evaluate the quality of a Bayesian Network structure
    given data. They are used by structure learning algorithms to guide
    the search for optimal network structures.
    """

    def __init__(self):
        pass

    @abstractmethod
    def estimate(
        self, data: Union[pd.DataFrame, np.ndarray], node: str = None, parents: list = None
    ) -> float:
        """
        Estimate the score for a given node-parent configuration.

        Args:
            data: The dataset (DataFrame or numpy array)
            node: The target node name (if DataFrame) or index (if array)
            parents: List of parent node names (if DataFrame) or indices (if array)

        Returns:
            float: The score value (higher is better for most metrics)
        """
        pass

    def score_structure(self, data: Union[pd.DataFrame, np.ndarray], edges: list) -> float:
        """
        Score an entire network structure.

        Args:
            data: The dataset
            edges: List of (parent, child) tuples

        Returns:
            float: Total network score
        """
        from bamt.preprocess.graph import edges_to_dict

        parents_dict = edges_to_dict(edges)
        nodes_with_edges = parents_dict.keys()

        # Score nodes with parents
        scores = []
        for child in nodes_with_edges:
            child_parents = parents_dict[child]
            if isinstance(data, pd.DataFrame):
                score_data = data[[child] + child_parents].copy()
            else:
                # For numpy arrays, assume indices
                indices = [child] + child_parents
                score_data = data[:, indices]
            scores.append(self.estimate(score_data, node=child, parents=child_parents))

        # Score nodes without parents
        if isinstance(data, pd.DataFrame):
            orphan_nodes = set(data.columns).difference(set(nodes_with_edges))
            for node in orphan_nodes:
                score_data = data[[node]].copy()
                scores.append(self.estimate(score_data, node=node, parents=[]))

        return sum(scores)
