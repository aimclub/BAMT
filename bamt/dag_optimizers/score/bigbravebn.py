"""
BigBraveBN optimizer for Bayesian Network structure learning.

This module implements the BigBrave algorithm which restricts the search space
by identifying likely edges using a proximity-based BRAVE metric, then applies
structure learning within this restricted space.
"""

import math
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import OrdinalEncoder

from bamt.score_functions import ScoreFunction

from .score_dag_optimizer import ScoreDAGOptimizer


class BigBraveBN(ScoreDAGOptimizer):
    """
    BigBrave structure optimizer.

    BigBrave restricts the search space by computing a BRAVE matrix that identifies
    likely edges based on proximity metrics (mutual information or Pearson correlation).
    It then returns a whitelist of possible edges that can be used with other optimizers
    like Hill Climbing.

    The algorithm:
    1. Computes proximity matrix (MI or Pearson)
    2. Identifies N nearest neighbors for each variable
    3. Computes BRAVE coefficients
    4. Filters edges above threshold

    Attributes:
        score_function: The scoring function to use (if combining with structure learning)
        n_nearest: Number of nearest neighbors to consider (default: 5)
        threshold: Threshold multiplier for selecting edges (default: 0.3)
        proximity_metric: Metric for proximity ("MI" or "pearson")
    """

    def __init__(
        self,
        score_function: Optional[ScoreFunction] = None,
        n_nearest: int = 5,
        threshold: float = 0.3,
        proximity_metric: str = "MI",
    ):
        """
        Initialize BigBrave optimizer.

        Args:
            score_function: Optional scoring function for structure learning
            n_nearest: Number of nearest neighbors (default: 5)
            threshold: Threshold multiplier (default: 0.3)
            proximity_metric: "MI" (mutual information) or "pearson"
        """
        super().__init__()
        self.score_function = score_function
        self.n_nearest = n_nearest
        self.threshold = threshold
        self.proximity_metric = proximity_metric
        self.possible_edges = []

    def optimize(
        self,
        data: pd.DataFrame,
        node_names: Optional[List[str]] = None,
        **kwargs
    ) -> nx.DiGraph:
        """
        Find possible edges using BigBrave algorithm.

        This returns a graph with all possible edges identified by BigBrave.
        To get a DAG structure, use the possible_edges as a whitelist with
        another optimizer like HillClimbing.

        Args:
            data: The dataset to analyze
            node_names: Names of nodes (defaults to column names)
            **kwargs: Additional parameters

        Returns:
            nx.DiGraph: Graph with possible edges (not necessarily a valid DAG)
        """
        if node_names is None:
            node_names = list(data.columns)

        # Compute possible edges using BRAVE algorithm
        self.possible_edges = self.set_possible_edges_by_brave(
            df=data,
            n_nearest=kwargs.get("n_nearest", self.n_nearest),
            threshold=kwargs.get("threshold", self.threshold),
            proximity_metric=kwargs.get("proximity_metric", self.proximity_metric),
        )

        # Create graph with possible edges
        graph = nx.DiGraph()
        graph.add_nodes_from(node_names)

        # Add edges (converting from column names to actual names if needed)
        for u, v in self.possible_edges:
            if u in node_names and v in node_names:
                graph.add_edge(u, v)

        return graph

    def set_possible_edges_by_brave(
        self,
        df: pd.DataFrame,
        n_nearest: int = 5,
        threshold: float = 0.3,
        proximity_metric: str = "MI",
    ) -> List[Tuple[str, str]]:
        """
        Returns list of possible edges for structure learning.

        Args:
            df: Input data
            n_nearest: Number of nearest neighbors to consider
            threshold: Threshold for selecting edges
            proximity_metric: Metric used to calculate proximity ("MI" or "pearson")

        Returns:
            List of edge tuples (source, target)
        """
        df_copy = df.copy(deep=True)
        proximity_matrix = self._get_proximity_matrix(df_copy, proximity_metric)
        brave_matrix = self._get_brave_matrix(df_copy.columns, proximity_matrix, n_nearest)

        threshold_value = brave_matrix.max(numeric_only=True).max() * threshold
        filtered_brave_matrix = brave_matrix[brave_matrix > threshold_value].stack()
        self.possible_edges = filtered_brave_matrix.index.tolist()
        return self.possible_edges

    @staticmethod
    def _get_n_nearest(
        data: pd.DataFrame, columns: list, corr: bool = False, number_close: int = 5
    ) -> list:
        """Returns N nearest neighbors for every column of dataframe."""
        groups = []
        for c in columns:
            close_ind = data[c].sort_values(ascending=not corr).index.tolist()
            groups.append(close_ind[: number_close + 1])
        return groups

    @staticmethod
    def _get_proximity_matrix(df: pd.DataFrame, proximity_metric: str) -> pd.DataFrame:
        """Returns matrix of proximity for the dataframe."""
        encoder = OrdinalEncoder()
        df_coded = df.copy()
        columns_to_encode = list(df_coded.select_dtypes(include=["category", "object"]))

        if columns_to_encode:
            df_coded[columns_to_encode] = encoder.fit_transform(df_coded[columns_to_encode])

        if proximity_metric == "MI":
            df_distance = pd.DataFrame(
                np.zeros((len(df.columns), len(df.columns))),
                columns=df.columns,
                index=df.columns,
            )
            for c1 in df.columns:
                for c2 in df.columns:
                    dist = mutual_info_score(df_coded[c1].values, df_coded[c2].values)
                    df_distance.loc[c1, c2] = dist
            return df_distance

        elif proximity_metric == "pearson":
            return df_coded.corr(method="pearson")

        else:
            raise ValueError(f"Unknown proximity_metric: {proximity_metric}. Use 'MI' or 'pearson'")

    def _get_brave_matrix(
        self, df_columns: pd.Index, proximity_matrix: pd.DataFrame, n_nearest: int = 5
    ) -> pd.DataFrame:
        """Returns matrix of Brave coefficients for the DataFrame."""
        brave_matrix = pd.DataFrame(
            np.zeros((len(df_columns), len(df_columns))),
            columns=df_columns,
            index=df_columns,
        )
        groups = self._get_n_nearest(
            proximity_matrix, df_columns.tolist(), corr=True, number_close=n_nearest
        )

        for c1 in df_columns:
            for c2 in df_columns:
                a = b = c = d = 0.0
                if c1 != c2:
                    for g in groups:
                        a += (c1 in g) & (c2 in g)
                        b += (c1 in g) & (c2 not in g)
                        c += (c1 not in g) & (c2 in g)
                        d += (c1 not in g) & (c2 not in g)

                    divisor = (math.sqrt((a + c) * (b + d))) * (
                        math.sqrt((a + b) * (c + d))
                    )
                    br = (a * len(groups) + (a + c) * (a + b)) / (
                        divisor if divisor != 0 else 0.0000000001
                    )
                    brave_matrix.loc[c1, c2] = br

        return brave_matrix
