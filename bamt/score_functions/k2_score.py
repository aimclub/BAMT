from typing import Union

import numpy as np
import pandas as pd
from pgmpy.estimators import K2Score as PgmpyK2Score

from .score_function import ScoreFunction


class K2Score(ScoreFunction):
    """
    K2 scoring function for discrete Bayesian Networks.

    The K2 score is a Bayesian scoring metric that evaluates the posterior
    probability of a network structure given data. It's particularly suited
    for discrete data and assumes a uniform prior over structures.

    Note: K2 score only works with discrete data. For continuous or mixed data,
    use MutualInformationScore or other appropriate metrics.
    """

    def __init__(self):
        super().__init__()
        self._pgmpy_scorer = None

    def estimate(
        self, data: Union[pd.DataFrame, np.ndarray], node: str = None, parents: list = None
    ) -> float:
        """
        Estimate the K2 score for a node given its parents.

        Args:
            data: The dataset (must be discrete)
            node: The target node
            parents: List of parent nodes

        Returns:
            float: The K2 score (higher is better)
        """
        # Convert to DataFrame if numpy array
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                data = pd.DataFrame(data, columns=["var"])
                node = "var"
                parents = []
            else:
                col_names = [f"var_{i}" for i in range(data.shape[1])]
                data = pd.DataFrame(data, columns=col_names)
                if node is None:
                    node = col_names[0]
                    parents = col_names[1:] if len(col_names) > 1 else []

        # Initialize pgmpy scorer if not done
        if self._pgmpy_scorer is None:
            self._pgmpy_scorer = PgmpyK2Score(data)

        # Calculate local score for this node
        if parents is None:
            parents = []

        return self._pgmpy_scorer.local_score(node, parents)
