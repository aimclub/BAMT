import sys
import warnings
from typing import Union

import numpy as np
import pandas as pd

from bamt.mi_entropy_gauss import entropy_all as entropy
from bamt.mi_entropy_gauss import mi_gauss as mutual_information
from bamt.preprocess.numpy_pandas import get_type_numpy

from .score_function import ScoreFunction


class MutualInformationScore(ScoreFunction):
    """
    Mutual Information-based scoring function.

    Uses mutual information and entropy to score network structures.
    Works with continuous, discrete, and mixed data types.

    The score decomposes as:
        Score = N * (MI(X, Parents(X)) - H(X))

    where N is the number of samples, MI is mutual information,
    and H is entropy.
    """

    def __init__(self, score_type="LL"):
        """
        Initialize the MutualInformationScore.

        Args:
            score_type: Type of score - "LL" (log-likelihood), "BIC", or "AIC"
        """
        super().__init__()
        self.score_type = score_type.upper()

    def estimate(
        self, data: Union[pd.DataFrame, np.ndarray], node: str = None, parents: list = None
    ) -> float:
        """
        Estimate the mutual information-based score.

        Args:
            data: The dataset containing [node, parent1, parent2, ...]
            node: The target node (optional, defaults to first column)
            parents: List of parent nodes (optional, inferred from data)

        Returns:
            float: The score value
        """
        NROW = data.shape[0] if isinstance(data, (pd.DataFrame, np.ndarray)) else len(data)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Handle single node (no parents)
            if isinstance(data, pd.Series):
                return 0.0

            # Handle DataFrame with single column
            if isinstance(data, pd.DataFrame) and len(data.columns) == 1:
                return 0.0

            # Handle numpy array with single column
            if isinstance(data, np.ndarray) and (data.ndim == 1 or data.shape[1] == 1):
                return 0.0

            # Calculate log-likelihood score
            if isinstance(data, pd.DataFrame):
                mi_score = mutual_information(data, method=self.score_type)
                ent_score = entropy(data.iloc[:, 0], method=self.score_type)
            else:
                mi_score = mutual_information(data, method=self.score_type)
                ent_score = entropy(data[:, 0], method=self.score_type)

            log_lik = NROW * (mi_score - ent_score)

            # Apply penalty based on score type
            if self.score_type == "BIC":
                penalty = 0.5 * self._num_params(data) * np.log(NROW)
                return log_lik - penalty
            elif self.score_type == "AIC":
                penalty = self._num_params(data)
                return log_lik - penalty
            else:  # LL
                return log_lik

    def _num_params(self, data):
        """
        Calculate the number of parameters for the penalty term.

        Args:
            data: The dataset

        Returns:
            int: Number of parameters
        """
        # Convert pandas DataFrame to numpy array
        if isinstance(data, pd.DataFrame):
            data = data.values
        # Convert pandas Series to numpy array
        if isinstance(data, pd.Series):
            data = np.array(data)

        # Calculate number of parameters for numpy array
        if isinstance(data, np.ndarray):
            node_type = get_type_numpy(data)
            columns_for_continuous = [
                param for param, node in node_type.items() if node == "cont"
            ]
            columns_for_discrete = [
                param for param, node in node_type.items() if node == "disc"
            ]

            prod = 1
            for var in columns_for_discrete:
                unique_vals = (
                    len(np.unique(data[:, var]))
                    if data.ndim != 1
                    else len(np.unique(data))
                )
                prod *= unique_vals

            if columns_for_continuous:
                prod *= len(columns_for_continuous)

            # Handle overflow error
            try:
                return prod
            except OverflowError:
                return sys.float_info.max

        return 0
