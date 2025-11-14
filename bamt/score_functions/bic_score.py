"""
Bayesian Information Criterion (BIC) score function.

BIC is a model selection criterion that penalizes model complexity
more heavily than AIC. It's particularly useful for larger datasets.
"""

from typing import Union

import numpy as np
import pandas as pd

from .mutual_information_score import MutualInformationScore


class BICScore(MutualInformationScore):
    """
    Bayesian Information Criterion (BIC) score.

    BIC = log-likelihood - 0.5 * k * log(n)

    where:
    - log-likelihood: log likelihood of the data given the model
    - k: number of parameters
    - n: number of samples

    Higher BIC values indicate better models.
    """

    def __init__(self):
        """Initialize BIC score function."""
        super().__init__(score_type="BIC")

    def estimate(
        self, data: Union[pd.DataFrame, np.ndarray], node: str = None, parents: list = None
    ) -> float:
        """
        Estimate the BIC score.

        Args:
            data: The dataset containing [node, parent1, parent2, ...]
            node: The target node (optional, defaults to first column)
            parents: List of parent nodes (optional, inferred from data)

        Returns:
            float: The BIC score (higher is better)
        """
        return super().estimate(data, node, parents)

    def __str__(self):
        return "BICScore"
