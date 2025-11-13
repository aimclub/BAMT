"""
Akaike Information Criterion (AIC) score function.

AIC is a model selection criterion that balances model fit with
complexity. It penalizes complexity less than BIC.
"""

from typing import Union

import numpy as np
import pandas as pd

from .mutual_information_score import MutualInformationScore


class AICScore(MutualInformationScore):
    """
    Akaike Information Criterion (AIC) score.

    AIC = log-likelihood - k

    where:
    - log-likelihood: log likelihood of the data given the model
    - k: number of parameters

    Higher AIC values indicate better models.
    """

    def __init__(self):
        """Initialize AIC score function."""
        super().__init__(score_type="AIC")

    def estimate(
        self, data: Union[pd.DataFrame, np.ndarray], node: str = None, parents: list = None
    ) -> float:
        """
        Estimate the AIC score.

        Args:
            data: The dataset containing [node, parent1, parent2, ...]
            node: The target node (optional, defaults to first column)
            parents: List of parent nodes (optional, inferred from data)

        Returns:
            float: The AIC score (higher is better)
        """
        return super().estimate(data, node, parents)

    def __str__(self):
        return "AICScore"
