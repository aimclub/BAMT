import numpy as np

from .distribution import Distribution


class EmpiricalDistribution(Distribution):
    def __init__(self):
        pass

    def fit(self, X: np.ndarray) -> None:
        pass

    def sample(self, num_samples: int) -> np.ndarray:
        pass

    def __str__(self):
        return "Empirical Distribution"
