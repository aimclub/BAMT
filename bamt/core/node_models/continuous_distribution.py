import numpy as np

from .distribution import Distribution


class ContinuousDistribution(Distribution):
    def __init__(self, distribution_model=None, **parameters):
        self._distribution = distribution_model
        self._parameters = parameters

    def fit(self, X: np.ndarray) -> None:
        if self._distribution is None:
            # TODO: implement an algorithm that finds a distribution and fits it with chosen parameters
            pass
        else:
            pass

    def sample(self, num_samples: int) -> np.ndarray:
        pass

    def __str__(self):
        return str(self._distribution.name) + " continuous distribution"
