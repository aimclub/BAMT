import numpy as np
from .root_node import RootNode
from bamt.core.node_models import ContinuousDistribution


class ContinuousNode(RootNode):
    """Class for continuous nodes of the Bayesian network.
    Continuous nodes are represented by `scipy.stats` continuous distributions.
    These distributions are wrapped in the `ContinuousDistribution` class.
    """

    def __init__(self, distribution: ContinuousDistribution = None):
        super().__init__()
        self._distribution = distribution

    def __str__(self):
        return "Continuous Node with " + str(self._distribution)

    def fit(self, X):
        self._distribution.fit(X)

    def sample(self, num_samples: int) -> np.ndarray:
        return self._distribution.sample(num_samples)

    @property
    def distribution(self):
        return self._distribution
