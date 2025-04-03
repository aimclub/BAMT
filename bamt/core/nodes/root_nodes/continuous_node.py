from typing import Optional

import numpy as np

from bamt.core.node_models.continuous_distribution import ContinuousDistribution
from bamt.core.nodes.root_nodes.root_node import RootNode


class ContinuousNode(RootNode):
    """
    Class for continuous nodes of the Bayesian network.
    Continuous nodes are represented by `scipy.stats` continuous distributions.
    These distributions are wrapped in the `ContinuousDistribution` class.
    Example Usage:


    >>> data = np.random.normal(0, 1, 1000)
    >>> dist = ContinuousDistribution()
    >>> node = ContinuousNode(distribution=dist)
    >>> node.fit(data)
    >>> print(node)
    >>> samples = node.sample(10)
    >>> print(samples)
    >>> print(node.get_parents())
    """

    def __init__(self, distribution: Optional[ContinuousDistribution] = None):
        """
        Initialize the ContinuousNode with an optional ContinuousDistribution.

        Args:
            distribution (Optional[ContinuousDistribution]): A ContinuousDistribution object.
        """
        super().__init__()
        self._distribution = (
            distribution if distribution is not None else ContinuousDistribution()
        )

    def __str__(self) -> str:
        """
        Return the string representation of the continuous node.

        Returns:
            str: The string representation of the node.
        """
        return "Continuous Node with " + str(self._distribution)
