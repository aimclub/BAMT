from typing import Optional

from .root_node import RootNode
from bamt.core.node_models import EmpiricalDistribution


class DiscreteNode(RootNode):
    def __init__(self, distribution: Optional[EmpiricalDistribution] = None):
        """
        Initialize the DisscreteNode with an optional EmpiricalDistribution.

        Args:
            distribution (Optional[ContinuousDistribution]): A ContinuousDistribution object.
        """
        super().__init__()
        self._distribution = (
            distribution if distribution is not None else EmpiricalDistribution
        )

    def __str__(self) -> str:
        """
        Return the string representation of the Discrete node.

        Returns:
            str: The string representation of the node.
        """
        return "Discrete Node with " + str(self._distribution)
