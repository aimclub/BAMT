from typing import Optional

from bamt.core.node_models import EmpiricalDistribution
from .root_node import RootNode


class DiscreteNode(RootNode):
    def __init__(self, name, distribution: Optional[EmpiricalDistribution] = None):
        """
        Initialize the DisscreteNode with an optional EmpiricalDistribution.

        Args:
            distribution (Optional[ContinuousDistribution]): A ContinuousDistribution object.
        """
        super().__init__(name)
        self._distribution = (
            distribution if distribution is not None else EmpiricalDistribution()
        )

    def __repr__(self) -> str:
        """
        Return the string representation of the Discrete node.

        Returns:
            str: The string representation of the node.
        """
        return f"{self.name}. Discrete Node with " + str(self._distribution)
