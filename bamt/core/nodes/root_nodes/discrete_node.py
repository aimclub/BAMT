from typing import Optional

from bamt.core.node_models import EmpiricalDistribution
from .root_node import RootNode


class DiscreteNode(RootNode):
    def __init__(self, name: Optional[str] = None, distribution: Optional[EmpiricalDistribution] = None):
        """
        Initialize the DiscreteNode with an optional name and EmpiricalDistribution.

        Args:
            name (Optional[str]): The name of the node.
            distribution (Optional[EmpiricalDistribution]): An EmpiricalDistribution object.
        """
        super().__init__(name=name)
        self._distribution = (
            distribution if distribution is not None else EmpiricalDistribution()
        )

    def __str__(self) -> str:
        """
        Return the string representation of the Discrete node.

        Returns:
            str: The string representation of the node.
        """
        return "Discrete Node with " + str(self._distribution)
