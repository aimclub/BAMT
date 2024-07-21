from abc import ABC

import numpy as np

from bamt.core.node_models.distribution import Distribution
from bamt.core.nodes.node import Node


class RootNode(Node, ABC):
    """Abstract Class based on Node Abstract class for root nodes of the
    Bayesian network. Root nodes are represented by
    Distributions."""

    def __init__(self, name):
        super().__init__(name)
        self._distribution = None
        self._children = []

    def __repr__(self):
        pass

    def get_children(self) -> list:
        """
        Get the children of this node.

        Returns:
            List[ContinuousNode]: A list of child nodes.
        """
        return self._children

    def get_parents(self) -> str:
        """
        Get the parents of this node. Since this is a root node, it cannot have parents.

        Returns:
            str: A message indicating that this node is a root node and cannot have parents.
        """
        return "This is a root node, thus it cannot have parents."

    def add_child(self, child) -> None:
        """
        Add a child to this node.

        Args:
            child: The child node to add.
        """
        if child not in self._children:
            self._children.append(child)

    def add_parent(self, parent) -> str:
        """
        Attempt to add a parent to this node. Since this is a root node, it cannot have parents.

        Args:
            parent: The parent node to add.

        Returns:
            str: A message indicating that this node is a root node and cannot have parents.
        """
        raise Exception("A parent cannot be added to a root node")

    def fit(self, X: np.ndarray) -> None:
        """
        Fit the distribution to the data.

        Args:
            X (np.ndarray): The data to fit.
        """
        self._distribution.fit(X)

    def sample(self, num_samples: int) -> np.ndarray:
        """
        Generate samples from the fitted distribution.

        Args:
            num_samples (int): The number of samples to generate.

        Returns:
            np.ndarray: The generated samples.
        """
        return self._distribution.sample(num_samples)

    @property
    def distribution(self) -> Distribution:
        """
        Get the continuous distribution of this node.

        Returns:
            Distribution: The desired distribution.
        """
        return self._distribution
