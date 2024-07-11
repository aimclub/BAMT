import random
from abc import abstractmethod, ABC
from random import choice
from typing import Optional, Iterable, List

from golem.core.optimisers.graph import OptNode


class OptNodeFactory(ABC):
    @abstractmethod
    def exchange_node(self,
                      node: OptNode) -> Optional[OptNode]:
        """
        Returns new node based on a current node using information about node and advisor.

        :param node: current node that must be changed.
        """
        pass

    @abstractmethod
    def get_parent_node(self, node: OptNode, **kwargs) -> Optional[OptNode]:
        """
        Returns new parent node for the current node
        based on the content of the current node and using advisor.

        :param node: current node for which a parent node is generated
        """
        pass

    @abstractmethod
    def get_node(self, **kwargs) -> Optional[OptNode]:
        """
        Returns new node based on the requirements for a node.
        """
        pass

    @abstractmethod
    def get_all_available_operations(self) -> List[str]:
        """
        Returns all available models and data operations.
        """
        pass


class DefaultOptNodeFactory(OptNodeFactory):
    """Default node factory that either randomly selects
    one node from the provided lists of available nodes
    or returns a node with a random numeric node name
    in the range from 0 to `num_node_types`."""

    def __init__(self,
                 available_node_types: Optional[Iterable[str]] = None,
                 num_node_types: Optional[int] = None):
        self.available_nodes = tuple(available_node_types) if available_node_types else None
        self._num_node_types = num_node_types or 1000

    def get_all_available_operations(self) -> Optional[List[str]]:
        """
        Returns all available models and data operations.
        """
        return self.available_nodes

    def exchange_node(self, node: OptNode) -> OptNode:
        return self.get_node()

    def get_parent_node(self, node: OptNode, **kwargs) -> OptNode:
        return self.get_node(**kwargs)

    def get_node(self, **kwargs) -> OptNode:
        chosen_node_type = choice(self.available_nodes) \
            if self.available_nodes \
            else random.randint(0, self._num_node_types)
        return OptNode(content={'name': chosen_node_type})
