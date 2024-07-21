from abc import ABC

from bamt.core.nodes.node import Node


class ChildNode(Node, ABC):
    def __init__(self, name):
        super().__init__(name)
