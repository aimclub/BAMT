from abc import ABC

from bamt.core.nodes.node import Node


class RootNode(Node, ABC):
    """Abstract Class based on Node Abstract class for root nodes of the
    Bayesian network. Root nodes are represented by
    Distributions."""

    def __init__(self):
        super().__init__()
