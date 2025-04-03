from enum import Enum


class RawNodeType(Enum):
    cont = "cont"
    disc = "disc"
    disc_num = "disc_num"


class NodeType(Enum):
    abstract = "BaseNode"
    conditional_continuous = "ConditionalContinuousNode"
    conditional_discrete = "ConditionalDiscreteNode"

    root_continuous = "GaussianNode"
    root_discrete = "DiscreteNode"


class NodeSign(Enum):
    pos = 1
    neg = -1


continuous_nodes = [
    NodeType.conditional_continuous,
    NodeType.root_continuous
]
discrete_nodes = [
    NodeType.conditional_discrete,
    NodeType.root_discrete,
]
