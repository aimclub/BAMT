from enum import Enum


class RawNodeType(Enum):
    cont = "cont"
    disc = "disc"
    disc_num = "disc_num"


class NodeType(Enum):
    abstract = "BaseNode"
    conditional_gaussian = "ConditionalGaussianNode"
    conditional_mixture_gaussian = "ConditionalMixtureGaussianNode"
    conditional_logit = "ConditionalLogitNode"
    logit = "LogitNode"

    mixture_gaussian = "MixtureGaussianNode"

    gaussian = "GaussianNode"
    discrete = "DiscreteNode"


class NetworkType(Enum):
    continuous = "Continuous"
    discrete = "Discrete"
    hybrid = "Hybrid"


class NodeSign(Enum):
    pos = 1
    neg = -1


continuous_nodes = [
    NodeType.gaussian,
    NodeType.conditional_gaussian,
    NodeType.mixture_gaussian,
    NodeType.conditional_mixture_gaussian,
]
discrete_nodes = [
    NodeType.conditional_logit,
    NodeType.logit,
    NodeType.discrete,
]

nodes_with_regressors = [NodeType.gaussian, NodeType.conditional_gaussian]
nodes_with_classifiers = [NodeType.logit, NodeType.conditional_logit]


mixture_nodes = [NodeType.mixture_gaussian, NodeType.conditional_mixture_gaussian]
logit_nodes = [NodeType.conditional_logit, NodeType.logit]

# whether distributions for nodes contain combinations of values (their distribution signature: {"hybcprob: {...}})
nodes_with_combinations = [
    NodeType.conditional_logit,
    NodeType.conditional_gaussian,
    NodeType.conditional_mixture_gaussian,
]
