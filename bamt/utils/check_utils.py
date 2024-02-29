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

mixture_nodes = [NodeType.mixture_gaussian, NodeType.conditional_mixture_gaussian]


class Checker:
    def __init__(self):
        self.node_type = NodeType

    def validate_argument(self, arg):
        enumerator = self.node_type.__class__
        if isinstance(arg, str):
            arg = enumerator(arg)
        if arg not in self.node_type.__class__:
            assert TypeError("Wrong type of argument.")
        return True


class RawNodeChecker(Checker):
    def __init__(self, node_type):
        # node_type can be only cont, disc or disc_num
        super().__init__()
        self.node_type = RawNodeType(node_type)

    def __repr__(self):
        return f"RawNodeChecker({self.node_type})"

    @property
    def is_cont(self):
        return True if self.node_type is RawNodeType.cont else False

    @classmethod
    def evolve(cls, node_type_evolved, cont_parents, disc_parents):
        """Method to create final Node Checker linked to node after stage 2"""
        return NodeChecker(node_type_evolved, cont_parents, disc_parents)


class NodeChecker(Checker):
    def __init__(self, node_type, cont_parents, disc_parents):
        super().__init__()
        self.node_type = NodeType(node_type)

        if self.node_type in discrete_nodes:
            self.discrete = True

        self.has_disc_parents = True if disc_parents else False
        self.has_cont_parents = True if cont_parents else False

    def __repr__(self):
        return f"NodeChecker({self.node_type})"

    def node_validation(self):
        if self.has_disc_parents:
            if self.node_type in (
                NodeType.mixture_gaussian,
                NodeType.gaussian,
                NodeType.logit,
            ):
                return False

        if self.has_cont_parents:
            if self.node_type is NodeType.discrete:
                return False

        if not self.has_cont_parents and not self.has_disc_parents:
            if self.node_type not in (
                NodeType.discrete,
                NodeType.gaussian,
                NodeType.mixture_gaussian,
            ):
                return False

        return True

    @property
    def is_mixture(self):
        return True if self.node_type in mixture_nodes else False

    @property
    def is_disc(self):
        return True if self.node_type in discrete_nodes else False

    @property
    def is_cont(self):
        return True if self.node_type in continuous_nodes else False


class NetworkChecker(Checker):
    def __init__(self, descriptor):
        super().__init__()
        self.RESTRICTIONS = [("cont", "disc"), ("cont", "disc_num")]
        self.checker_descriptor = {
            "types": {
                node_name: RawNodeChecker(node_type)
                for node_name, node_type in descriptor["types"].items()
            },
            "signs": {
                node_name: NodeSign(1 if v == "pos" else -1)
                for node_name, v in descriptor["signs"].items()
            },
        }
        if all(
            node_checker.is_cont
            for node_checker in self.checker_descriptor["types"].values()
        ):
            self.network_type = NetworkType.continuous
        elif all(
            not node_checker.is_cont
            for node_checker in self.checker_descriptor["types"].values()
        ):
            self.network_type = NetworkType.discrete
        else:
            self.network_type = NetworkType.hybrid

    def is_restricted_pair(self, node1, node2):
        node_type_checkers = self.checker_descriptor["types"]
        if (
            node_type_checkers[node1].node_type.name,
            node_type_checkers[node2].node_type.name,
        ) in self.RESTRICTIONS:
            return True
        else:
            return False

    def get_checker_rules(self):
        return {
            "descriptor": self.checker_descriptor,
            "restriction_rule": self.is_restricted_pair,
        }

    @property
    def is_disc(self):
        return True if self.network_type is NetworkType.discrete else False

    # def get_cont_nodes(self):
    #     return [name for name, checker in self.checker_descriptor["types"].items() if checker.is_cont]


def is_model(model):
    try:
        methods = dir(model)
        if "fit" in methods:
            return True
        else:
            return False
    except Exception:
        return False


# checker = NodeChecker(node_type="GaussianNode", cont_parents=[1,2], disc_parents=[])
# print(checker.node_type.discrete)

# checker = NetworkChecker(
#     {
#         "types": {"Node1": "cont", "Node2": "cont", "Node3": "disc"},
#         "signs": {"Node1": "pos", "Node2": "neg"},
#     }
# )
# print(checker.checker_descriptor)
