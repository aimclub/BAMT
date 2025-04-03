from bamt.checkers.base import Checker
from bamt.local_typing.node_types import (
    RawNodeType,
    NodeType,
    discrete_nodes,
    continuous_nodes
)


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

    @property
    def is_disc(self):
        return True if self.node_type in [RawNodeType, RawNodeType.disc_num] else False

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

        self.root = (
            True if not self.has_cont_parents and not self.has_disc_parents else False
        )

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

        if not (self.has_cont_parents or self.has_disc_parents):
            if self.node_type not in (
                NodeType.discrete,
                NodeType.gaussian,
                NodeType.mixture_gaussian,
            ):
                return False

        return True

    @property
    def does_require_regressor(self):
        return True if self.node_type in nodes_with_regressors else False

    @property
    def does_require_classifier(self):
        return True if self.node_type in nodes_with_classifiers else False

    @property
    def has_combinations(self):
        return True if self.node_type in nodes_with_combinations else False

    @property
    def is_mixture(self):
        return True if self.node_type in mixture_nodes else False

    @property
    def is_logit(self):
        return True if self.node_type in logit_nodes else False

    @property
    def is_disc(self):
        return True if self.node_type in discrete_nodes else False

    @property
    def is_cont(self):
        return True if self.node_type in continuous_nodes else False
