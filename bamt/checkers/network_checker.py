from bamt.checkers.base import Checker
from bamt.checkers.node_checkers import RawNodeChecker

from bamt.local_typing.network_types import NetworkType
from bamt.local_typing.node_types import NodeSign


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

    def __getitem__(self, node_name):
        node_checker = self.checker_descriptor["types"][node_name]

        if node_checker.is_cont:
            signs = {"signs": self.checker_descriptor["signs"][node_name]}
        else:
            signs = {}

        return {
            "node_checker": node_checker,
        } | signs

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

    def has_mixture_nodes(self):
        if not getattr(
            next(iter(self.checker_descriptor["types"].values())), "is_mixture"
        ):
            return None

        if any(
            node_checker.is_mixture
            for node_checker in self.checker_descriptor["types"].values()
        ):
            return True
        else:
            return False

    def has_logit_nodes(self):
        if not getattr(
            next(iter(self.checker_descriptor["types"].values())), "is_logit"
        ):
            return None

        if any(
            node_checker.is_logit
            for node_checker in self.checker_descriptor["types"].values()
        ):
            return True
        else:
            return False

    def validate_load(self, input_dict, network):
        # check compatibility with father network.
        if not network.use_mixture:
            for node_name, node_data in input_dict["parameters"].items():
                node_checker = self[node_name]["node_checker"]
                if node_checker.is_disc:
                    continue
                else:
                    # Since we don't have information about types of nodes, we
                    # should derive it from parameters.
                    if not node_checker.has_combinations:
                        if list(node_data.keys()) == ["covars", "mean", "coef"]:
                            return "use_mixture"
                    else:
                        if any(
                            list(node_keys.keys()) == ["covars", "mean", "coef"]
                            for node_keys in node_data["hybcprob"].values()
                        ):
                            return "use_mixture"

        # check if edges before and after are the same.They can be different in
        # the case when user sets forbidden edges.
        if not network.has_logit:
            if not all(
                edges_before == [edges_after[0], edges_after[1]]
                for edges_before, edges_after in zip(input_dict["edges"], network.edges)
            ):
                # logger_network.error(
                #     f"This crucial parameter is not the same as father's parameter: has_logit."
                # )
                return False
        return True

    @property
    def is_disc(self):
        return True if self.network_type is NetworkType.discrete else False
