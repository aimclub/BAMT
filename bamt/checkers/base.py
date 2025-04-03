from bamt.local_typing.node_types import NodeType
from abc import ABC

# todo: abstract methods
class Checker(ABC):
    def __init__(self):
        self.node_type = NodeType
        self.is_mixture = False
        self.is_logit = False

    def validate_argument(self, arg):
        enumerator = self.node_type.__class__
        if isinstance(arg, str):
            arg = enumerator(arg)
        if arg not in self.node_type.__class__:
            assert TypeError("Wrong type of argument.")
        return True
