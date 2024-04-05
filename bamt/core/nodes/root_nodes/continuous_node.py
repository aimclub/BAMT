from .root_node import RootNode


class ContinuousNode(RootNode):
    def __init__(self):
        super().__init__()
        self._distribution = None

    def __str__(self):
        return "Continuous Node with " + str(self._distribution)
