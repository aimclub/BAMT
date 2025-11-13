from .child_node import ChildNode


class ConditionalContinuousNode(ChildNode):
    def __init__(self):
        super().__init__()
        self._model = None
