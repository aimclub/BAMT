from .child_node import ChildNode


class ConditionalContinuousNode(ChildNode):
    def __init__(self, name):
        super().__init__(name)
        self._model = None
