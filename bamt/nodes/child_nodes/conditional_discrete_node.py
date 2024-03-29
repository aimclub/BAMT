from .child_node import ChildNode


class ConditionalDiscreteNode(ChildNode):
    def __init__(self):
        super().__init__()
        self._model = None
