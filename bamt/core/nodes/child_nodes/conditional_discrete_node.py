from .child_node import ChildNode


class ConditionalDiscreteNode(ChildNode):
    def __init__(self, name):
        super().__init__(name)
        self._model = None

    def __repr__(self):
        return f"{self.name}. Conditional Discrete Node with {self._model}"

    def get_children(self):
        pass

    def get_parents(self):
        pass

    def fit(self, X):
        pass
