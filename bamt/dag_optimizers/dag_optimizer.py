from abc import ABC, abstractmethod


class DAGOptimizer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def optimize(self, data, scorer, formatter=None):
        pass
