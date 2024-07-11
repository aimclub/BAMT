from abc import abstractmethod
from typing import Any

import numpy as np

from golem.core.dag.graph import Graph


class SurrogateModel:
    """
    Model for evaluating fitness function without time-consuming fitting pipeline
    """
    @abstractmethod
    def __call__(self, graph: Graph, **kwargs: Any):
        raise NotImplementedError()


class RandomValuesSurrogateModel(SurrogateModel):
    """
        Model for evaluating fitness function based on returning random values for any model
    """
    def __call__(self, graph: Graph, **kwargs: Any):
        return np.random.random(1)
