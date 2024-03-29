from .probabilistic_structural_model import ProbabilisticStructuralModel
from abc import abstractmethod


class BayesianNetwork(ProbabilisticStructuralModel):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def sample(self):
        pass
