from abc import abstractmethod

from .probabilistic_structural_model import ProbabilisticStructuralModel


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
