from abc import ABC, abstractmethod


class ProbabilisticStructuralModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, data, parameters_estimator):
        pass

    @abstractmethod
    def predict(self, data):
        pass
