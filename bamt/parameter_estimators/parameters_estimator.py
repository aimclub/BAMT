from abc import ABC, abstractmethod


class ParametersEstimator(ABC):
    def __init__(self, network):
        self.network = network

    @abstractmethod
    def estimate(self):
        pass
