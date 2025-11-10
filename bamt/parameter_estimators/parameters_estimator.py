from abc import ABC, abstractmethod


class ParametersEstimator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def estimate(self):
        pass
