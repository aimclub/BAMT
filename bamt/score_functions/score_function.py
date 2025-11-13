from abc import ABC, abstractmethod


class ScoreFunction(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def estimate(self):
        pass
