from abc import ABC, abstractmethod


class ScoreFunction(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def score(self, data):
        pass

    @abstractmethod
    def local_score(self, variable, parents):
        pass
