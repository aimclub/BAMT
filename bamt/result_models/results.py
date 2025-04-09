from abc import ABC, abstractmethod

class Results(ABC):
    @abstractmethod
    def get(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass