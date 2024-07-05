from abc import ABC, abstractmethod


class Node(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def get_children(self):
        pass

    @abstractmethod
    def get_parents(self):
        pass

    @abstractmethod
    def fit(self, X):
        pass
