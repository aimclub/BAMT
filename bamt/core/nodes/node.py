from abc import ABC, abstractmethod


class Node(ABC):
    def __init__(self, name):
        self.name = name
        self.disc_parents = []
        self.cont_parents = []
        self.children = []

    @abstractmethod
    def get_children(self):
        pass

    @abstractmethod
    def get_parents(self):
        pass

    @abstractmethod
    def fit(self, X):
        pass
