from abc import ABC, abstractmethod


class Graph(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def add_edge(self):
        pass

    @abstractmethod
    def remove_edge(self):
        pass

    @abstractmethod
    def get_parents(self):
        pass

    @abstractmethod
    def get_children(self):
        pass

    @abstractmethod
    def get_edges(self):
        pass

    @abstractmethod
    def get_nodes(self):
        pass
