from abc import ABC


class Graph(ABC):
    def __init__(self):
        self.nodes = []
        self.edges = []

    def __repr__(self):
        return f"{self.__class__.__name__} with \nNodes:{self.nodes}\nEdges:{self.edges}"
