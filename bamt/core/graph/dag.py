from .graph import Graph
from networkx import DiGraph


class DirectedAcyclicGraph(Graph):
    def __init__(self):
        super().__init__()
        self._networkx_graph = DiGraph()

    def __getattr__(self, item):
        return getattr(self._networkx_graph, item)

    def __setattr__(self, key, value):
        setattr(self._networkx_graph, key, value)

    def __delattr__(self, item):
        delattr(self._networkx_graph, item)

    @property
    def networkx_graph(self):
        return self._networkx_graph
