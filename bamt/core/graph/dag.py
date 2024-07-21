from pgmpy.base.DAG import DAG
from bamt.core.graph.graph import Graph

from networkx import DiGraph, topological_sort
from bamt.core.nodes.node import Node

from typing import Type, Sequence
from bamt.loggers.logger import logger_graphs
from bamt.local_typing.node_types import continuous_nodes


class DirectedAcyclicGraph(Graph):
    def __init__(self):
        super().__init__()
        self.nodes: list[Type[Node]] = []
        self.edges: list[Sequence] = []

    def has_cycle(self):
        pass

    def from_container(self, container):
        pass

    def from_networkx(self, net):
        pass

    def get_family(self, descriptor):
        """
                A function that updates each node accordingly structure;
                """
        if not self.nodes:
            logger_graphs.error("Vertex list is None")
            return None
        if not self.edges:
            logger_graphs.error("Edges list is None")
            return None

        node_mapping = {
            node_name: {"disc_parents": [], "cont_parents": [], "children": []} for node_name in self.nodes
        }

        for edge in self.edges:
            parent, child = edge[0], edge[1]

            if descriptor["types"][parent] in continuous_nodes:
                node_mapping[child]["cont_parents"].append(parent)
            else:
                node_mapping[child]["disc_parents"].append(parent)
        return node_mapping

    @staticmethod
    def top_order(nodes: list[Type[Node]],
                  edges: list[Sequence]) -> list[str]:
        """
        Function for topological sorting
        """
        G = DiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        return list(topological_sort(G))

    def from_pgmpy(self, pgmpy_dag: DAG):
        self.nodes = pgmpy_dag.nodes
        self.edges = pgmpy_dag.edges

    # def __getattr__(self, item):
    #     return getattr(self._networkx_graph, item)
    #
    # def __setattr__(self, key, value):
    #     setattr(self._networkx_graph, key, value)
    #
    # def __delattr__(self, item):
    #     delattr(self._networkx_graph, item)

    # @property
    # def networkx_graph(self):
    #     return self._networkx_graph
