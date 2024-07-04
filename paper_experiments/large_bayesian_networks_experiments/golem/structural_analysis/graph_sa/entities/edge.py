from typing import Tuple, List

from golem.core.dag.graph_node import GraphNode


class Edge:
    def __init__(self, parent_node: GraphNode, child_node: GraphNode):
        self.parent_node = parent_node
        self.child_node = child_node
        self.rating = None

    @staticmethod
    def from_tuple(edges_in_tuple: List[Tuple[GraphNode, GraphNode]]) -> List['Edge']:
        edges = []
        for edge in edges_in_tuple:
            edges.append(Edge(child_node=edge[1], parent_node=edge[0]))
        return edges
