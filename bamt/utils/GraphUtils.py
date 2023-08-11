import networkx as nx
from pandas import DataFrame

from bamt.log import logger_preprocessor
from bamt.nodes.base import BaseNode

from typing import Dict, List, Tuple, Type


def nodes_types(data: DataFrame) -> Dict[str, str]:
    """
    Function to define the type of the node
           disc - discrete node
           cont - continuous
        Args:
            data: input dataset

        Returns:
            dict: output dictionary where 'key' - node name and 'value' - node type
    """

    column_type = dict()
    for c in data.columns.to_list():
        disc = ["str", "O", "b", "categorical", "object", "bool"]
        disc_numerical = ["int32", "int64"]
        cont = ["float32", "float64"]
        if data[c].dtype.name in disc:
            column_type[c] = "disc"
        elif data[c].dtype.name in cont:
            column_type[c] = "cont"
        elif data[c].dtype.name in disc_numerical:
            column_type[c] = "disc_num"
        else:
            logger_preprocessor.error(f"Unsupported data type. Dtype: {data[c].dtypes}")

    return column_type


def nodes_signs(nodes_types: dict, data: DataFrame) -> Dict[str, str]:
    """Function to define sign of the node
       neg - if node has negative values
       pos - if node has only positive values

    Args:
        data (pd.DataFrame): input dataset

    Returns:
        dict: output dictionary where 'key' - node name and 'value' - sign of data
    """
    if list(nodes_types.keys()) != data.columns.to_list():
        logger_preprocessor.error("Nodes_types dictionary is not full.")
        return
    columns_sign = dict()
    for c in data.columns.to_list():
        if nodes_types[c] == "cont":
            if (data[c] < 0).any():
                columns_sign[c] = "neg"
            else:
                columns_sign[c] = "pos"
    return columns_sign


def get_descriptor(data) -> Dict[str, Dict[str, str]]:
    return {"types": nodes_types(data), "signs": nodes_signs(nodes_types(data), data)}


def toporder(nodes: List[Type[BaseNode]], edges: List[Tuple]) -> List[List[str]]:
    """
    Function for topological sorting
    """
    G = nx.DiGraph()
    G.add_nodes_from([node.name for node in nodes])
    G.add_edges_from(edges)
    return list(nx.topological_sort(G))


class GraphAnalyzer(object):
    def __init__(self, bn):
        self.bn = bn

    def _isolate_structure(self, nodes):
        isolated_edges = []
        for edge in self.bn.edges:
            if edge[0] in nodes and edge[1] in nodes:
                isolated_edges.append(edge)
        return isolated_edges

    def markov_blanket(self, node_name: str):
        node = self.bn[node_name]

        parents = node.cont_parents + node.disc_parents
        children = node.children
        fremd_eltern = []

        for child in node.children:
            all_parents = self.bn[child].cont_parents + self.bn[child].disc_parents

            if all_parents == [node_name]:
                continue
            else:
                new = all_parents
            fremd_eltern.extend(new)

        nodes = parents + children + fremd_eltern + [node_name]

        edges = self._isolate_structure(nodes)
        return {"nodes": list(set(nodes)), "edges": edges}

    def _collect_height(self, node_name, height):
        nodes = []
        node = self.bn[node_name]
        if height <= 0:
            return []

        if height == 1:
            return node.disc_parents + node.cont_parents

        for parent in node.cont_parents + node.disc_parents:
            nodes.append(parent)
            nodes.extend(self._collect_height(parent, height=height - 1))
        return nodes

    def _collect_depth(self, node_name, depth):
        nodes = []
        node = self.bn[node_name]

        if depth <= 0:
            return []

        if depth == 1:
            return node.children

        for child in node.children:
            nodes.append(child)
            nodes.extend(self._collect_depth(child, depth=depth - 1))

        return nodes

    def find_family(self, *args):
        node_name, height, depth, with_nodes = args
        if not with_nodes:
            with_nodes = []
        else:
            with_nodes = list(with_nodes)
        nodes = (
            self._collect_depth(node_name, depth)
            + self._collect_height(node_name, height)
            + [node_name]
        )

        nodes = list(set(nodes + with_nodes))

        return {"nodes": nodes, "edges": self._isolate_structure(nodes + with_nodes)}
