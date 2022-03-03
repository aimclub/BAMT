import networkx as nx
from bamt.log import logger_preprocessor
from pandas import DataFrame
from bamt.Nodes import BaseNode

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
        disc = ['str', 'O', 'b', 'categorical', 'object']
        disc_numerical = ['int32', 'int64']
        cont = ['float32', 'float64']
        if data[c].dtype.name in disc:
            column_type[c] = 'disc'
        elif data[c].dtype.name in cont:
            column_type[c] = 'cont'
        elif data[c].dtype.name in disc_numerical:
            column_type[c] = 'disc_num'
        else:
            logger_preprocessor.error(f'Unsupported data type. Dtype: {data[c].dtypes}')

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
        if nodes_types[c] == 'cont':
            if (data[c] < 0).any():
                columns_sign[c] = 'neg'
            else:
                columns_sign[c] = 'pos'
    return columns_sign


def get_descriptor(data) -> Dict[str, Dict[str, str]]:
    return {'types': nodes_types(data),
            'signs': nodes_signs(nodes_types(data), data)}


def toporder(nodes: List[Type[BaseNode]], edges: List[Tuple]) -> List[List[str]]:
    """
    Function for topological sorting
    """
    G = nx.DiGraph()
    G.add_nodes_from([node.name for node in nodes])
    G.add_edges_from(edges)
    return list(nx.topological_sort(G))
