# import matplotlib.pyplot as plt
import networkx as nx
from log import logger_preprocessor


def nodes_types(data):
    """Function to define the type of the node
           disc - discrete node
           cont - continuous
        Args:
            data (pd.DataFrame): input dataset

        Returns:
            dict: output dictionary where 'key' - node name and 'value' - node type
        """
    column_type = dict()
    for c in data.columns.to_list():
        disc = ['str', 'O', 'b']
        disc_numerical = ['int32', 'int64']
        cont = ['float32', 'float64']
        if data[c].dtypes in disc:
            column_type[c] = 'disc'
        elif data[c].dtype in cont:
            column_type[c] = 'cont'
        elif data[c].dtype in disc_numerical:
            column_type[c] = 'disc_num'
        else:
            logger_preprocessor.error(f'Unsupported data type. Dtype: {data[c].dtypes}')

    return column_type


def nodes_signs(nodes_types: dict, data):
    """Function to define sign of the node
           neg - if node has negative values
           pos - if node has only positive values

        Args:
            data (pd.DataFrame): input dataset

        Returns:
            dict: output dictionary where 'key' - node name and 'value' - sign of data
        """
    columns_sign = dict()
    for c in data.columns.to_list():
        if nodes_types[c] == 'cont':
            if (data[c] < 0).any():
                columns_sign[c] = 'neg'
            else:
                columns_sign[c] = 'pos'
    return columns_sign


def get_descriptor(data):
    return {'types': nodes_types(data),
            'signs': nodes_signs(nodes_types(data), data)}


def toporder(edges):
    G = nx.from_edgelist(edges, create_using=nx.DiGraph)
    # nx.draw(G, with_labels=True, font_weight='bold')
    # plt.show()
    return list(nx.topological_sort(G))