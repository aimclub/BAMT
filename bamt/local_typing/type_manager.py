from pandas import DataFrame
from bamt.loggers.logger import logger_type_manager
# from bamt.core.nodes.root_nodes.continuous_node import ContinuousNode
# from bamt.core.nodes.root_nodes.discrete_node import DiscreteNode
# from bamt.core.nodes.child_nodes.conditional_continuous_node import ConditionalContinuousNode
# from bamt.core.nodes.child_nodes.conditional_discrete_node import ConditionalDiscreteNode

from .node_types import RawNodeType, NodeSign, continuous_nodes, NodeType


# todo: type aliases
class TypeManager:
    def __init__(self):
        pass

    def nodes_types(self, data: DataFrame):
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
                column_type[c] = RawNodeType.disc
            elif data[c].dtype.name in cont:
                column_type[c] = RawNodeType.cont
            elif data[c].dtype.name in disc_numerical:
                column_type[c] = RawNodeType.disc_num
            else:
                logger_type_manager.error(f"Unsupported data type. Dtype: {data[c].dtypes}")

        return column_type

    def nodes_signs(self, nodes_types: dict, data: DataFrame):
        """Function to define sign of the node
           neg - if node has negative values
           pos - if node has only positive values

        Args:
            data (pd.DataFrame): input dataset
            nodes_types (dict): dict with nodes_types

        Returns:
            dict: output dictionary where 'key' - node name and 'value' - sign of data
        """
        if list(nodes_types.keys()) != data.columns.to_list():
            logger_type_manager.error("Nodes_types dictionary is not full.")
            return {}
        columns_sign = dict()
        for c in data.columns.to_list():
            if nodes_types[c] == RawNodeType.cont:
                if (data[c] < 0).any():
                    columns_sign[c] = NodeSign.neg
                else:
                    columns_sign[c] = NodeSign.pos
        return columns_sign

    def get_descriptor(self, data) -> dict[str, dict[str, str]]:
        return {"types": self.nodes_types(data), "signs": self.nodes_signs(self.nodes_types(data), data)}

    @staticmethod
    def find_node_types(family: dict, descriptor):
        nodes2type = {}
        for node_name, node_family in family.items():
            node_raw_type = descriptor["types"][node_name]
            has_parents = len(node_family["disc_parents"] + node_family["cont_parents"]) > 0
            is_cont = node_raw_type in continuous_nodes

            match [has_parents, is_cont]:
                case [True, False]:
                    node_type = NodeType.conditional_discrete
                case [True, True]:
                    node_type = NodeType.conditional_continuous
                case [False, False]:
                    node_type = NodeType.root_discrete
                case [False, True]:
                    node_type = NodeType.root_continuous
                case _:
                    logger_type_manager.error("Node type was not found.")

            nodes2type[node_name] = node_type
        return nodes2type
