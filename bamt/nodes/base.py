import pickle
from typing import Union

from bamt.utils.check_utils import NodeType


class BaseNode(object):
    """
    Base class for nodes.
    """

    def __init__(self, name: str):
        """
        :param name: name for node (taken from column name)
        type: node type
        disc_parents: list with discrete parents
        cont_parents: list with continuous parents
        children: node's children
        """
        self.name = name
        self.type = NodeType(type(self).__name__)

        self.disc_parents = []
        self.cont_parents = []
        self.children = []

    def __repr__(self):
        model = getattr(self, 'regressor', False) or getattr(self, 'classifier', False)
        return f"{self.name} ({model if model else None})"

    def __eq__(self, other):
        if not isinstance(other, BaseNode):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return (
            self.name == other.name
            and self.type == other.type
            and self.disc_parents == other.disc_parents
            and self.cont_parents == other.cont_parents
            and self.children == other.children
        )

    @staticmethod
    def choose_serialization(model) -> Union[str, Exception]:
        try:
            ex_b = pickle.dumps(model, protocol=4)
            model_ser = ex_b.decode("latin1").replace("'", '"')

            if type(model).__name__ == "CatBoostRegressor":
                a = model_ser.encode("latin1")
            else:
                a = model_ser.replace('"', "'").encode("latin1")

            classifier_body = pickle.loads(a)
            return "pickle"
        except Exception as ex:
            return ex

    @staticmethod
    def get_dist(node_info, pvals):
        pass
