from bamt.config import config

from typing import Union

import pickle
import os

STORAGE = config.get(
    'NODES',
    'models_storage',
    fallback='models_storage is not defined')


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
        self.type = 'abstract'

        self.disc_parents = []
        self.cont_parents = []
        self.children = []

    def __repr__(self):
        return f"{self.name}"

    def __eq__(self, other):
        if not isinstance(other, BaseNode):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.name == other.name and \
            self.type == other.type and \
            self.disc_parents == other.disc_parents and \
            self.cont_parents == other.cont_parents and \
            self.children == other.children

    @staticmethod
    def choose_serialization(model) -> Union[str, Exception]:
        try:
            ex_b = pickle.dumps(model, protocol=4)
            model_ser = ex_b.decode('latin1').replace('\'', '\"')

            if type(model).__name__ == "CatBoostRegressor":
                a = model_ser.encode('latin1')
            else:
                a = model_ser.replace('\"', '\'').encode('latin1')

            classifier_body = pickle.loads(a)
            return 'pickle'
        except Exception as ex:
            return ex

    @staticmethod
    def get_path_joblib(node_name: str, specific: str = "") -> str:
        """
        Args:
            node_name: name of node
            specific: more specific unique name for node.
            For example, combination.

        Return:
            Path to save a joblib file.
        """
        if not isinstance(specific, str):
            specific = str(specific)

        index = str(int(os.listdir(STORAGE)[-1]))
        path_to_check = os.path.join(
            STORAGE,
            index,
            f"{node_name.replace(' ', '_')}")

        if not os.path.isdir(path_to_check):
            os.makedirs(
                os.path.join(
                    STORAGE,
                    index,
                    f"{node_name.replace(' ', '_')}"))

        path = os.path.abspath(
            os.path.join(path_to_check, f"{specific}.joblib.compressed"))
        return path
