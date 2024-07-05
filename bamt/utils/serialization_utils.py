import os
import pickle
from typing import Union, Tuple

import joblib

import bamt.utils.check_utils as check_utils
from bamt.log import logger_nodes


class ModelsSerializer:
    def __init__(self, bn_name, models_dir):
        self.bn_name = bn_name
        self.models_dir = models_dir
        self.serialization = None

        if os.path.isdir(models_dir):
            if bn_name in os.listdir(models_dir):
                raise AssertionError(f"Name must be unique. | {os.listdir(models_dir)}")

    @staticmethod
    def choose_serialization(model) -> Tuple[str, Union[Exception, int]]:
        try:
            ex_b = pickle.dumps(model, protocol=4)
            model_ser = ex_b.decode("latin1").replace("'", '"')

            if type(model).__name__ == "CatBoostRegressor":
                a = model_ser.encode("latin1")
            else:
                a = model_ser.replace('"', "'").encode("latin1")

            classifier_body = pickle.loads(a)
            return "pickle", 1
        except Exception:
            return "joblib", -1

    def get_path_joblib(self, models_dir, node_name: str) -> str:
        """
        Args:
            node_name: name of node
            specific: more specific unique name for node.
            For example, combination.

        Return:
            Path to node.
        """
        path = os.path.join(models_dir, self.bn_name, f"{node_name.replace(' ', '_')}")

        if not os.path.isdir(path):
            os.makedirs(
                os.path.join(models_dir, self.bn_name, f"{node_name.replace(' ', '_')}")
            )
        return path

    def serialize_instance(self, instance: dict, model_type, node_name, specific=False):
        """Every distribution contains a dict with params with models"""
        model_ser = None

        model = instance[f"{model_type}_obj"]
        if not check_utils.is_model(model):
            return instance

        serialization, status = self.choose_serialization(model)
        if status == -1:
            logger_nodes.warning(
                f"{node_name}:{'' if not specific else specific}::Pickle failed. BAMT will use Joblib."
            )
        if serialization == "pickle":
            ex_b = pickle.dumps(model, protocol=4)
            model_ser = ex_b.decode("latin1")
        elif serialization == "joblib":
            path = self.get_path_joblib(
                models_dir=self.models_dir, node_name=node_name.replace(" ", "_")
            )
            if not specific:
                destination = f"{node_name.replace(' ', '_')}.joblib.compressed"
            else:
                destination = f"{specific}.joblib.compressed"
            path = os.path.abspath(os.path.join(path, destination))
            joblib.dump(model, path, compress=True, protocol=4)
        else:
            logger_nodes.error("Serialization detection failed.")
            return
        instance["serialization"] = serialization
        instance[f"{model_type}_obj"] = model_ser or path
        return instance

    def serialize(self, distributions):
        result = {}
        for node_name, [node_type, dist] in distributions.items():
            result[node_name] = {}
            model_type = "regressor" if "Gaussian" in node_type else "classifier"
            if "Conditional" in node_type:
                result[node_name]["hybcprob"] = {}
                for combination, dist_nested in dist["hybcprob"].items():
                    instance_serialized = self.serialize_instance(
                        instance=dist_nested,
                        model_type=model_type,
                        node_name=node_name,
                        specific=combination,
                    )
                    result[node_name]["hybcprob"][combination] = instance_serialized
            else:
                instance_serialized = self.serialize_instance(
                    instance=dist, model_type=model_type, node_name=node_name
                )
                result[node_name] = instance_serialized
        return result


class Deserializer:
    def __init__(self, models_dir):
        self.models_dir = models_dir

    @staticmethod
    def deserialize_instance(instance: dict, model_type):
        model_repr = instance[f"{model_type}_obj"]
        if model_repr is None:
            return instance

        serialization = instance["serialization"]

        if serialization == "pickle":
            bytes_model = model_repr.encode("latin1")
            model = pickle.loads(bytes_model)
        else:
            model = joblib.load(model_repr)

        instance[f"{model_type}_obj"] = model
        return instance

    def apply(self, distributions):
        result = {}
        for node_name, [node_type, dist] in distributions.items():
            model_type = "regressor" if "Gaussian" in node_type else "classifier"
            result[node_name] = {}
            if "Conditional" in node_type:
                result[node_name]["hybcprob"] = {}
                for combination, dist_nested in dist["hybcprob"].items():
                    instance_deserialized = self.deserialize_instance(
                        instance=dist_nested,
                        model_type=model_type,
                    )
                    result[node_name]["hybcprob"][combination] = instance_deserialized
            else:
                instance_deserialized = self.deserialize_instance(
                    instance=dist, model_type=model_type
                )
                result[node_name] = instance_deserialized
        return result
