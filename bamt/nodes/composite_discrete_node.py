import pickle
import random
from typing import List, Optional, Union
from pandas import DataFrame
from .schema import LogitParams
from bamt.log import logger_nodes
import numpy as np
from .logit_node import LogitNode
import joblib


class CompositeDiscreteNode(LogitNode):
    """
    Class for composite discrete node.
    """

    def __init__(self, name, classifier: Optional[object] = None):
        super().__init__(name, classifier)
        self.type = "CompositeDiscrete" + f" ({type(self.classifier).__name__})"

    def fit_parameters(self, data: DataFrame):
        parents = self.disc_parents + self.cont_parents
        self.classifier.fit(data[parents].values, data[self.name].values)
        serialization = self.choose_serialization(self.classifier)

        model_ser, serialization_name, path = self.serialize_classifier(serialization)
        return {
            "classes": list(self.classifier.classes_),
            "classifier_obj": path or model_ser,
            "classifier": type(self.classifier).__name__,
            "serialization": serialization_name,
        }

    def choose(self, node_info: LogitParams, pvals: List[Union[float]]) -> str:
        rindex = 0

        if len(node_info["classes"]) > 1:
            if node_info["serialization"] == "joblib":
                model = joblib.load(node_info["classifier_obj"])
            else:
                a = node_info["classifier_obj"].encode("latin1")
                model = pickle.loads(a)
            distribution = model.predict_proba(np.array(pvals).reshape(1, -1))[0]

            # choose
            rand = random.random()
            lbound = 0
            ubound = 0
            for interval in range(len(node_info["classes"])):
                ubound += distribution[interval]
                if lbound <= rand < ubound:
                    rindex = interval
                    break
                else:
                    lbound = ubound

            return str(node_info["classes"][rindex])

        else:
            return str(node_info["classes"][0])

    def predict(self, node_info: LogitParams, pvals: List[Union[float]]) -> str:
        if len(node_info["classes"]) > 1:
            if node_info["serialization"] == "joblib":
                model = joblib.load(node_info["classifier_obj"])
            else:
                a = node_info["classifier_obj"].encode("latin1")
                model = pickle.loads(a)

            pred = model.predict(np.array(pvals).reshape(1, -1))[0]

            return str(pred)

        else:
            return str(node_info["classes"][0])

    def serialize_classifier(self, serialization):
        if serialization == "pickle":
            ex_b = pickle.dumps(self.classifier, protocol=4)
            model_ser = ex_b.decode("latin1")
            serialization_name = "pickle"
            path = None
        else:
            logger_nodes.warning(
                f"{self.name}::Pickle failed. BAMT will use Joblib. | "
                + str(serialization.args[0])
            )

            path = self.get_path_joblib(self.name, specific=self.name.replace(" ", "_"))

            joblib.dump(self.classifier, path, compress=True, protocol=4)
            model_ser = None
            serialization_name = "joblib"

        return model_ser, serialization_name, path
