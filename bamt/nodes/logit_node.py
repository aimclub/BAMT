import numpy as np

import pickle
import joblib
import random

from .base import BaseNode
from .schema import LogitParams
from bamt.log import logger_nodes

from sklearn import linear_model
from pandas import DataFrame
from typing import Optional, List, Union


class LogitNode(BaseNode):
    """
    Main class for logit node
    """

    def __init__(self, name, classifier: Optional[object] = None):
        super(LogitNode, self).__init__(name)
        if classifier is None:
            classifier = linear_model.LogisticRegression(
                multi_class="multinomial", solver="newton-cg", max_iter=100
            )
        self.classifier = classifier
        self.type = "Logit" + f" ({type(self.classifier).__name__})"

    @BaseNode.encode_categorical_data_if_any
    def fit_parameters(self, data: DataFrame, **kwargs) -> LogitParams:
        model_ser = None
        path = None

        parents = self.disc_parents + self.cont_parents
        self.classifier.fit(X=data[parents].values, y=data[self.name].values, **kwargs)
        serialization = self.choose_serialization(self.classifier)

        if serialization == "pickle":
            ex_b = pickle.dumps(self.classifier, protocol=4)
            # model_ser = ex_b.decode('latin1').replace('\'', '\"')
            model_ser = ex_b.decode("latin1")
            serialization_name = "pickle"
        else:
            logger_nodes.warning(
                f"{self.name}::Pickle failed. BAMT will use Joblib. | "
                + str(serialization.args[0])
            )

            path = self.get_path_joblib(self.name, specific=self.name.replace(" ", "_"))

            joblib.dump(self.classifier, path, compress=True, protocol=4)
            serialization_name = "joblib"
        return {
            "classes": list(self.classifier.classes_),
            "classifier_obj": path or model_ser,
            "classifier": type(self.classifier).__name__,
            "serialization": serialization_name,
        }

    def choose(self, node_info: LogitParams, pvals: List[Union[float]]) -> str:
        """
        Return value from Logit node
        params:
        node_info: nodes info from distributions
        pvals: parent values
        """

        rindex = 0

        if len(node_info["classes"]) > 1:
            if node_info["serialization"] == "joblib":
                model = joblib.load(node_info["classifier_obj"])
            else:
                # str_model = node_info["classifier_obj"].decode('latin1').replace('\'', '\"')
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
        """
        Return prediction from Logit node
        params:
        node_info: nodes info from distributions
        pvals: parent values
        """

        if len(node_info["classes"]) > 1:
            if node_info["serialization"] == "joblib":
                model = joblib.load(node_info["classifier_obj"])
            else:
                # str_model = node_info["classifier_obj"].decode('latin1').replace('\'', '\"')
                a = node_info["classifier_obj"].encode("latin1")
                model = pickle.loads(a)

            pred = model.predict(np.array(pvals).reshape(1, -1))[0]

            return str(pred)

        else:
            return str(node_info["classes"][0])
