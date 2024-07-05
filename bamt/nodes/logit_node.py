import random
from typing import Optional, List, Union

import numpy as np
from pandas import DataFrame
from sklearn import linear_model

from .base import BaseNode
from .schema import LogitParams


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

    def fit_parameters(self, data: DataFrame, **kwargs) -> LogitParams:
        parents = self.cont_parents + self.disc_parents
        self.classifier.fit(X=data[parents].values, y=data[self.name].values, **kwargs)

        return {
            "classes": list(self.classifier.classes_),
            "classifier_obj": self.classifier,
            "classifier": type(self.classifier).__name__,
            "serialization": None,
        }

    def get_dist(self, node_info, pvals):
        if len(node_info["classes"]) > 1:
            model = node_info["classifier_obj"]
            if type(self).__name__ == "CompositeDiscreteNode":
                pvals = [int(item) if isinstance(item, str) else item for item in pvals]

            return model.predict_proba(np.array(pvals).reshape(1, -1))[0]
        else:
            return np.array([1.0])

    def choose(self, node_info: LogitParams, pvals: List[Union[float]]) -> str:
        """
        Return value from Logit node
        params:
        node_info: nodes info from distributions
        pvals: parent values
        """

        rindex = 0

        distribution = self.get_dist(node_info, pvals)

        if len(node_info["classes"]) > 1:
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

    @staticmethod
    def predict(node_info: LogitParams, pvals: List[Union[float]]) -> str:
        """
        Return prediction from Logit node
        params:
        node_info: nodes info from distributions
        pvals: parent values
        """

        if len(node_info["classes"]) > 1:
            model = node_info["classifier_obj"]
            pred = model.predict(np.array(pvals).reshape(1, -1))[0]

            return str(pred)

        else:
            return str(node_info["classes"][0])
