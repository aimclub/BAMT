import re
from bamt.networks.base import BaseNetwork
from bamt.log import logger_network
import pandas as pd
import numpy as np
from typing import Optional, Dict
from bamt.builders.builders_base import ParamDict
from bamt.builders.composite_builder import CompositeStructureBuilder
from bamt.utils.composite_utils.MLUtils import MlModels


class CompositeBN(BaseNetwork):
    """
    Composite Bayesian Network with Machine Learning Models support
    """

    def __init__(self):
        super(CompositeBN, self).__init__()
        self._allowed_dtypes = ["cont", "disc", "disc_num"]
        self.type = "Composite"
        self.parent_models = {}

    def add_edges(
        self,
        data: pd.DataFrame,
        progress_bar: bool = True,
        classifier: Optional[object] = None,
        regressor: Optional[object] = None,
        **kwargs,
    ):
        worker = CompositeStructureBuilder(
            data=data, descriptor=self.descriptor, regressor=regressor
        )

        worker.build(
            data=data,
            classifier=classifier,
            regressor=regressor,
            progress_bar=progress_bar,
            **kwargs,
        )

        # update family
        self.nodes = worker.skeleton["V"]
        self.edges = worker.skeleton["E"]
        self.parent_models = worker.parent_models_dict
        self.set_models(self.parent_models)

    def set_models(self, parent_models):
        ml_models = MlModels()
        ml_models_dict = ml_models.dict_models
        for node in self.nodes:
            if (
                type(node).__name__ == "CompositeDiscreteNode"
                and parent_models[node.name] is not None
            ):
                self.set_classifiers(
                    {node.name: ml_models_dict[parent_models[node.name]]}
                )
                print(f"{ml_models_dict[parent_models[node.name]]} classifier has been set for {node.name}")
            elif (
                type(node).__name__ == "CompositeContinuousNode"
                and parent_models[node.name] is not None
            ):
                self.set_regressor(
                    {node.name: ml_models_dict[parent_models[node.name]]}
                )
            else:
                pass

    def set_classifiers(self, classifiers: Dict[str, object]):
        """
        Set classifiers for logit nodes.
        classifiers: dict with node_name and Classifier
        """
        for node in self.nodes:
            if node.name in classifiers.keys():
                node.classifier = classifiers[node.name]
                node.type = re.sub(
                    r"\([\s\S]*\)", f"({type(node.classifier).__name__})", node.type
                )
            else:
                continue

    def set_regressor(self, regressors: Dict[str, object]):
        """
        Set regressor for gaussian nodes.
        classifiers: dict with node_name and Classifier
        """
        for node in self.nodes:
            if node.name in regressors.keys():
                node.regressor = regressors[node.name]
                node.type = re.sub(
                    r"\([\s\S]*\)", f"({type(node.regressor).__name__})", node.type
                )
            else:
                continue
