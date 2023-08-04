import os
import re
import random

import numpy as np
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm
from bamt.log import logger_network
from bamt.networks.base import BaseNetwork, STORAGE
import pandas as pd
from typing import Optional, Dict, Union, List
from bamt.builders.composite_builder import CompositeStructureBuilder, CompositeDefiner
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
        self.encoders = {}

    def add_nodes(self, descriptor: Dict[str, Dict[str, str]]):
        """
        Function for initializing nodes in Bayesian Network
        descriptor: dict with types and signs of nodes
        """
        self.descriptor = descriptor

        worker_1 = CompositeDefiner(descriptor=descriptor, regressor=None)
        self.nodes = worker_1.vertices

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
                and len(node.cont_parents + node.disc_parents) > 0
            ):
                self.set_classifiers(
                    {node.name: ml_models_dict[parent_models[node.name]]()}
                )
                print(
                    f"{ml_models_dict[parent_models[node.name]]} classifier has been set for {node.name}"
                )
            elif (
                type(node).__name__ == "CompositeContinuousNode"
                and parent_models[node.name] is not None
                and len(node.cont_parents + node.disc_parents) > 0
            ):
                self.set_regressor(
                    {node.name: ml_models_dict[parent_models[node.name]]()}
                )
                print(
                    f"{ml_models_dict[parent_models[node.name]]} regressor has been set for {node.name}"
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

    def sample(
        self,
        n: int,
        models_dir: Optional[str] = None,
        progress_bar: bool = True,
        evidence: Optional[Dict[str, Union[str, int, float]]] = None,
        as_df: bool = True,
        predict: bool = False,
        parall_count: int = 1,
    ) -> Union[None, pd.DataFrame, List[Dict[str, Union[str, int, float]]]]:
        """
        Sampling from Bayesian Network
        n: int number of samples
        evidence: values for nodes from user
        parall_count: number of threads. Defaults to 1.
        """
        from joblib import Parallel, delayed

        random.seed()
        if not self.distributions.items():
            logger_network.error(
                "Parameter learning wasn't done. Call fit_parameters method"
            )
            return None
        if evidence:
            for node in self.nodes:
                if (node.type == "Discrete") & (node.name in evidence.keys()):
                    if not (isinstance(evidence[node.name], str)):
                        evidence[node.name] = str(int(evidence[node.name]))

        def wrapper():
            output = {}
            for node in self.nodes:
                parents = node.cont_parents + node.disc_parents
                if evidence and node.name in evidence.keys():
                    output[node.name] = evidence[node.name]
                else:
                    if not parents:
                        pvals = None
                    else:
                        if self.type == "Discrete":
                            pvals = [str(output[t]) for t in parents]
                        else:
                            pvals = [output[t] for t in parents]

                        # If any nan from parents, sampling from node blocked.
                        if any(pd.isnull(pvalue) for pvalue in pvals):
                            output[node.name] = np.nan
                            continue

                    node_data = self.distributions[node.name]
                    if models_dir and ("hybcprob" in node_data.keys()):
                        for obj, obj_data in node_data["hybcprob"].items():
                            if "serialization" in obj_data.keys():
                                if "gaussian" in node.type.lower():
                                    model_type = "regressor"
                                else:
                                    model_type = "classifier"
                                if (
                                    obj_data["serialization"] == "joblib"
                                    and obj_data[f"{model_type}_obj"]
                                ):
                                    new_path = (
                                        models_dir
                                        + f"\\{node.name.replace(' ', '_')}\\{obj}.joblib.compressed"
                                    )
                                    node_data["hybcprob"][obj][
                                        f"{model_type}_obj"
                                    ] = new_path

                    if predict:
                        output[node.name] = node.predict(node_data, pvals=pvals)
                    else:
                        output[node.name] = node.choose(node_data, pvals=pvals)
            return output

        if progress_bar:
            # seq = Parallel(n_jobs=parall_count)(
            #     delayed(wrapper)() for _ in tqdm(range(n), position=0, leave=True)
            # )
            seq = []
            for _ in tqdm(range(n), position=0, leave=True):
                result = wrapper()
                seq.append(result)
        else:
            seq = Parallel(n_jobs=parall_count)(delayed(wrapper)() for _ in range(n))
        seq_df = pd.DataFrame.from_dict(seq, orient="columns")
        seq_df.dropna(inplace=True)
        print("Descriptor \n", self.descriptor)
        cont_nodes = [
            c.name
            for c in self.nodes
            if type(c).__name__
            not in ("DiscreteNode", "LogitNode", "CompositeDiscreteNode")
        ]
        print("Cont_nodes \n", cont_nodes)
        positive_columns = [
            c for c in cont_nodes if self.descriptor["signs"][c] == "pos"
        ]
        seq_df = seq_df[(seq_df[positive_columns] >= 0).all(axis=1)]
        seq_df.reset_index(inplace=True, drop=True)
        seq = seq_df.to_dict("records")
        sample_output = pd.DataFrame.from_dict(seq, orient="columns")

        if as_df:
            sample_output = self._decode_categorical_data(sample_output)
            return sample_output
        else:
            return seq

    def fit_parameters(self, data: pd.DataFrame, dropna: bool = True, n_jobs: int = -1):
        """
        Base function for parameter learning
        """
        if dropna:
            data = data.dropna()
            data.reset_index(inplace=True, drop=True)

        if not os.path.isdir(STORAGE):
            os.makedirs(STORAGE)

        # init folder
        if not os.listdir(STORAGE):
            os.makedirs(os.path.join(STORAGE, "0"))

        index = sorted([int(id) for id in os.listdir(STORAGE)])[-1] + 1
        os.makedirs(os.path.join(STORAGE, str(index)))

        data = self._encode_categorical_data(data)

        # Turn all discrete values to str for learning algorithm
        if "disc_num" in self.descriptor["types"].values():
            columns_names = [
                name
                for name, t in self.descriptor["types"].items()
                if t in ["disc_num"]
            ]
            data[columns_names] = data.loc[:, columns_names].astype("str")

        def worker(node):
            return node.fit_parameters(data)

        # results = Parallel(n_jobs=n_jobs)(delayed(worker)(node) for node in self.nodes)

        results = [worker(node) for node in self.nodes]

        for result, node in zip(results, self.nodes):
            self.distributions[node.name] = result

    def _encode_categorical_data(self, data):
        for column in data.select_dtypes(include=["object", "string"]).columns:
            encoder = LabelEncoder()
            data[column] = encoder.fit_transform(data[column])
            self.encoders[column] = encoder
        return data

    def _decode_categorical_data(self, data):
        data = data.apply(lambda col: pd.to_numeric(col).astype(int) if col.dtype == 'object' else col)
        for column, encoder in self.encoders.items():
            data[column] = encoder.inverse_transform(data[column])
        return data
