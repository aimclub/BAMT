import pytest
import itertools
import bamt.networks as networks
import bamt.preprocessors as pp
from bamt.log import logger_preprocessor, logger_network
from pgmpy.estimators import K2Score
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import logging

# disable bamt preprocessor logger
logger_preprocessor.disabled = True

# disable warnings from networks
logger_network.setLevel(level=logging.ERROR)

from tqdm import tqdm
from functools import partialmethod

# disable tqdm globally in runtime
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

# disable golem logger at all
root_logger = logging.getLogger()
for hndlr in root_logger.handlers:
    root_logger.removeHandler(hndlr)

class Builder:
    def __init__(self):
        self.data_paths = {
            "Continuous": "data/benchmark/auto_price.csv",
            "Discrete": "tests/hack_discrete/hack_data.csv",
            "Hybrid": "data/benchmark/new_thyroid.csv",
        }

        self.tail = {
            "Continuous": ["Continuous", "target"],
            "Discrete": ["Discrete", "Tectonic regime"],
            "Hybrid": ["Hybrid", "target"],
        }

        self.scoring = [("K2", K2Score), ("BIC", ), ("MI", )]
        self.optimizer = ["HC"]
        self.use_mixture = [False, True]
        self.has_logit = [False, True]

        self.static = {}

        self.dynamic = {
            "Continuous": [self.use_mixture, [False], self.optimizer, self.scoring],
            "Discrete": [[False], [False], self.optimizer, self.scoring],
            "Hybrid": [self.use_mixture, self.has_logit, self.optimizer, self.scoring],
        }

    def create_from_config(self):
        """Method to collect data from config"""
        self.static = dict(
            Discrete=[self.data_paths["Discrete"], *self.tail["Discrete"]],
            Continuous=[self.data_paths["Continuous"], *self.tail["Continuous"]],
            Hybrid=[self.data_paths["Hybrid"], *self.tail["Hybrid"]],
            evo=[False, False, "Evo", self.scoring[0]],
        )

    def create_evo_item(self, net_type):
        evo_item = self.static["evo"][:]
        evo_item.insert(0, self.data_paths[net_type])
        evo_item.extend(self.tail[net_type])
        return evo_item

    @staticmethod
    def insert_list(loc, what, to):
        new = to[:]
        new[loc:loc] = what
        return new

    def create_net_items(self, net_type):
        static = self.static[net_type][:]
        dynamic_part = map(list, itertools.product(*self.dynamic[net_type]))
        return list(map(lambda x: self.insert_list(1, x, static), dynamic_part))

    def get_params(self):
        self.create_from_config()
        params = []
        for net_type in ["Discrete", "Continuous", "Hybrid"]:
            params.extend(
                self.create_net_items(net_type) + [self.create_evo_item(net_type)]
            )
        return params


params = Builder().get_params()

def initialize_bn(bn_type, use_mixture, has_logit):
    if bn_type == "Discrete":
        bn = networks.DiscreteBN()
    elif bn_type == "Continuous":
        bn = networks.ContinuousBN(use_mixture=use_mixture)
    elif bn_type == "Hybrid":
        bn = networks.HybridBN(has_logit=has_logit, use_mixture=use_mixture)
    elif bn_type == "Composite":
        bn = networks.CompositeBN()
    return bn


def prepare_data(directory):
    data = pd.read_csv(directory, index_col=0)
    train, test = train_test_split(data, test_size=0.33, random_state=42)

    encoder = preprocessing.LabelEncoder()
    discretizer = preprocessing.KBinsDiscretizer(
        n_bins=5, encode="ordinal", strategy="quantile"
    )

    p = pp.Preprocessor([("encoder", encoder), ("discretizer", discretizer)])
    discretized_data, est = p.apply(train)
    info = p.info
    return info, discretized_data, train, test


class TestNetwork:
    # Checking the equality of predictions (trained and loaded network) before and after saving
    @pytest.mark.parametrize(
        "directory, use_mixture, has_logit, optimizer, scoring, bn_type, target", params
    )
    def test_1(
        self, directory, use_mixture, has_logit, optimizer, scoring, bn_type, target
    ):
        bn = initialize_bn(bn_type, use_mixture, has_logit)
        info, discretized_data, train, test = prepare_data(directory)
        bn.add_nodes(info)
        if bn_type != "Composite":
            bn.add_edges(
                discretized_data,
                optimizer=optimizer,
                scoring_function=scoring,
                progress_bar=False,
            )
        else:
            bn.add_edges(train, progress_bar=False)

        assert bn.edges != []
        bn.fit_parameters(train)
        predict = bn.predict(
            test[[x for x in test.columns if x != target]], progress_bar=False
        )
        bn.save("bn")

        bn = initialize_bn(bn_type, use_mixture, has_logit)
        bn.load("bn.json")
        predict_loaded = bn.predict(
            test[[x for x in test.columns if x != target]], progress_bar=False
        )

        try:
            assert_frame_equal(pd.DataFrame(predict), pd.DataFrame(predict_loaded))
        except AssertionError:
            print(
                f"params: {dict(zip(['use_mixture', 'has_logit', 'optimizer', 'scoring', 'bn_type'], use_mixture, has_logit, optimizer, scoring, bn_type))}"
            )
            raise

    # Checking the prediction algorithm (trained network) before and after saving
    @pytest.mark.parametrize(
        "directory, use_mixture, has_logit, optimizer, scoring, bn_type, target", params
    )
    def test_2(
        self, directory, use_mixture, has_logit, optimizer, scoring, bn_type, target
    ):

        bn = initialize_bn(bn_type, use_mixture, has_logit)
        info, discretized_data, train, test = prepare_data(directory)
        bn.add_nodes(info)
        if bn_type != "Composite":
            bn.add_edges(
                discretized_data,
                optimizer=optimizer,
                scoring_function=scoring,
                progress_bar=False,
            )
        else:
            bn.add_edges(train)


        bn.fit_parameters(train)
        predict = bn.predict(
            test[[x for x in test.columns if x != target]], progress_bar=False
        )
        bn.save("bn")

        predict2 = bn.predict(
            test[[x for x in test.columns if x != target]], progress_bar=False
        )

        try:
            assert_frame_equal(pd.DataFrame(predict), pd.DataFrame(predict2))
        except AssertionError:
            print(
                f"params: {dict(zip(['use_mixture', 'has_logit', 'optimizer', 'scoring', 'bn_type'], use_mixture, has_logit, optimizer, scoring, bn_type))}"
            )
            raise

    # Checking network predictions without edges
    @pytest.mark.parametrize(
        "directory, use_mixture, has_logit, optimizer, scoring, bn_type, target", params
    )
    def test_3(
        self, directory, use_mixture, has_logit, optimizer, scoring, bn_type, target
    ):

        bn = initialize_bn(bn_type, use_mixture, has_logit)
        info, discretized_data, train, test = prepare_data(directory)
        bn.add_nodes(info)
        bn.fit_parameters(train)

        predict = bn.predict(
            test[[x for x in test.columns if x != target]], progress_bar=False
        )

        try:
            if info["types"][target] == "cont":
                if use_mixture:
                    mean = bn.distributions[target]["mean"]
                    w = bn.distributions[target]["coef"]
                    sample = 0
                    for ind, wi in enumerate(w):
                        sample += wi * mean[ind][0]
                else:
                    sample = train[target].mean()

                assert np.all(np.array(predict[target]) == sample)

            elif info["types"][target] == "disc_num":
                most_frequent = train[target].value_counts().index[0]
                assert np.all(np.array(predict[target]) == most_frequent)

        except AssertionError:
            print(
                f"params: {dict(zip(['use_mixture', 'has_logit', 'optimizer', 'scoring', 'bn_type'], use_mixture, has_logit, optimizer, scoring, bn_type))}"
            )
            raise

    # Checking the network trained on the 1 sample
    @pytest.mark.skip(reason="network will not learn from 1 sample")
    @pytest.mark.parametrize(
        "directory, use_mixture, has_logit, optimizer, scoring, bn_type, target", params
    )
    def test_4(
        self, directory, use_mixture, has_logit, optimizer, scoring, bn_type, target
    ):
        if use_mixture == False:

            bn = initialize_bn(bn_type, use_mixture, has_logit)
            info, discretized_data, train, test = prepare_data(directory)

            bn.add_nodes(info)

            train_data_1 = pd.DataFrame(train.iloc[0].to_dict(), index=[0])
            disc_data_1 = pd.DataFrame(discretized_data.iloc[0].to_dict(), index=[0])

            if bn_type != "Composite":
                bn.add_edges(
                    disc_data_1,
                    optimizer=optimizer,
                    scoring_function=scoring,
                    progress_bar=False,
                )
            else:
                bn.add_edges(train_data_1)

            bn.fit_parameters(train_data_1)

            predict = bn.predict(
                test[[x for x in test.columns if x != target]], progress_bar=False
            )

            assert np.all(np.array(predict[target]) == train_data_1[target][0])
        else:
            pass
