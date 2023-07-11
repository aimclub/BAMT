import pathlib as pl
import json
import unittest

from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn import preprocessing as pp

import logging

import pandas as pd

from bamt.networks.hybrid_bn import BaseNetwork, HybridBN

from bamt.nodes.gaussian_node import GaussianNode
from bamt.nodes.discrete_node import DiscreteNode
from bamt.nodes.logit_node import LogitNode
from bamt import preprocessors

logging.getLogger("network").setLevel(logging.CRITICAL)


class TestCaseBase(unittest.TestCase):
    def assertIsFile(self, path):
        if not pl.Path(path).resolve().is_file():
            raise AssertionError("File does not exist: %s" % str(path))

    def assertIsDir(self, path):
        if not pl.Path(path).resolve().is_dir():
            raise AssertionError("Direction does not exist: %s" % str(path))

    def prepare_bn_and_data(self):
        # prepare bn where models were set by set_model
        hack_data = pd.read_csv("data/real data/hack_processed_with_rf.csv")[
            [
                "Tectonic regime",
                "Period",
                "Lithology",
                "Structural setting",
                "Gross",
                "Netpay",
                "Porosity",
                "Permeability",
                "Depth",
            ]
        ]

        encoder = pp.LabelEncoder()
        discretizer = pp.KBinsDiscretizer(
            n_bins=5, encode="ordinal", strategy="quantile"
        )

        p = preprocessors.Preprocessor(
            [("encoder", encoder), ("discretizer", discretizer)]
        )

        discretized_data, est = p.apply(hack_data)

        self.bn = HybridBN(has_logit=True)
        info = p.info

        self.bn.add_nodes(info)

        self.bn.add_edges(
            discretized_data, scoring_function=("BIC",), progress_bar=False
        )

        self.bn.set_regressor(
            regressors={
                "Depth": CatBoostRegressor(
                    logging_level="Silent", allow_writing_files=False
                ),
                "Gross": RandomForestRegressor(),
                "Porosity": DecisionTreeRegressor(),
            }
        )
        return hack_data


class TestBaseNetwork(TestCaseBase):
    def setUp(self):
        self.bn = BaseNetwork()

        self.nodes = [
            GaussianNode(name="Node0"),
            DiscreteNode(name="Node1"),
            GaussianNode(name="Node2"),
        ]

        self.edges = [("Node0", "Node1"), ("Node1", "Node2")]
        self.descriptor = {
            "types": {"Node0": "cont", "Node1": "disc", "Node2": "cont"},
            "signs": {"Node0": "pos", "Node1": "neg"},
        }

    def test_validate(self):
        descriptor_t = {"types": {"Node0": "Abstract", "Node1": "Abstract"}}
        descriptor_f = {"types": {"Node0": "Abstract", "Node1": "cont"}}

        self.assertFalse(self.bn.validate(descriptor_f))
        self.assertTrue(self.bn.validate(descriptor_t))

    def test_update_descriptor(self):
        self.bn.descriptor = self.descriptor
        # Nodes out
        self.bn.nodes = [GaussianNode(name="Node0")]
        self.bn.update_descriptor()
        self.assertEqual({"Node0": "cont"}, self.bn.descriptor["types"])

    # It uses only Vertices Definer, test of this is in builders tests.
    def test_add_nodes(self):
        pass

    def test_add_edges(self):
        # It uses builders
        pass

    def test_calculate_weights(self):
        pass

    def test_set_nodes(self):
        class MyNode:
            def __init__(self, name):
                self.name = name

        # set without mapping
        self.assertIsNone(self.bn.set_nodes(nodes=[GaussianNode(name="Node0")]))

        map = {
            "types": {"Node0": "cont", "Node1": "disc", "Node2": "cont"},
            "signs": {},
        }

        self.bn.set_nodes(nodes=self.nodes, info=map)
        self.assertEqual(self.bn.nodes, self.nodes)

        self.bn.set_nodes(
            nodes=[MyNode(name="Node-1"), MyNode("Node-2")],
            info={"types": {"Node-1": "cont", "Node-2": "disc"}, "signs": {}},
        )
        self.assertEqual(self.bn.nodes, [])

    def test_set_edges(self):
        self.edges.extend([(0, 1), (pd.NA, "1")])

        self.bn.has_logit = False
        self.bn.set_nodes(nodes=self.nodes, info=self.descriptor)
        self.bn.set_edges(edges=self.edges)

        self.assertEqual([("Node1", "Node2")], self.bn.edges)

    # The test consists of 2 previous methods that are tested,
    # plus methods of builders, they are tested as well.
    def test_set_structure(self):
        pass

    # Node testing.
    def test_param_validation(self):
        pass

    def test_set_parameters(self):
        pass

    def test_save_params(self):
        self.bn.distributions = {"AAA": "BBB"}
        self.assertIsNone(self.bn.save_params("out.txt"))

        self.assertTrue(self.bn.save_params("out.json"))

        self.assertIsFile("out.json")

        self.assertEqual(json.load(open("out.json")), self.bn.distributions)
        pl.Path("out.json").unlink()

    def test_save_structure(self):
        self.bn.edges = self.edges
        self.assertIsNone(self.bn.save_structure("out.txt"))
        self.assertTrue(self.bn.save_structure("out.json"))
        self.assertIsFile("out.json")
        f = json.load(open("out.json"))
        for i in range(len(f)):
            self.assertEqual(list(self.bn.edges[i]), f[i])
        pl.Path("out.json").unlink()

    # Covers by prev tests.
    def test_save(self):
        pass

    def test_fit_parameters(self):
        from bamt.networks.base import STORAGE

        # here we test only initialization of the folder
        self.bn.has_logit = True
        self.bn.nodes = [LogitNode(name="Node0")]

        try:
            self.bn.fit_parameters(data=pd.DataFrame())
        except KeyError:
            pass

        self.assertIsDir(STORAGE)
        self.assertIsDir(pl.Path(STORAGE).joinpath("0"))

    def test_joblib_pathsave(self):
        hack_data = self.prepare_bn_and_data()
        self.bn.fit_parameters(hack_data)

        self.assertGreater(
            self.bn.sample(100, progress_bar=False).size, 0, "Sampling is broken"
        )

        saveloc = self.bn.distributions["Gross"]["hybcprob"]["['COMPRESSION']"][
            "regressor_obj"
        ]

        self.assertIsFile(saveloc)

    def test_sample(self):
        data = {
            "Tectonic regime": [
                0,
                1,
                4,
                4,
                0,
                2,
                0,
                0,
                0,
                0,
                3,
                1,
                0,
                3,
                0,
                1,
                4,
                0,
                4,
                3,
                4,
                0,
                1,
                1,
                1,
                0,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                3,
                2,
                3,
                2,
                3,
                3,
                3,
                0,
            ],
            "Period": [
                3,
                1,
                4,
                4,
                1,
                1,
                0,
                0,
                3,
                5,
                3,
                9,
                0,
                5,
                0,
                3,
                5,
                3,
                2,
                4,
                4,
                1,
                5,
                7,
                7,
                7,
                1,
                1,
                1,
                1,
                4,
                6,
                8,
                4,
                4,
                5,
                4,
                7,
                5,
                5,
                0,
            ],
            "Lithology": [
                2,
                4,
                6,
                4,
                2,
                2,
                2,
                2,
                4,
                4,
                4,
                4,
                1,
                4,
                1,
                4,
                4,
                4,
                5,
                3,
                2,
                2,
                2,
                4,
                1,
                1,
                3,
                4,
                4,
                4,
                4,
                2,
                0,
                3,
                4,
                4,
                4,
                4,
                4,
                4,
                2,
            ],
            "Structural setting": [
                2,
                6,
                10,
                10,
                7,
                5,
                8,
                8,
                2,
                2,
                6,
                6,
                3,
                7,
                3,
                6,
                10,
                9,
                3,
                0,
                0,
                7,
                6,
                6,
                6,
                7,
                6,
                6,
                6,
                6,
                8,
                2,
                9,
                4,
                7,
                6,
                1,
                8,
                4,
                4,
                3,
            ],
            "Gross": [
                1,
                3,
                1,
                3,
                1,
                0,
                2,
                3,
                0,
                4,
                4,
                4,
                0,
                3,
                0,
                0,
                3,
                4,
                0,
                4,
                3,
                2,
                2,
                4,
                0,
                4,
                1,
                2,
                2,
                4,
                2,
                4,
                3,
                1,
                1,
                1,
                2,
                3,
                0,
                2,
                1,
            ],
            "Netpay": [
                3,
                2,
                1,
                4,
                2,
                0,
                2,
                2,
                1,
                4,
                3,
                4,
                0,
                3,
                1,
                1,
                0,
                4,
                1,
                3,
                4,
                3,
                3,
                4,
                0,
                4,
                0,
                1,
                2,
                4,
                2,
                3,
                2,
                1,
                2,
                0,
                2,
                4,
                1,
                3,
                0,
            ],
            "Porosity": [
                3,
                0,
                4,
                3,
                3,
                1,
                0,
                0,
                3,
                0,
                2,
                1,
                2,
                3,
                0,
                2,
                3,
                0,
                0,
                4,
                2,
                4,
                2,
                2,
                1,
                1,
                1,
                3,
                3,
                2,
                4,
                3,
                1,
                4,
                4,
                4,
                3,
                1,
                4,
                4,
                0,
            ],
            "Permeability": [
                4,
                0,
                3,
                3,
                2,
                1,
                1,
                1,
                1,
                0,
                4,
                4,
                1,
                3,
                1,
                4,
                3,
                0,
                0,
                3,
                0,
                1,
                2,
                0,
                2,
                2,
                1,
                2,
                3,
                4,
                3,
                2,
                2,
                2,
                4,
                4,
                3,
                0,
                4,
                4,
                0,
            ],
            "Depth": [
                1,
                4,
                3,
                4,
                1,
                3,
                1,
                3,
                1,
                4,
                3,
                4,
                1,
                2,
                1,
                4,
                0,
                4,
                0,
                0,
                3,
                2,
                3,
                2,
                2,
                3,
                4,
                2,
                2,
                4,
                1,
                0,
                2,
                0,
                4,
                0,
                1,
                2,
                0,
                0,
                3,
            ],
        }
        nodes = [
            DiscreteNode(name="Tectonic regime"),
            DiscreteNode(name="Period"),
            DiscreteNode(name="Lithology"),
            DiscreteNode(name="Structural setting"),
            DiscreteNode(name="Gross"),
            DiscreteNode(name="Netpay"),
            DiscreteNode(name="Porosity"),
            DiscreteNode(name="Permeability"),
            DiscreteNode(name="Depth"),
        ]

        self.bn.set_nodes(nodes, info={"types": {k.name: "disc" for k in nodes}})
        self.bn.set_edges(
            [
                ["Tectonic regime", "Period"],
                ["Structural setting", "Period"],
                ["Tectonic regime", "Lithology"],
                ["Lithology", "Structural setting"],
            ]
        )
        self.bn.fit_parameters(pd.DataFrame.from_records(data))
        self.assertIsNotNone(self.bn.sample(50, as_df=False, progress_bar=False))

    def test_predict(self):
        seq = {
            "Tectonic regime": [
                0,
                1,
                4,
                4,
                0,
                2,
                0,
                0,
                0,
                0,
                3,
                1,
                0,
                3,
                0,
                1,
                4,
                0,
                4,
                3,
                4,
                0,
                1,
                1,
                1,
                0,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                3,
                2,
                3,
                2,
                3,
                3,
                3,
                0,
            ],
            "Period": [
                3,
                1,
                4,
                4,
                1,
                1,
                0,
                0,
                3,
                5,
                3,
                9,
                0,
                5,
                0,
                3,
                5,
                3,
                2,
                4,
                4,
                1,
                5,
                7,
                7,
                7,
                1,
                1,
                1,
                1,
                4,
                6,
                8,
                4,
                4,
                5,
                4,
                7,
                5,
                5,
                0,
            ],
            "Lithology": [
                2,
                4,
                6,
                4,
                2,
                2,
                2,
                2,
                4,
                4,
                4,
                4,
                1,
                4,
                1,
                4,
                4,
                4,
                5,
                3,
                2,
                2,
                2,
                4,
                1,
                1,
                3,
                4,
                4,
                4,
                4,
                2,
                0,
                3,
                4,
                4,
                4,
                4,
                4,
                4,
                2,
            ],
            "Structural setting": [
                2,
                6,
                10,
                10,
                7,
                5,
                8,
                8,
                2,
                2,
                6,
                6,
                3,
                7,
                3,
                6,
                10,
                9,
                3,
                0,
                0,
                7,
                6,
                6,
                6,
                7,
                6,
                6,
                6,
                6,
                8,
                2,
                9,
                4,
                7,
                6,
                1,
                8,
                4,
                4,
                3,
            ],
            "Gross": [
                1,
                3,
                1,
                3,
                1,
                0,
                2,
                3,
                0,
                4,
                4,
                4,
                0,
                3,
                0,
                0,
                3,
                4,
                0,
                4,
                3,
                2,
                2,
                4,
                0,
                4,
                1,
                2,
                2,
                4,
                2,
                4,
                3,
                1,
                1,
                1,
                2,
                3,
                0,
                2,
                1,
            ],
            "Netpay": [
                3,
                2,
                1,
                4,
                2,
                0,
                2,
                2,
                1,
                4,
                3,
                4,
                0,
                3,
                1,
                1,
                0,
                4,
                1,
                3,
                4,
                3,
                3,
                4,
                0,
                4,
                0,
                1,
                2,
                4,
                2,
                3,
                2,
                1,
                2,
                0,
                2,
                4,
                1,
                3,
                0,
            ],
            "Porosity": [
                3,
                0,
                4,
                3,
                3,
                1,
                0,
                0,
                3,
                0,
                2,
                1,
                2,
                3,
                0,
                2,
                3,
                0,
                0,
                4,
                2,
                4,
                2,
                2,
                1,
                1,
                1,
                3,
                3,
                2,
                4,
                3,
                1,
                4,
                4,
                4,
                3,
                1,
                4,
                4,
                0,
            ],
            "Permeability": [
                4,
                0,
                3,
                3,
                2,
                1,
                1,
                1,
                1,
                0,
                4,
                4,
                1,
                3,
                1,
                4,
                3,
                0,
                0,
                3,
                0,
                1,
                2,
                0,
                2,
                2,
                1,
                2,
                3,
                4,
                3,
                2,
                2,
                2,
                4,
                4,
                3,
                0,
                4,
                4,
                0,
            ],
            "Depth": [
                1,
                4,
                3,
                4,
                1,
                3,
                1,
                3,
                1,
                4,
                3,
                4,
                1,
                2,
                1,
                4,
                0,
                4,
                0,
                0,
                3,
                2,
                3,
                2,
                2,
                3,
                4,
                2,
                2,
                4,
                1,
                0,
                2,
                0,
                4,
                0,
                1,
                2,
                0,
                0,
                3,
            ],
        }
        data = pd.DataFrame.from_records(seq)
        nodes = [
            DiscreteNode(name="Tectonic regime"),
            DiscreteNode(name="Period"),
            DiscreteNode(name="Lithology"),
            DiscreteNode(name="Structural setting"),
        ]

        self.bn.set_nodes(nodes, info={"types": {k.name: "disc" for k in nodes}})
        self.bn.set_edges(
            [
                ["Tectonic regime", "Period"],
                ["Structural setting", "Period"],
                ["Tectonic regime", "Lithology"],
                ["Lithology", "Structural setting"],
            ]
        )
        self.bn.fit_parameters(data)

        result = self.bn.predict(data.iloc[:, :3], parall_count=2, progress_bar=False)
        self.assertNotEqual(result, {})

        for v in result.values():
            for item in v:
                self.assertFalse(pd.isna(item))


class TestBigBraveBN(unittest.SkipTest):
    pass


if __name__ == "__main__":
    unittest.main(verbosity=3)
