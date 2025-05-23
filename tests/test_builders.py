import itertools
import logging
import unittest

import pandas as pd

from bamt.builders.builders_base import StructureBuilder, VerticesDefiner
from bamt.builders.evo_builder import EvoStructureBuilder
from bamt.builders.hc_builder import HillClimbDefiner
from bamt.nodes.discrete_node import DiscreteNode
from bamt.nodes.gaussian_node import GaussianNode
from bamt.utils.MathUtils import precision_recall

logging.getLogger("builder").setLevel(logging.CRITICAL)
from functools import partialmethod
from tqdm import tqdm

# disable tqdm globally in runtime
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


class TestStructureBuilder(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(columns=["Node0", "Node1", "Node2"])
        self.descriptor = {
            "types": {"Node0": "cont", "Node1": "disc", "Node2": "disc_num"},
            "signs": {"Node0": "pos"},
        }
        self.SB = StructureBuilder(descriptor=self.descriptor)

    def test_restrict(self):
        self.SB.has_logit = True

        self.SB.restrict(data=self.data, init_nodes=None, bl_add=None)

        self.assertEqual(self.SB.black_list, [], msg="Restrict wrong edges.")

        # ---------
        self.SB.has_logit = False

        self.SB.restrict(data=self.data, init_nodes=None, bl_add=None)

        self.assertEqual(
            self.SB.black_list,
            [("Node0", "Node1"), ("Node0", "Node2")],
            msg="Restricted edges are allowed.",
        )

    def test_get_family(self):
        self.assertIsNone(self.SB.get_family())

        self.SB.skeleton["V"] = [
            GaussianNode(name="Node0"),
            DiscreteNode(name="Node1"),
            DiscreteNode(name="Node2"),
        ]
        self.assertIsNone(self.SB.get_family())
        # Note that the method get_family is not supposed to be used by user (only developer),
        # so we don't cover a case with restricted edges here (we did this in
        # the previous test).
        self.SB.skeleton["E"] = [
            ("Node1", "Node0"),
            ("Node2", "Node1"),
            ("Node2", "Node0"),
        ]
        self.SB.get_family()

        # Node: [[cont_parents], [disc_parents], [children]]
        data = [
            [[], [], ["Node1", "Node0"]],
            [[], ["Node2"], ["Node0"]],
            [[], ["Node1", "Node2"], []],
        ]
        for node_nummer in range(3):
            self.assertEqual(
                self.SB.skeleton["V"][node_nummer].cont_parents, data[node_nummer][0]
            )
            self.assertEqual(
                self.SB.skeleton["V"][node_nummer].disc_parents, data[node_nummer][1]
            )
            self.assertEqual(
                self.SB.skeleton["V"][node_nummer].children, data[node_nummer][2]
            )


class TestVerticesDefiner(unittest.TestCase):
    def setUp(self):
        self.descriptor = {
            "types": {
                "Node0": "cont",
                "Node1": "cont",
                "Node2": "cont",
                "Node3": "cont",
                "Node4": "disc",
                "Node5": "disc",
                "Node6": "disc_num",
                "Node7": "disc_num",
            },
            "signs": {"Node0": "pos", "Node1": "neg"},
        }

        self.VD = VerticesDefiner(descriptor=self.descriptor, regressor=None)

    def test_first_level(self):
        self.assertEqual(
            self.VD.vertices,
            [
                GaussianNode(name="Node0"),
                GaussianNode(name="Node1"),
                GaussianNode(name="Node2"),
                GaussianNode(name="Node3"),
                DiscreteNode(name="Node4"),
                DiscreteNode(name="Node5"),
                DiscreteNode(name="Node6"),
                DiscreteNode(name="Node7"),
            ],
        )

    def test_overwrite_vetrex(self):
        self.assertEqual(self.VD.skeleton, {"V": [], "E": []})

        def reload():
            self.VD.skeleton["V"] = self.VD.vertices
            self.VD.skeleton["E"] = [
                ("Node0", "Node7"),
                ("Node0", "Node1"),
                ("Node0", "Node2"),
                ("Node0", "Node5"),
                ("Node4", "Node2"),
                ("Node4", "Node5"),
                ("Node4", "Node6"),
                ("Node4", "Node3"),
            ]
            self.VD.get_family()

        data = {
            "True, True": {
                "Node0": "MixtureGaussian",
                "Node4": "Discrete",
                "Node7": "Logit (LogisticRegression)",
                "Node1": "MixtureGaussian",
                "Node2": "ConditionalMixtureGaussian",
                "Node5": "ConditionalLogit (LogisticRegression)",
                "Node6": "Discrete",
                "Node3": "ConditionalMixtureGaussian",
            },
            "True, False": {
                "Node0": "MixtureGaussian",
                "Node4": "Discrete",
                "Node7": "Discrete",
                "Node1": "MixtureGaussian",
                "Node2": "ConditionalMixtureGaussian",
                "Node5": "Discrete",
                "Node6": "Discrete",
                "Node3": "ConditionalMixtureGaussian",
            },
            "False, True": {
                "Node0": "Gaussian (LinearRegression)",
                "Node4": "Discrete",
                "Node7": "Logit (LogisticRegression)",
                "Node1": "Gaussian (LinearRegression)",
                "Node2": "ConditionalGaussian (LinearRegression)",
                "Node5": "ConditionalLogit (LogisticRegression)",
                "Node6": "Discrete",
                "Node3": "ConditionalGaussian (LinearRegression)",
            },
            "False, False": {
                "Node0": "Gaussian (LinearRegression)",
                "Node4": "Discrete",
                "Node7": "Discrete",
                "Node1": "Gaussian (LinearRegression)",
                "Node2": "ConditionalGaussian (LinearRegression)",
                "Node5": "Discrete",
                "Node6": "Discrete",
                "Node3": "ConditionalGaussian (LinearRegression)",
            },
        }

        for use_mixture, has_logit in itertools.product([True, False], repeat=2):
            reload()
            self.VD.overwrite_vertex(
                has_logit=has_logit,
                use_mixture=use_mixture,
                classifier=None,
                regressor=None,
            )
            self.assertEqual(
                {node.name: node.type for node in self.VD.skeleton["V"]},
                data[f"{use_mixture}, {has_logit}"],
                msg=f"failed on use_mixture={use_mixture} and has_logit={has_logit}",
            )


class TestHillClimbDefiner(unittest.TestCase):
    def setUp(self):
        self.descriptor = {
            "signs": {
                "Depth": "pos",
                "Gross": "pos",
                "Netpay": "pos",
                "Permeability": "pos",
                "Porosity": "pos",
            },
            "types": {
                "Depth": "cont",
                "Gross": "cont",
                "Lithology": "disc",
                "Netpay": "cont",
                "Period": "disc",
                "Permeability": "cont",
                "Porosity": "cont",
                "Structural setting": "disc",
                "Tectonic regime": "disc",
            },
        }
        self.data = {
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

    def test_apply_K2(self):
        hcd = HillClimbDefiner(
            data=pd.DataFrame(self.data),
            descriptor=self.descriptor,
            scoring_function=("K2",),
        )

        hcd.apply_K2(
            data=pd.DataFrame(self.data),
            init_edges=None,
            progress_bar=False,
            remove_init_edges=False,
            white_list=None,
        )

        right_edges = [
            ["Tectonic regime", "Structural setting"],
            ["Tectonic regime", "Gross"],
            ["Tectonic regime", "Permeability"],
            ["Tectonic regime", "Porosity"],
            ["Tectonic regime", "Depth"],
            ["Tectonic regime", "Period"],
            ["Tectonic regime", "Netpay"],
            ["Period", "Structural setting"],
            ["Lithology", "Structural setting"],
            ["Lithology", "Gross"],
            ["Lithology", "Permeability"],
            ["Lithology", "Porosity"],
            ["Lithology", "Depth"],
            ["Lithology", "Tectonic regime"],
            ["Lithology", "Period"],
            ["Lithology", "Netpay"],
            ["Gross", "Structural setting"],
            ["Gross", "Period"],
            ["Netpay", "Structural setting"],
            ["Netpay", "Gross"],
            ["Netpay", "Permeability"],
            ["Netpay", "Porosity"],
            ["Netpay", "Depth"],
            ["Netpay", "Period"],
            ["Porosity", "Structural setting"],
            ["Porosity", "Gross"],
            ["Porosity", "Depth"],
            ["Porosity", "Period"],
            ["Permeability", "Structural setting"],
            ["Permeability", "Gross"],
            ["Permeability", "Depth"],
            ["Permeability", "Period"],
            ["Permeability", "Porosity"],
            ["Depth", "Structural setting"],
            ["Depth", "Gross"],
            ["Depth", "Period"],
        ]

        self.assertEqual(hcd.skeleton["E"], right_edges)

    def test_apply_group1(self):
        hcd = HillClimbDefiner(
            data=pd.DataFrame(self.data),
            descriptor=self.descriptor,
            scoring_function=("MI",),
        )

        hcd.restrict(data=pd.DataFrame(self.data), bl_add=None, init_nodes=None)
        hcd.apply_group1(
            data=pd.DataFrame(self.data),
            progress_bar=False,
            init_edges=None,
            remove_init_edges=False,
            white_list=None,
        )

        right_edges = [
            ["Lithology", "Depth"],
            ["Period", "Gross"],
            ["Netpay", "Gross"],
            ["Period", "Netpay"],
            ["Depth", "Period"],
            ["Depth", "Permeability"],
            ["Netpay", "Permeability"],
            ["Period", "Porosity"],
            ["Netpay", "Porosity"],
            ["Permeability", "Structural setting"],
            ["Netpay", "Structural setting"],
            ["Period", "Tectonic regime"],
            ["Netpay", "Tectonic regime"],
        ]

        self.assertEqual(hcd.skeleton["E"], right_edges)


class TestEvoStructureBuilder(unittest.TestCase):
    def setUp(self):
        self.data = pd.read_csv(r"data/benchmark/asia.csv", index_col=0)
        self.descriptor = {
            "types": {
                "asia": "disc",
                "tub": "disc",
                "smoke": "disc",
                "lung": "disc",
                "bronc": "disc",
                "either": "disc",
                "xray": "disc",
                "dysp": "disc",
            },
            "signs": {},
        }
        self.evo_builder = EvoStructureBuilder(
            data=self.data,
            descriptor=self.descriptor,
            regressor=None,
            has_logit=True,
            use_mixture=True,
        )
        # Replace this with your actual reference DAG
        self.reference_dag = [
            ("asia", "tub"),
            ("tub", "either"),
            ("smoke", "lung"),
            ("smoke", "bronc"),
            ("lung", "either"),
            ("bronc", "dysp"),
            ("either", "xray"),
            ("either", "dysp"),
        ]

    def test_build(self):
        # placeholder kwargs
        kwargs = {}
        self.evo_builder.build(
            data=self.data, classifier=None, regressor=None, verbose=False, **kwargs
        )

        obtained_dag = self.evo_builder.skeleton["E"]
        num_edges = len(obtained_dag)
        self.assertGreaterEqual(
            num_edges, 1, msg="Obtained graph should have at least one edge."
        )

        dist = precision_recall(obtained_dag, self.reference_dag)["SHD"]
        self.assertLess(
            dist,
            15,
            msg=f"Structural Hamming Distance should be less than 15, obtained SHD = {dist}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
