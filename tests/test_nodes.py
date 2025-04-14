import logging
import unittest

import numpy as np
import pandas as pd

from bamt.networks.hybrid_bn import HybridBN
from bamt.nodes import *

logging.getLogger("nodes").setLevel(logging.CRITICAL)


class MyTest(unittest.TestCase):
    unittest.skip("This is an assertion.")

    def assertDist(self, dist):
        has_probs = ["discrete", "logit", "conditional_logit"]
        probs_or_params = dist.get()

        if dist.node_type in has_probs:
            return self.assertAlmostEqual(sum(probs_or_params[0]), 1)
        else:
            return self.assertEqual(
                len(probs_or_params), 2, msg=f"Error on {probs_or_params}"
            ) and self.assertTrue(probs_or_params[0].size)

    def assertDistMixture(self, dist):
        return self.assertEqual(len(dist.get()), 3, msg=f"Error on {dist}")


class TestBaseNode(MyTest):
    def setUp(self):
        np.random.seed(510)

        hybrid_bn = HybridBN(has_logit=True)

        info = {
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
            "signs": {"Node0": "pos", "Node1": "neg", "Node2": "neg", "Node3": "neg"},
        }

        data = pd.DataFrame(
            {
                "Node0": np.random.normal(0, 4, 30),
                "Node1": np.random.normal(0, 0.1, 30),
                "Node2": np.random.normal(0, 0.3, 30),
                "Node3": np.random.normal(0, 0.3, 30),
                "Node4": np.random.choice(["cat1", "cat2", "cat3"], 30),
                "Node5": np.random.choice(["cat4", "cat5", "cat6"], 30),
                "Node6": np.random.choice(["cat7", "cat8", "cat9"], 30),
                "Node7": np.random.choice(["cat7", "cat8", "cat9"], 30),
            }
        )

        nodes = [
            gaussian_node.GaussianNode(name="Node0"),
            gaussian_node.GaussianNode(name="Node1"),
            gaussian_node.GaussianNode(name="Node2"),
            gaussian_node.GaussianNode(name="Node3"),
            discrete_node.DiscreteNode(name="Node4"),
            discrete_node.DiscreteNode(name="Node5"),
            discrete_node.DiscreteNode(name="Node6"),
            discrete_node.DiscreteNode(name="Node7"),
        ]

        edges = [
            ("Node0", "Node7"),
            ("Node0", "Node1"),
            ("Node0", "Node2"),
            ("Node0", "Node5"),
            ("Node4", "Node2"),
            ("Node4", "Node5"),
            ("Node4", "Node6"),
            ("Node4", "Node3"),
        ]

        hybrid_bn.set_structure(info, nodes=nodes, edges=edges)
        hybrid_bn.fit_parameters(data)
        self.bn = hybrid_bn
        self.data = data
        self.info = info

    def test_equality(self):
        test = base.BaseNode(name="node0")
        first = base.BaseNode(name="node1")

        test_clone = test

        # different name
        self.assertFalse(test == first)
        self.assertTrue(test == test_clone)

        # different type
        test_clone.type = "gaussian"
        test.type = "gaussian"

        first.type = "discrete"

        self.assertFalse(test == first)
        self.assertTrue(test == test_clone)

        # different disc_parents
        disc_parents = [f"node{i}" for i in range(2, 6)]
        test.disc_parents, test_clone.disc_parents = disc_parents, disc_parents
        first = disc_parents[:3]

        self.assertFalse(test == first)
        self.assertTrue(test == test_clone)

        # different cont_parents
        cont_parents = [f"node{i}" for i in range(6, 10)]
        test.disc_parents, test_clone.disc_parents = cont_parents, cont_parents
        first = cont_parents[:3]

        self.assertFalse(test == first)
        self.assertTrue(test == test_clone)

        # different children
        children = [f"node{i}" for i in range(5)]
        test.disc_parents, test_clone.disc_parents = children, children
        first = children[:3]

        self.assertFalse(test == first)
        self.assertTrue(test == test_clone)

    def test_get_dist_mixture(self):
        hybrid_bn = HybridBN(use_mixture=True, has_logit=True)

        hybrid_bn.set_structure(self.info, self.bn.nodes, self.bn.edges)
        hybrid_bn.fit_parameters(self.data)

        mixture_gauss = hybrid_bn["Node1"]
        cond_mixture_gauss = hybrid_bn["Node2"]

        for i in range(-2, 0, 2):
            dist = hybrid_bn.get_dist(mixture_gauss.name, pvals={"Node0": i})
            self.assertDistMixture(dist)

        for i in range(-2, 0, 2):
            for j in self.data[cond_mixture_gauss.disc_parents[0]].unique().tolist():
                dist = hybrid_bn.get_dist(
                    cond_mixture_gauss.name, pvals={"Node0": float(i), "Node4": j}
                )
                self.assertDistMixture(dist)

    def test_get_dist(self):
        for node in self.bn.nodes:
            if not node.cont_parents + node.disc_parents:
                dist = self.bn.get_dist(node.name)
                self.assertDist(dist)
                continue

            if len(node.cont_parents + node.disc_parents) == 1:
                if node.disc_parents:
                    pvals = self.data[node.disc_parents[0]].unique().tolist()
                    parent = node.disc_parents[0]
                else:
                    pvals = range(-5, 5, 1)
                    parent = node.cont_parents[0]

                for pval in pvals:
                    dist = self.bn.get_dist(node.name, {parent: pval})
                    self.assertDist(dist)
            else:
                for i in self.data[node.disc_parents[0]].unique().tolist():
                    for j in range(-5, 5, 1):
                        dist = self.bn.get_dist(
                            node.name,
                            {node.cont_parents[0]: float(j), node.disc_parents[0]: i},
                        )
                        self.assertDist(dist)

    # ???
    def test_choose_serialization(self):
        pass


class TestDiscreteNode(unittest.TestCase):
    def setUp(self):
        self.node = discrete_node.DiscreteNode(name="test")
        self.data_dict = {
            "node0": np.random.normal(0, 4, 30),
            "node1": np.random.normal(0, 0.1, 30),
            "node2": np.random.normal(0, 0.3, 30),
            "test": np.random.choice(["cat1", "cat2", "cat3"], 30),
            "node4": np.random.choice(["cat4", "cat5", "cat6"], 30),
            "node5": np.random.choice(["cat7", "cat8", "cat9"], 30),
            "node6": np.random.choice(["cat7", "cat8", "cat9"], 30),
        }

        self.node.disc_parents = ["node4", "node5"]
        # self.node.cont_parents = ["node0", "node1"]
        self.node.children = ["node6"]

    def test_fit_parameters(self):
        params = self.node.fit_parameters(pd.DataFrame.from_records(self.data_dict))
        self.assertIsNotNone(params["vals"])
        self.assertNotEqual(params["vals"], [])

        for comb, probas in params["cprob"].items():
            self.assertAlmostEqual(sum(probas), 1, delta=1e-5)

    def test_choose(self):
        pvals = ["cat4", "cat7"]
        params = self.node.fit_parameters(pd.DataFrame.from_records(self.data_dict))

        self.assertTrue([self.node.choose(params, pvals) in params["vals"]])

    def test_predict(self):
        pvals = ["cat4", "cat7"]
        params = self.node.fit_parameters(pd.DataFrame.from_records(self.data_dict))

        self.assertTrue([self.node.predict(params, pvals) in params["vals"]])
        self.assertRaises(KeyError, self.node.predict, params, ["bad", "values"])


class TestGaussianNode(unittest.TestCase):
    def setUp(self):
        self.node = gaussian_node.GaussianNode(name="test")
        self.data_dict = {
            "node0": np.random.normal(1, 4, 30),
            "node1": np.random.normal(2, 0.1, 30),
            "foster-son": np.random.normal(2.5, 0.2, 30),
            "test": np.random.normal(3, 0.3, 30),
            "node2": np.random.choice(["cat1", "cat2", "cat3"], 30),
            "node4": np.random.choice(["cat4", "cat5", "cat6"], 30),
            "node5": np.random.choice(["cat7", "cat8", "cat9"], 30),
            "node6": np.random.choice(["cat7", "cat8", "cat9"], 30),
        }

        # self.node.disc_parents = ["node4", "node5"]
        self.node.cont_parents = ["node0", "node1"]
        self.node.children = ["node6"]

    def test_fit_parameters(self):
        node_without_parents = gaussian_node.GaussianNode(name="foster-son")
        node_without_parents.children = ["node6", "node5"]

        params_parents = self.node.fit_parameters(
            pd.DataFrame.from_records(self.data_dict)
        )
        params_foster = node_without_parents.fit_parameters(
            pd.DataFrame.from_records(self.data_dict)
        )

        self.assertIsNotNone(params_parents["regressor_obj"])
        self.assertTrue(pd.isna(params_parents["mean"]))
        self.assertIsNotNone(params_parents["std"])

        self.assertIsNone(params_foster["regressor_obj"])
        self.assertFalse(pd.isna(params_foster["mean"]))
        self.assertIsNotNone(params_foster["std"])

    def test_choose(self):
        pvals = [1.05, 1.95]
        params = self.node.fit_parameters(pd.DataFrame.from_records(self.data_dict))

        self.assertTrue(isinstance(self.node.choose(params, pvals), float))

    def test_predict(self):
        pvals = [1.05, 1.95]
        params = self.node.fit_parameters(pd.DataFrame.from_records(self.data_dict))
        self.assertTrue(isinstance(self.node.predict(params, pvals), float))
        self.assertRaises(ValueError, self.node.predict, params, ["bad", "values"])


class TestConditionalGaussianNode(unittest.TestCase):
    def setUp(self):
        self.node = conditional_gaussian_node.ConditionalGaussianNode(name="test")
        self.data_dict = {
            "node0": np.random.normal(1, 4, 30),
            "node1": np.random.normal(2, 0.1, 30),
            "foster-son": np.random.normal(2.5, 0.2, 30),
            "test": np.random.normal(3, 0.3, 30),
            "node2": np.random.choice(["cat1", "cat2", "cat3"], 30),
            "node4": np.random.choice(["cat4", "cat5", "cat6"], 30),
            "node5": np.random.choice(["cat7", "cat8", "cat9"], 30),
            "node6": np.random.choice(["cat7", "cat8", "cat9"], 30),
        }

        self.node.disc_parents = ["node4", "node5"]
        self.node.cont_parents = ["node0", "node1"]
        self.node.children = ["node6"]

    def fit_parameters(self, regressor=None):
        if regressor is not None:
            self.node.regressor = regressor
            self.node.type = "ConditionalGaussian" + f" ({type(regressor).__name__})"

        node_without_parents = conditional_gaussian_node.ConditionalGaussianNode(
            name="foster-son"
        )
        node_without_parents.children = ["node6", "node5"]

        params_parents = self.node.fit_parameters(
            pd.DataFrame.from_records(self.data_dict)
        )["hybcprob"]
        params_foster = node_without_parents.fit_parameters(
            pd.DataFrame.from_records(self.data_dict)
        )["hybcprob"]["[]"]

        self.assertIsNone(params_foster["regressor_obj"])
        self.assertIsNotNone(params_foster["mean"])

        # Since there can be empty dictionaries,
        # we have to count them and if a fraction is greater than the threshold
        # test failed.
        report = []
        for comb, data in params_parents.items():
            if all(pd.isna(x) for x in data.values()):
                report.append(1)
                continue
            else:
                report.append(0)

            if pd.isna(data["mean"]):
                self.assertIsNotNone(data["regressor_obj"])
                self.assertIsNotNone(data["std"])
            else:
                self.assertIsNone(data["regressor_obj"])
                self.assertIsNotNone(data["mean"])

        self.assertLess(sum(report) / len(report), 0.3)
        # print(params_parents, params_foster, sep="\n\n")

        # for k, v in params_parents.items():
        #     print(k, True if v["regressor_obj"] else False, v["regressor"])

        # print(sum(report) / len(report), node_without_parents.regressor)

    def test_choose(self):
        pvals = [1.05, 1.95, "cat4", "cat7"]
        params = self.node.fit_parameters(pd.DataFrame.from_records(self.data_dict))
        self.assertTrue(isinstance(self.node.choose(params, pvals), float))

    def test_predict(self):
        pvals = [1.05, 1.95, "cat4", "cat7"]
        params = self.node.fit_parameters(pd.DataFrame.from_records(self.data_dict))

        self.assertTrue(isinstance(self.node.predict(params, pvals), float))
        self.assertRaises(KeyError, self.node.predict, params, ["bad", "values"])


class TestMixtureGaussianNode(unittest.TestCase):
    def setUp(self):
        self.node = mixture_gaussian_node.MixtureGaussianNode(name="test")
        self.data_dict = {
            "node0": np.random.normal(1, 4, 30),
            "node1": np.random.normal(2, 0.1, 30),
            "test": np.random.normal(3, 0.3, 30),
            "node2": np.random.choice(["cat1", "cat2", "cat3"], 30),
            "node4": np.random.choice(["cat4", "cat5", "cat6"], 30),
            "node5": np.random.choice(["cat7", "cat8", "cat9"], 30),
            "node6": np.random.choice(["cat7", "cat8", "cat9"], 30),
        }

        # self.node.disc_parents = ["node4", "node5"]
        self.node.cont_parents = ["node0", "node1"]
        self.node.children = ["node6"]

    def test_fit_parameters(self):
        params = self.node.fit_parameters(pd.DataFrame.from_records(self.data_dict))
        self.assertAlmostEqual(sum(params["coef"]), 1, delta=1e-5)

    def test_choose(self):
        pvals = [1.05, 1.95]
        params = self.node.fit_parameters(pd.DataFrame.from_records(self.data_dict))
        self.assertTrue(isinstance(self.node.choose(params, pvals), float))

    def test_predict(self):
        pvals = [1.05, 1.95]
        params = self.node.fit_parameters(pd.DataFrame.from_records(self.data_dict))

        self.assertTrue(isinstance(self.node.predict(params, pvals), float))


class TestConditionalMixtureGaussianNode(unittest.TestCase):
    def setUp(self):
        self.node = conditional_mixture_gaussian_node.ConditionalMixtureGaussianNode(
            name="test"
        )
        self.data_dict = {
            "node0": np.random.normal(1, 4, 30),
            "node1": np.random.normal(2, 0.1, 30),
            "test": np.random.normal(3, 0.3, 30),
            "node2": np.random.choice(["cat1", "cat2", "cat3"], 30),
            "node4": np.random.choice(["cat4", "cat5", "cat6"], 30),
            "node5": np.random.choice(["cat7", "cat8", "cat9"], 30),
            "node6": np.random.choice(["cat7", "cat8", "cat9"], 30),
        }

        self.node.disc_parents = ["node4", "node5"]
        self.node.cont_parents = ["node0", "node1"]
        self.node.children = ["node6"]

    def test_fit_parameters(self):
        params = self.node.fit_parameters(pd.DataFrame.from_records(self.data_dict))[
            "hybcprob"
        ]
        report = []
        # sometimes combination's data can be empty, so we set the percent of
        # empty combinations
        for comb, data in params.items():
            if np.isclose(sum(data["coef"]), 1, atol=1e-5):
                report.append(0)
            else:
                report.append(1)
        self.assertLess(sum(report) / len(report), 0.3)

    def test_choose(self):
        pvals = [1.05, 1.95, "cat4", "cat7"]
        params = self.node.fit_parameters(pd.DataFrame.from_records(self.data_dict))
        self.assertTrue(isinstance(self.node.choose(params, pvals), float))

    def test_predict(self):
        pvals = [1.05, 1.95, "cat4", "cat7"]
        params = self.node.fit_parameters(pd.DataFrame.from_records(self.data_dict))

        self.assertTrue(isinstance(self.node.predict(params, pvals), float))


class TestLogitNode(unittest.TestCase):
    def setUp(self):
        self.node = logit_node.LogitNode(name="test")
        self.data_dict = {
            "node0": np.random.normal(1, 4, 30),
            "node1": np.random.normal(2, 0.1, 30),
            "node2": np.random.normal(3, 0.3, 30),
            "test": np.random.choice(["cat1", "cat2", "cat3"], 30),
            "node4": np.random.choice(["cat4", "cat5", "cat6"], 30),
            "node5": np.random.choice(["cat7", "cat8", "cat9"], 30),
            "node6": np.random.choice(["cat7", "cat8", "cat9"], 30),
        }

        # self.node.disc_parents = ["node4", "node5"]
        self.node.cont_parents = ["node0", "node1"]
        self.node.children = ["node6"]

    def test_fit_parameters(self):
        params = self.node.fit_parameters(pd.DataFrame.from_records(self.data_dict))
        self.assertIsNotNone(params["classifier_obj"])

    def test_choose(self):
        pvals = [1.05, 1.95]
        params = self.node.fit_parameters(pd.DataFrame.from_records(self.data_dict))
        self.assertTrue(self.node.choose(params, pvals) in params["classes"])

    def test_predict(self):
        pvals = [1.05, 1.95]
        params = self.node.fit_parameters(pd.DataFrame.from_records(self.data_dict))

        self.assertTrue([self.node.predict(params, pvals) in params["classes"]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
