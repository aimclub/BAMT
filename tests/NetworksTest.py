import json
import time
import itertools

# import abc

import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing as pp

from bamt.preprocessors import Preprocessor
import bamt.networks as Networks
import bamt.nodes as Nodes


class NetworkTest(object):
    def __init__(
        self,
        directory: str,
        verbose: bool = False,
        case_id: int = 0,
        sample_n: int = 500,
        sample_tol: float = 0.6,
    ):
        """
        sample_n: number of rows in sample
        sample_tol: precent of acceptable number of nans.
        If number of nan more than sample_int * sample_tol, then sample test failed.
        """
        self.bn = Networks.BaseNetwork()
        self.sf = ""
        self.type = "abstract"
        self.case_id = case_id
        self.sample_n = sample_n
        self.sample_tol = sample_tol
        self.verboseprint = print if verbose else lambda *a: None
        self.directory = directory

    @staticmethod
    def _tabularize_output(message1: str, message2: str):
        # message1 - usually class of output (error, success)
        # message2 - usually description of message 1
        return f"{message1: <52} | {message2: >3}"

    def test_preprocess(self):
        failed = False

        if self.case_id == 0:
            self.discrete_cols = [
                "Tectonic regime",
                "Period",
                "Lithology",
                "Structural setting",
            ]
            self.cont_cols = ["Gross", "Netpay", "Porosity", "Permeability", "Depth"]
            self.hybrid_cols = [
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
            # Base of standards
            self.base = "hack_" + self.type
        else:
            self.discrete_cols = []
            self.cont_cols = []
            self.hybrid_cols = []

        if self.type == "discrete":
            data = pd.read_csv(self.directory)[self.discrete_cols]
        elif self.type == "continuous":
            data = pd.read_csv(self.directory)[self.cont_cols]
        else:
            data = pd.read_csv(self.directory)[self.hybrid_cols]

        encoder = pp.LabelEncoder()
        discretizer = pp.KBinsDiscretizer(
            n_bins=5, encode="ordinal", strategy="uniform"
        )

        p = Preprocessor([("encoder", encoder), ("discretizer", discretizer)])

        discretized_data, est = p.apply(data)
        info = p.info

        try:
            assert info == json.load(open(f"{self.base}/hack_info.json"))
        except AssertionError:
            failed = True
            self.verboseprint(self._tabularize_output("ERROR", "Bad descriptor"))

        try:
            assert_frame_equal(
                discretized_data,
                pd.read_csv(f"{self.base}/hack_data.csv", index_col=0),
                check_dtype=False,
            )
        except Exception as ex:
            failed = True
            self.verboseprint(self._tabularize_output("ERROR", str(ex)))

        if not failed:
            status = "OK"
        else:
            status = "Failed"
        print(self._tabularize_output("Preprocess", status))

        self.info = info
        self.data = discretized_data

    def test_structure_learning(self):
        pass

    def test_parameters_learning(self):
        pass

    def test_sampling(self):
        failed = False

        sample = self.bn.sample(n=self.sample_n, progress_bar=False)

        if sample.empty:
            self.verboseprint("Sampling", "Dataframe is empty")
            return

        if sample.isna().sum().sum() > (self.sample_n * self.sample_tol):
            failed = True

        if not failed:
            status = "OK"
        else:
            status = "Failed"

        print(self._tabularize_output(f"Sampling ({self.sf})", status))

    def test_predict(self):
        failed = False

        if self.type == "discrete":
            cols = self.discrete_cols
        elif self.type == "continuous":
            cols = self.cont_cols
        elif self.type == "hybrid":
            cols = self.hybrid_cols
        else:
            raise Exception("Inner error")

        preds = self.bn.predict(
            test=pd.read_csv(self.directory)[cols[:2]].dropna(),
            progress_bar=False,
            parall_count=2,
        )

        # with open(f"{self.base}/hack_predict.json", "r") as f:
        #     p = json.load(f)

        if self.type == "continuous":
            # cols: ['Porosity', 'Permeability', 'Depth']
            for node in preds.keys():
                right_val = json.load(open(f"{self.base}/hack_predict.json"))[self.sf][
                    node
                ]
                test_val = np.mean([mx for mx in preds[node] if not np.isnan(mx)])
                assert np.all(
                    np.isclose(test_val, right_val, rtol=0.4)
                ), f"Predict failed: {node, right_val, test_val}"
        elif self.type == "discrete":
            # cols: ['Lithology', 'Structural setting']
            for node in preds.keys():
                test_vals = pd.Series(preds[node]).value_counts().to_dict()
                for category, right_val in json.load(
                    open(f"{self.base}/hack_predict.json")
                )[self.sf][node].items():
                    try:
                        assert np.all(
                            np.isclose(test_vals[category], right_val, atol=5)
                        ), f"Predict failed: {node, test_vals[category], right_val}"
                    except KeyError as ex:
                        print("Unknown preds category: ", ex.args[0])
                        continue
        elif self.type == "hybrid":
            cont_nodes = [
                node
                for node in self.bn.nodes_names
                if self.info["types"][node] == "cont"
            ]
            for node in preds.keys():
                if node in cont_nodes:
                    right_val = json.load(open(f"{self.base}/hack_predict.json"))[
                        self.sf
                    ][node]
                    test_val = np.mean([mx for mx in preds[node] if not np.isnan(mx)])
                    # p[self.sf][node] = test_val
                    s = [right_val, test_val]
                    assert np.all(
                        np.isclose(min(s), max(s), atol=5, rtol=0.6)
                    ), f"Predict failed: {node, test_val, right_val}"
                else:
                    test_vals = pd.Series(preds[node]).value_counts().to_dict()
                    # p[self.sf][node] = test_vals
                    for category, right_val in json.load(
                        open(f"{self.base}/hack_predict.json")
                    )[self.sf][node].items():
                        try:
                            assert np.all(
                                np.isclose(
                                    min(test_vals[category], right_val),
                                    max(right_val, test_vals[category]),
                                    atol=100,
                                    rtol=0.5,
                                )
                            ), f"Predict failed: {node, test_vals[category], right_val}"
                        except KeyError as ex:
                            print("Unknown preds category: ", ex.args[0])
                            continue

        if not failed:
            status = "OK"
        else:
            status = "Failed"

        print(self._tabularize_output(f"Predict ({self.sf})", status))

        # with open(f"{self.base}/hack_predict.json", "w") as f:
        #     json.dump(p, f)

    def apply(self):
        pass

    @staticmethod
    def use_rules(*args, **kwargs):
        for rule in args:
            rule(**kwargs)


class TestDiscreteBN(NetworkTest):
    def __init__(self, **kwargs):
        super(TestDiscreteBN, self).__init__(**kwargs)
        self.type = "discrete"

    def test_structure_learning(self):
        failed = False

        bn = Networks.DiscreteBN()
        bn.add_nodes(descriptor=self.info)

        try:
            assert bn.nodes_names == [
                "Tectonic regime",
                "Period",
                "Lithology",
                "Structural setting",
            ]
        except AssertionError:
            failed = True
            self.verboseprint(
                self._tabularize_output(
                    "ERROR", "first stage failed (wrong init nodes)."
                )
            )

        bn.add_edges(self.data, (self.sf,), progress_bar=False)

        try:
            assert bn.edges == json.load(open(f"{self.base}/hack_edges.json"))[self.sf]
        except AssertionError:
            failed = True
            self.verboseprint(f"Stage 2 failed with {self.sf}.")

        if not failed:
            self.bn = bn
            status = "OK"
        else:
            self.bn = None
            status = "Failed"

        print(self._tabularize_output(f"Structure ({self.sf})", status))

    def test_parameters_learning(self):
        failed = False

        self.bn.fit_parameters(pd.read_csv(self.directory)[self.discrete_cols])

        try:
            assert (
                self.bn.distributions
                == json.load(open(f"{self.base}/hack_params.json"))[self.sf]
            )
        except AssertionError:
            failed = True
            self.verboseprint(
                self._tabularize_output(f"Parameters ({self.sf})", "bad distributions")
            )

        if not failed:
            status = "OK"
        else:
            status = "Failed"

        print(self._tabularize_output(f"Parameters ({self.sf})", status))

    def apply(self):
        print(f"Executing {self.type} BN tests.")
        self.test_preprocess()
        t0 = time.time()
        for sf in ["MI", "K2", "BIC"]:
            self.sf = sf
            t1 = time.time()
            self.test_structure_learning()
            if not self.bn:
                print(self._tabularize_output(f"Error on {sf}", "No structure"))
                print("-" * 8)
                continue
            self.test_parameters_learning()
            self.test_sampling()
            self.test_predict()
            t2 = time.time()
            print("-" * 8, f"Elapsed time: {t2 - t1}", "-" * 8, "\n")
        t3 = time.time()
        print(f"\nElapsed time: {t3 - t0}")


class TestContinuousBN(NetworkTest):
    def __init__(self, **kwargs):
        super(TestContinuousBN, self).__init__(**kwargs)
        self.type = "continuous"

    def test_setters(self):
        failed = False

        bn = Networks.ContinuousBN()
        ns = []
        for d in [Nodes.GaussianNode(name="Node" + str(id)) for id in range(0, 4)]:
            ns.append(d)

        bn.set_structure(nodes=ns)
        bn.set_classifiers(
            classifiers={
                "Node0": DecisionTreeClassifier(),
                "Node1": RandomForestClassifier(),
                "Node2": KNeighborsClassifier(n_neighbors=2),
            }
        )

        assert [str(bn[node].classifier) for node in ["Node0", "Node1", "Node2"]] == [
            "DecisionTreeClassifier()",
            "RandomForestClassifier()",
            "KNeighborsClassifier(n_neighbors=2)",
        ], "Setter | Classifiers are wrong."

        if not failed:
            status = "OK"
        else:
            status = "Failed"

        print(self._tabularize_output("Setters", status))

    def test_structure_learning(self, use_mixture: bool = False):
        self.use_mixture = use_mixture
        failed = False

        bn = Networks.ContinuousBN(use_mixture=use_mixture)
        bn.add_nodes(descriptor=self.info)

        try:
            assert (
                bn.nodes_names
                == json.load(open(f"{self.base}/hack_nodes.json"))[
                    f"use_mixture={use_mixture}"
                ][self.sf]
            )
        except AssertionError:
            failed = True
            self.verboseprint(
                self._tabularize_output(
                    "ERROR", "first stage failed (wrong init nodes)."
                )
            )

        bn.add_edges(self.data, (self.sf,), progress_bar=False)

        try:
            assert (
                bn.edges
                == json.load(open(f"{self.base}/hack_edges.json"))[
                    f"use_mixture={use_mixture}"
                ][self.sf]
            )
        except AssertionError:
            failed = True
            self.verboseprint(f"Stage 2 failed with {self.sf}.")

        if not failed:
            self.bn = bn
            status = "OK"
        else:
            status = "Failed"

        print(
            self._tabularize_output(
                f"Structure ({self.sf}, use_mixture={self.use_mixture})", status
            )
        )

    def test_parameters_learning(self):
        failed = False

        self.bn.fit_parameters(pd.read_csv(self.directory)[self.cont_cols])
        try:
            if self.use_mixture:
                empty_data = {"mean": [], "covars": [], "coef": []}
                for k, v in self.bn.distributions.items():
                    assert all(
                        [v[obj] != empty for obj, empty in empty_data.items()]
                    ), f"Empty data in {k}."
                    assert (
                        0.9 <= sum(v["coef"]) <= 1.1
                    ), f"{sum(v['coef'])} || {k}'s: coefs are wrong."
            else:
                assert (
                    self.bn.distributions
                    == json.load(open(f"{self.base}/hack_params.json"))[
                        "use_mixture=False"
                    ][self.sf]
                ), "Bad distributions."
        except AssertionError as ex:
            failed = True
            self.verboseprint(
                self._tabularize_output(
                    f"Parameters ({self.sf}, use_mixture={self.use_mixture})",
                    ex.args[0],
                )
            )

        if not failed:
            status = "OK"
        else:
            status = "Failed"

        print(
            self._tabularize_output(
                f"Parameters ({self.sf}, use_mixture={self.use_mixture})", status
            )
        )

    def apply(self):
        print(f"Executing {self.type} BN tests.")
        self.test_preprocess()
        # Auskommentieren mich
        # self._test_setters()
        t0 = time.time()
        for use_mixture in [True, False]:
            for sf in ["MI", "K2", "BIC"]:
                self.sf = sf
                t1 = time.time()
                self.test_structure_learning(use_mixture=use_mixture)
                if not self.bn:
                    print(self._tabularize_output(f"Error on {sf}", "No structure"))
                    print("-" * 8)
                    continue
                self.test_parameters_learning()
                self.test_sampling()
                self.test_predict()
                t2 = time.time()
                print("-" * 8, f"Elapsed time: {t2 - t1}", "-" * 8)
        t3 = time.time()
        print(f"\nElapsed time: {t3 - t0}")


class TestHybridBN(NetworkTest):
    def __init__(self, **kwargs):
        super(TestHybridBN, self).__init__(**kwargs)
        self.type = "hybrid"

    def test_setters(self):
        failed = False

        bn = Networks.HybridBN(has_logit=True)
        ns = []
        for d, g in zip(
            [Nodes.GaussianNode(name="Node" + str(id)) for id in range(0, 3)],
            [Nodes.DiscreteNode(name="Node" + str(id)) for id in range(3, 6)],
        ):
            ns.append(d)
            ns.append(g)
        edges = [
            ("Node0", "Node3"),
            ("Node3", "Node1"),
            ("Node1", "Node4"),
            ("Node4", "Node2"),
            ("Node2", "Node5"),
        ]
        test_info = {
            "types": {
                "Node0": "cont",
                "Node1": "cont",
                "Node2": "cont",
                "Node3": "disc",
                "Node4": "disc",
                "Node5": "disc",
            },
            "signs": {"Node0": "pos", "Node1": "pos", "Node2": "pos"},
        }

        # Structure setter
        bn.set_structure(info=test_info, nodes=ns, edges=edges)

        assert [
            "Gaussian (LinearRegression)",
            "Logit (LogisticRegression)",
            "ConditionalGaussian (LinearRegression)",
            "Logit (LogisticRegression)",
            "ConditionalGaussian (LinearRegression)",
            "Logit (LogisticRegression)",
        ] == [node.type for node in bn.nodes], "Setter | Nodes are not the same."
        assert edges == bn.edges, "Setter | Edges are not the same."

        # Classifiers setters

        bn.set_classifiers(
            classifiers={
                "Node3": DecisionTreeClassifier(),
                "Node4": RandomForestClassifier(),
                "Node5": KNeighborsClassifier(n_neighbors=2),
            }
        )

        assert [str(bn[node].classifier) for node in ["Node3", "Node4", "Node5"]] == [
            "DecisionTreeClassifier()",
            "RandomForestClassifier()",
            "KNeighborsClassifier(n_neighbors=2)",
        ], "Setter | Classifiers are wrong."

        # Parameters setters

        with open("test_params.json") as f:
            p = json.load(f)

        bn.set_parameters(p)
        assert bn.distributions == p, "Setter | Parameters are not the same."

        if not failed:
            status = "OK"
        else:
            status = "Failed"

        print(self._tabularize_output("Setters", status))

    def test_structure_learning(
        self, use_mixture: bool = False, has_logit: bool = False
    ):
        self.use_mixture = use_mixture
        self.has_logit = has_logit
        failed = False

        bn = Networks.HybridBN(use_mixture=use_mixture, has_logit=has_logit)
        bn.add_nodes(descriptor=self.info)

        try:
            assert (
                bn.nodes_names
                == json.load(open(f"{self.base}/hack_nodes.json"))[
                    f"use_mixture={use_mixture}"
                ][f"has_logit={has_logit}"][self.sf]
            )
        except AssertionError:
            failed = True
            self.verboseprint(
                self._tabularize_output(
                    "ERROR", "first stage failed (wrong init nodes)."
                )
            )

        bn.add_edges(self.data, (self.sf,), progress_bar=False)

        try:
            assert (
                bn.edges
                == json.load(open(f"{self.base}/hack_edges.json"))[
                    f"use_mixture={use_mixture}"
                ][f"has_logit={has_logit}"][self.sf]
            )
        except AssertionError:
            failed = True
            self.verboseprint(f"Stage 2 failed with {self.sf}.")

        if not failed:
            self.bn = bn
            status = "OK"
        else:
            status = "Failed"

        print(
            self._tabularize_output(
                f"Structure ({self.sf}, use_mixture={self.use_mixture}, has_logit={self.has_logit})",
                status,
            )
        )

    @staticmethod
    def non_empty_gaussian_nodes(name, node_params):
        empty_data = {"mean": [], "covars": [], "coef": []}
        assert all(
            [node_params[obj] != empty for obj, empty in empty_data.items()]
        ), f"Empty data in {name}."

    @staticmethod
    def non_empty_logit_nodes(name, node_params):
        empty_data = {"classes": [], "classifier_obj": None}
        assert all(
            [node_params[obj] != empty for obj, empty in empty_data.items()]
        ), f"Empty data in {name}."

    @staticmethod
    def sum_equals_to_1(name, node_params):
        assert 0.9 <= sum(node_params["coef"]) <= 1.1, f"{name}'s: coefs are wrong."

    def _validate_node(self, name, type, node_params, true_vals):
        try:
            if type == "MixtureGaussian":
                self.use_rules(
                    self.non_empty_gaussian_nodes,
                    self.sum_equals_to_1,
                    name=name,
                    node_params=node_params,
                )
            elif type == "ConditionalMixtureGaussian":
                for comb, data in node_params["hybcprob"].items():
                    self.use_rules(
                        self.non_empty_gaussian_nodes,
                        self.sum_equals_to_1,
                        name=name,
                        node_params=data,
                    )
            elif type.startswith("Logit"):
                self.use_rules(
                    self.non_empty_logit_nodes, name=name, node_params=node_params
                )
            elif type.startswith("ConditionalLogit"):
                for comb, data in node_params["hybcprob"].items():
                    self.use_rules(
                        self.non_empty_logit_nodes, name=name, node_params=data
                    )
            else:
                assert node_params == true_vals, f"Parameters error on  {name}, {type}"
        except AssertionError as ex:
            self.verboseprint(
                self._tabularize_output(
                    f"Parameters ({self.sf}, use_mixture={self.use_mixture}, has_logit={self.has_logit})",
                    ex.args[0],
                )
            )

    def test_parameters_learning(self):
        failed = False

        self.bn.fit_parameters(pd.read_csv(self.directory)[self.hybrid_cols])
        try:
            true_params = json.load(open(f"{self.base}/hack_params.json"))[
                f"use_mixture={self.use_mixture}"
            ][f"has_logit={self.has_logit}"][self.sf]

            node_type_dict = {node.name: node.type for node in self.bn.nodes}
            for name, type in node_type_dict.items():
                node_params = self.bn.distributions[name]
                self._validate_node(name, type, node_params, true_params[name])
        except AssertionError as ex:
            failed = True
            self.verboseprint(
                self._tabularize_output(
                    f"Parameters ({self.sf}, use_mixture={self.use_mixture}, has_logit={self.has_logit})",
                    ex.args[0],
                )
            )

        if not failed:
            status = "OK"
        else:
            status = "Failed"

        print(
            self._tabularize_output(
                f"Parameters ({self.sf}, use_mixture={self.use_mixture}, has_logit={self.has_logit})",
                status,
            )
        )

    def apply(self):
        print(f"Executing {self.type} BN tests.")
        self.test_preprocess()
        self.test_setters()
        t0 = time.time()

        for use_mixture, has_logit in itertools.product([True, False], repeat=2):
            for sf in ["MI", "K2", "BIC"]:
                self.sf = sf
                t1 = time.time()
                self.test_structure_learning(
                    use_mixture=use_mixture, has_logit=has_logit
                )
                self.test_parameters_learning()
                if not self.bn:
                    print(self._tabularize_output(f"Error on {sf}", "No structure"))
                    print("-" * 8)
                    continue
                self.test_sampling()
                self.test_predict()
                t2 = time.time()
                print("-" * 8, f"Elapsed time: {t2 - t1}", "-" * 8)

        t3 = time.time()
        print(f"\nElapsed time: {t3 - t0}")
