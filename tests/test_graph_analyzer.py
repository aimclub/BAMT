import logging
import unittest

from bamt.builders.builders_base import VerticesDefiner
from bamt.networks.discrete_bn import DiscreteBN
from bamt.nodes.discrete_node import DiscreteNode
from bamt.utils import GraphUtils

logging.getLogger("builder").setLevel(logging.CRITICAL)


class TestGraphAnalyzer(unittest.TestCase):
    def setUp(self):
        self.bn = DiscreteBN()
        definer = VerticesDefiner(
            descriptor={
                "types": {
                    "Node0": "disc",
                    "Node1": "disc",
                    "Node2": "disc",
                    "Node3": "disc_num",
                    "Node4": "disc",
                    "Node5": "disc",
                    "Node6": "disc",
                    "Node7": "disc",
                    "Node8": "disc",
                    "Node9": "disc",
                },
                "signs": {},
            },
            regressor=None,
        )

        definer.skeleton["V"] = [DiscreteNode(name=f"Node{i}") for i in range(10)]
        definer.skeleton["E"] = [
            ("Node0", "Node1"),
            ("Node0", "Node2"),
            ("Node2", "Node3"),
            ("Node4", "Node7"),
            ("Node1", "Node5"),
            ("Node5", "Node6"),
            ("Node7", "Node0"),
            ("Node8", "Node1"),
            ("Node9", "Node2"),
        ]
        definer.get_family()

        self.bn.nodes = definer.skeleton["V"]
        self.bn.edges = definer.skeleton["E"]

        self.analyzer = GraphUtils.GraphAnalyzer(self.bn)

    def test_markov_blanket(self):
        result = self.analyzer.markov_blanket("Node0")
        result["nodes"] = sorted(result["nodes"])
        self.assertEqual(
            {
                "edges": [
                    ("Node0", "Node1"),
                    ("Node0", "Node2"),
                    ("Node7", "Node0"),
                    ("Node8", "Node1"),
                    ("Node9", "Node2"),
                ],
                "nodes": sorted(["Node0", "Node1", "Node2", "Node7", "Node8", "Node9"]),
            },
            result,
        )

    def test_find_family(self):
        without_parents = self.analyzer.find_family("Node0", 0, 2, None)
        without_parents["nodes"] = sorted(without_parents["nodes"])
        self.assertEqual(
            {
                "nodes": sorted(["Node3", "Node2", "Node1", "Node0", "Node5"]),
                "edges": [
                    ("Node0", "Node1"),
                    ("Node0", "Node2"),
                    ("Node2", "Node3"),
                    ("Node1", "Node5"),
                ],
            },
            without_parents,
        )

        without_children = self.analyzer.find_family("Node0", 2, 0, None)
        without_children["nodes"] = sorted(without_children["nodes"])
        self.assertEqual(
            {
                "nodes": sorted(["Node4", "Node7", "Node0"]),
                "edges": [("Node4", "Node7"), ("Node7", "Node0")],
            },
            without_children,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
