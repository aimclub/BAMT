"""
Unit tests for DAG optimizers in BAMT 2.0.0 architecture.

Tests BigBraveBN, GOLEM, HybridDAGOptimizer, and other structure learning algorithms.
"""

import unittest

import networkx as nx
import pandas as pd
import numpy as np
from sklearn import preprocessing

from bamt.dag_optimizers.score.bigbravebn import BigBraveBN
from bamt.dag_optimizers.score.hill_climbing import HillClimbing
from bamt.dag_optimizers.constraint.pc_algorithm import PCAlgorithm
from bamt.dag_optimizers.hybrid.hybrid_dag_optimizer import HybridDAGOptimizer
from bamt.score_functions.k2_score import K2Score
from bamt.score_functions.mutual_information_score import MutualInformationScore


class TestBigBraveBN(unittest.TestCase):
    """Test BigBraveBN optimizer."""

    def setUp(self):
        """Set up test data."""
        # Create simple discrete test data
        np.random.seed(42)
        n_samples = 100

        # Create correlated variables
        x1 = np.random.randint(0, 3, n_samples)
        x2 = (x1 + np.random.randint(0, 2, n_samples)) % 3
        x3 = (x2 + np.random.randint(0, 2, n_samples)) % 3
        x4 = np.random.randint(0, 3, n_samples)  # Independent variable

        self.data = pd.DataFrame({
            'A': x1,
            'B': x2,
            'C': x3,
            'D': x4
        })

    def test_initialization(self):
        """Test BigBraveBN initialization."""
        optimizer = BigBraveBN()
        self.assertEqual(optimizer.n_nearest, 5)
        self.assertEqual(optimizer.threshold, 0.3)
        self.assertEqual(optimizer.proximity_metric, "MI")
        self.assertEqual(optimizer.possible_edges, [])

    def test_custom_parameters(self):
        """Test BigBraveBN with custom parameters."""
        optimizer = BigBraveBN(n_nearest=3, threshold=0.5, proximity_metric="pearson")
        self.assertEqual(optimizer.n_nearest, 3)
        self.assertEqual(optimizer.threshold, 0.5)
        self.assertEqual(optimizer.proximity_metric, "pearson")

    def test_set_possible_edges_by_brave_mi(self):
        """Test BRAVE edge identification with MI metric."""
        optimizer = BigBraveBN(n_nearest=3, threshold=0.1)
        edges = optimizer.set_possible_edges_by_brave(self.data, n_nearest=3, threshold=0.1)

        # Should return list of tuples
        self.assertIsInstance(edges, list)

        # Check that edges are tuples of strings (column names)
        if edges:
            self.assertIsInstance(edges[0], tuple)
            self.assertEqual(len(edges[0]), 2)

    def test_set_possible_edges_by_brave_pearson(self):
        """Test BRAVE edge identification with Pearson correlation."""
        optimizer = BigBraveBN(proximity_metric="pearson")
        edges = optimizer.set_possible_edges_by_brave(
            self.data,
            proximity_metric="pearson",
            threshold=0.1
        )

        self.assertIsInstance(edges, list)

    def test_optimize(self):
        """Test BigBraveBN optimize method."""
        optimizer = BigBraveBN(n_nearest=3, threshold=0.1)
        graph = optimizer.optimize(self.data)

        # Should return NetworkX DiGraph
        self.assertIsInstance(graph, nx.DiGraph)

        # Should have all nodes
        self.assertEqual(len(graph.nodes), len(self.data.columns))

        # Nodes should match column names
        self.assertEqual(set(graph.nodes), set(self.data.columns))

    def test_proximity_matrix_mi(self):
        """Test proximity matrix calculation with MI."""
        optimizer = BigBraveBN()
        proximity_matrix = optimizer._get_proximity_matrix(self.data, "MI")

        # Should return DataFrame
        self.assertIsInstance(proximity_matrix, pd.DataFrame)

        # Shape should be (n_cols, n_cols)
        expected_shape = (len(self.data.columns), len(self.data.columns))
        self.assertEqual(proximity_matrix.shape, expected_shape)

        # Diagonal should be positive (variable with itself)
        for col in self.data.columns:
            self.assertGreater(proximity_matrix.loc[col, col], 0)

    def test_proximity_matrix_pearson(self):
        """Test proximity matrix calculation with Pearson."""
        optimizer = BigBraveBN()
        proximity_matrix = optimizer._get_proximity_matrix(self.data, "pearson")

        self.assertIsInstance(proximity_matrix, pd.DataFrame)
        expected_shape = (len(self.data.columns), len(self.data.columns))
        self.assertEqual(proximity_matrix.shape, expected_shape)

    def test_invalid_proximity_metric(self):
        """Test that invalid proximity metric raises error."""
        optimizer = BigBraveBN()
        with self.assertRaises(ValueError):
            optimizer._get_proximity_matrix(self.data, "invalid_metric")


class TestHybridDAGOptimizer(unittest.TestCase):
    """Test HybridDAGOptimizer."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 100

        # Create correlated variables
        x1 = np.random.randint(0, 3, n_samples)
        x2 = (x1 + np.random.randint(0, 2, n_samples)) % 3
        x3 = (x2 + np.random.randint(0, 2, n_samples)) % 3

        self.data = pd.DataFrame({
            'A': x1,
            'B': x2,
            'C': x3
        })

    def test_initialization(self):
        """Test HybridDAGOptimizer initialization."""
        score_fn = K2Score()
        optimizer = HybridDAGOptimizer(score_function=score_fn)

        self.assertIsInstance(optimizer.constraint_optimizer, PCAlgorithm)
        self.assertIsInstance(optimizer.score_optimizer, HillClimbing)

    def test_custom_parameters(self):
        """Test HybridDAGOptimizer with custom parameters."""
        score_fn = K2Score()
        optimizer = HybridDAGOptimizer(
            score_function=score_fn,
            significance_level=0.01,
            max_iter=100,
            max_parents=2
        )

        self.assertEqual(optimizer.constraint_optimizer.significance_level, 0.01)
        self.assertEqual(optimizer.score_optimizer.max_iter, 100)
        self.assertEqual(optimizer.score_optimizer.max_parents, 2)


class TestIntegration(unittest.TestCase):
    """Integration tests for DAG optimizers."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 200

        # Create more complex correlated variables
        x1 = np.random.randint(0, 3, n_samples)
        x2 = (x1 + np.random.randint(0, 2, n_samples)) % 3
        x3 = (x2 + np.random.randint(0, 2, n_samples)) % 3
        x4 = np.random.randint(0, 3, n_samples)

        self.data = pd.DataFrame({
            'A': x1,
            'B': x2,
            'C': x3,
            'D': x4
        })

    def test_bigbrave_with_hill_climbing(self):
        """Test using BigBrave to restrict Hill Climbing search space."""
        # Get possible edges from BigBrave
        brave_optimizer = BigBraveBN(n_nearest=3, threshold=0.1)
        brave_graph = brave_optimizer.optimize(self.data)

        # Use BigBrave edges as whitelist for Hill Climbing
        whitelist = list(brave_graph.edges())

        # Run Hill Climbing with whitelist
        score_fn = K2Score()
        hc_optimizer = HillClimbing(
            score_function=score_fn,
            white_list=whitelist,
            max_iter=50
        )

        final_graph = hc_optimizer.optimize(self.data)

        # Should return valid DAG
        self.assertIsInstance(final_graph, nx.DiGraph)
        self.assertTrue(nx.is_directed_acyclic_graph(final_graph))

        # All edges should be in whitelist
        for edge in final_graph.edges():
            self.assertIn(edge, whitelist)


class TestContinuousData(unittest.TestCase):
    """Test optimizers with continuous data."""

    def setUp(self):
        """Set up continuous test data."""
        np.random.seed(42)
        n_samples = 100

        # Create correlated continuous variables
        x1 = np.random.randn(n_samples)
        x2 = 0.5 * x1 + np.random.randn(n_samples) * 0.5
        x3 = 0.3 * x2 + np.random.randn(n_samples) * 0.7

        self.data = pd.DataFrame({
            'A': x1,
            'B': x2,
            'C': x3
        })

    def test_bigbrave_continuous(self):
        """Test BigBrave with continuous data using Pearson."""
        optimizer = BigBraveBN(proximity_metric="pearson", threshold=0.1)
        graph = optimizer.optimize(self.data)

        self.assertIsInstance(graph, nx.DiGraph)
        self.assertEqual(len(graph.nodes), len(self.data.columns))


class TestMixedData(unittest.TestCase):
    """Test optimizers with mixed discrete/continuous data."""

    def setUp(self):
        """Set up mixed test data."""
        np.random.seed(42)
        n_samples = 100

        # Discrete variables
        x1 = np.random.randint(0, 3, n_samples)
        x2 = (x1 + np.random.randint(0, 2, n_samples)) % 3

        # Continuous variables
        x3 = np.random.randn(n_samples)
        x4 = 0.5 * x3 + np.random.randn(n_samples) * 0.5

        self.data = pd.DataFrame({
            'Discrete1': x1,
            'Discrete2': x2,
            'Continuous1': x3,
            'Continuous2': x4
        })

    def test_bigbrave_mixed(self):
        """Test BigBrave with mixed data types."""
        optimizer = BigBraveBN(proximity_metric="MI", threshold=0.1)
        graph = optimizer.optimize(self.data)

        self.assertIsInstance(graph, nx.DiGraph)
        self.assertEqual(len(graph.nodes), len(self.data.columns))


if __name__ == '__main__':
    unittest.main()
