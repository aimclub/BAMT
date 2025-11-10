"""
Tests for BAMT 2.0.0 ContinuousBayesianNetwork implementation
Following TDD principles for migrating from v1.x to v2.0 architecture
"""
import unittest

try:
    import numpy as np
    import pandas as pd
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

from bamt.models.probabilistic_structural_models import ContinuousBayesianNetwork


class TestContinuousBayesianNetwork20(unittest.TestCase):
    """Test suite for 2.0.0 ContinuousBayesianNetwork"""
    
    def setUp(self):
        """Set up test data"""
        if DEPS_AVAILABLE:
            # Create simple continuous dataset
            np.random.seed(42)
            self.data = pd.DataFrame({
                'A': np.random.normal(0, 1, 100),
                'B': np.random.normal(0, 1, 100),
                'C': np.random.normal(0, 1, 100)
            })
        else:
            self.data = None
        
    def test_initialization(self):
        """Test that ContinuousBayesianNetwork can be initialized"""
        bn = ContinuousBayesianNetwork()
        self.assertIsNotNone(bn)
        
    @unittest.skipIf(not DEPS_AVAILABLE, "numpy/pandas not available")
    def test_set_structure_from_edges(self):
        """Test setting structure from edge list"""
        bn = ContinuousBayesianNetwork()
        edges = [('A', 'B'), ('A', 'C')]
        bn.set_structure(edges)
        self.assertEqual(len(bn.edges), 2)
        
    @unittest.skipIf(not DEPS_AVAILABLE, "numpy/pandas not available")
    def test_fit_parameters(self):
        """Test fitting parameters given structure"""
        bn = ContinuousBayesianNetwork()
        edges = [('A', 'B'), ('A', 'C')]
        bn.set_structure(edges)
        bn.fit(self.data)
        # Check that nodes have been fitted
        self.assertTrue(hasattr(bn, 'nodes'))
        self.assertTrue(len(bn.nodes) > 0)
        
    @unittest.skipIf(not DEPS_AVAILABLE, "numpy/pandas not available")
    def test_sample(self):
        """Test sampling from fitted network"""
        bn = ContinuousBayesianNetwork()
        edges = [('A', 'B'), ('A', 'C')]
        bn.set_structure(edges)
        bn.fit(self.data)
        samples = bn.sample(10)
        self.assertEqual(len(samples), 10)
        self.assertEqual(len(samples.columns), 3)
        
    @unittest.skipIf(not DEPS_AVAILABLE, "numpy/pandas not available")
    def test_predict(self):
        """Test prediction with evidence"""
        bn = ContinuousBayesianNetwork()
        edges = [('A', 'B'), ('A', 'C')]
        bn.set_structure(edges)
        bn.fit(self.data)
        
        # Predict B and C given A
        evidence = pd.DataFrame({'A': [0.5]})
        predictions = bn.predict(evidence, target=['B', 'C'])
        self.assertIn('B', predictions.columns)
        self.assertIn('C', predictions.columns)


if __name__ == '__main__':
    unittest.main()
