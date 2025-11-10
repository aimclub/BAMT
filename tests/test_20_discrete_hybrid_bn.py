"""
Tests for BAMT 2.0.0 Discrete and Hybrid Bayesian Networks
Following TDD principles
"""
import unittest

try:
    import numpy as np
    import pandas as pd
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

from bamt.models.probabilistic_structural_models import DiscreteBayesianNetwork, HybridBayesianNetwork


class TestDiscreteBayesianNetwork20(unittest.TestCase):
    """Test suite for 2.0.0 DiscreteBayesianNetwork"""
    
    def setUp(self):
        """Set up test data"""
        if DEPS_AVAILABLE:
            np.random.seed(42)
            # Create discrete dataset
            self.data = pd.DataFrame({
                'A': np.random.choice(['low', 'medium', 'high'], 100),
                'B': np.random.choice(['yes', 'no'], 100),
                'C': np.random.choice([0, 1, 2], 100)
            })
        
    def test_initialization(self):
        """Test that DiscreteBayesianNetwork can be initialized"""
        bn = DiscreteBayesianNetwork()
        self.assertIsNotNone(bn)
        
    @unittest.skipIf(not DEPS_AVAILABLE, "numpy/pandas not available")
    def test_set_structure(self):
        """Test setting structure from edge list"""
        bn = DiscreteBayesianNetwork()
        edges = [('A', 'B'), ('A', 'C')]
        bn.set_structure(edges)
        self.assertEqual(len(bn.edges), 2)
        
    @unittest.skipIf(not DEPS_AVAILABLE, "numpy/pandas not available")
    def test_fit(self):
        """Test fitting parameters"""
        bn = DiscreteBayesianNetwork()
        edges = [('A', 'B'), ('A', 'C')]
        bn.set_structure(edges)
        bn.fit(self.data)
        self.assertTrue(len(bn.node_models) > 0)
        
    @unittest.skipIf(not DEPS_AVAILABLE, "numpy/pandas not available")
    def test_sample(self):
        """Test sampling from fitted network"""
        bn = DiscreteBayesianNetwork()
        edges = [('A', 'B')]
        bn.set_structure(edges)
        bn.fit(self.data)
        samples = bn.sample(10)
        self.assertEqual(len(samples), 10)


class TestHybridBayesianNetwork20(unittest.TestCase):
    """Test suite for 2.0.0 HybridBayesianNetwork"""
    
    def setUp(self):
        """Set up test data"""
        if DEPS_AVAILABLE:
            np.random.seed(42)
            # Create hybrid dataset (continuous + discrete)
            self.data = pd.DataFrame({
                'cont1': np.random.normal(0, 1, 100),
                'cont2': np.random.normal(0, 1, 100),
                'disc1': np.random.choice(['A', 'B', 'C'], 100),
                'disc2': np.random.choice([0, 1], 100)
            })
        
    def test_initialization(self):
        """Test that HybridBayesianNetwork can be initialized"""
        bn = HybridBayesianNetwork()
        self.assertIsNotNone(bn)
        
    @unittest.skipIf(not DEPS_AVAILABLE, "numpy/pandas not available")
    def test_set_structure(self):
        """Test setting structure with mixed types"""
        bn = HybridBayesianNetwork()
        edges = [('cont1', 'disc1'), ('disc1', 'cont2')]
        bn.set_structure(edges)
        self.assertEqual(len(bn.edges), 2)
        
    @unittest.skipIf(not DEPS_AVAILABLE, "numpy/pandas not available")
    def test_fit_mixed_data(self):
        """Test fitting with mixed continuous/discrete data"""
        bn = HybridBayesianNetwork()
        edges = [('cont1', 'disc1'), ('disc1', 'cont2')]
        bn.set_structure(edges)
        bn.fit(self.data)
        self.assertTrue(len(bn.node_models) > 0)


if __name__ == '__main__':
    unittest.main()
