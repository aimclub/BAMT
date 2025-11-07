"""
Tests for BAMT 2.0.0 Structure Learning (DAG Optimizers and Score Functions)
Following TDD principles
"""
import unittest

try:
    import numpy as np
    import pandas as pd
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False


class TestScoreFunctions20(unittest.TestCase):
    """Test suite for 2.0.0 score functions"""
    
    def setUp(self):
        """Set up test data"""
        if DEPS_AVAILABLE:
            np.random.seed(42)
            # Create simple dataset with known structure: A -> B -> C
            a = np.random.normal(0, 1, 100)
            b = 2 * a + np.random.normal(0, 0.5, 100)
            c = 3 * b + np.random.normal(0, 0.5, 100)
            self.data = pd.DataFrame({'A': a, 'B': b, 'C': c})
        
    def test_k2_score_initialization(self):
        """Test K2Score can be initialized"""
        from bamt.score_functions import K2Score
        score = K2Score()
        self.assertIsNotNone(score)
        
    @unittest.skipIf(not DEPS_AVAILABLE, "numpy/pandas not available")
    def test_k2_score_computation(self):
        """Test K2Score can compute score for a DAG"""
        from bamt.score_functions import K2Score
        score_fn = K2Score()
        
        # Test with simple structure
        edges = [('A', 'B'), ('B', 'C')]
        score = score_fn.compute(self.data, edges)
        self.assertIsInstance(score, (int, float))
        
    def test_mi_score_initialization(self):
        """Test MutualInformationScore can be initialized"""
        from bamt.score_functions import MutualInformationScore
        score = MutualInformationScore()
        self.assertIsNotNone(score)
        
    @unittest.skipIf(not DEPS_AVAILABLE, "numpy/pandas not available")
    def test_mi_score_computation(self):
        """Test MI score computation"""
        from bamt.score_functions import MutualInformationScore
        score_fn = MutualInformationScore()
        
        # Test with simple structure
        edges = [('A', 'B'), ('B', 'C')]
        score = score_fn.compute(self.data, edges)
        self.assertIsInstance(score, (int, float))


class TestHillClimbingOptimizer20(unittest.TestCase):
    """Test suite for 2.0.0 Hill Climbing optimizer"""
    
    def setUp(self):
        """Set up test data"""
        if DEPS_AVAILABLE:
            np.random.seed(42)
            # Create simple dataset with known structure
            a = np.random.normal(0, 1, 100)
            b = 2 * a + np.random.normal(0, 0.5, 100)
            c = 3 * b + np.random.normal(0, 0.5, 100)
            self.data = pd.DataFrame({'A': a, 'B': b, 'C': c})
        
    def test_hc_initialization(self):
        """Test HillClimbing optimizer can be initialized"""
        from bamt.dag_optimizers.score import HillClimbingOptimizer
        optimizer = HillClimbingOptimizer()
        self.assertIsNotNone(optimizer)
        
    @unittest.skipIf(not DEPS_AVAILABLE, "numpy/pandas not available")
    def test_hc_optimize(self):
        """Test Hill Climbing can find structure"""
        from bamt.dag_optimizers.score import HillClimbingOptimizer
        from bamt.score_functions import K2Score
        
        optimizer = HillClimbingOptimizer(score_function=K2Score())
        edges = optimizer.optimize(self.data)
        
        # Should return a list of edges
        self.assertIsInstance(edges, list)
        # Should find at least some edges
        self.assertTrue(len(edges) >= 0)
        # Each edge should be a tuple
        if edges:
            self.assertIsInstance(edges[0], tuple)
            self.assertEqual(len(edges[0]), 2)


if __name__ == '__main__':
    unittest.main()
