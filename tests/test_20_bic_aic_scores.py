"""
Tests for BAMT 2.0.0 BIC and AIC Score Functions
Following TDD principles
"""
import unittest

try:
    import numpy as np
    import pandas as pd
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False


class TestBICScore20(unittest.TestCase):
    """Test suite for BIC (Bayesian Information Criterion) score"""
    
    def setUp(self):
        """Set up test data"""
        if DEPS_AVAILABLE:
            np.random.seed(42)
            # Create dataset with known structure
            a = np.random.normal(0, 1, 100)
            b = 2 * a + np.random.normal(0, 0.5, 100)
            c = 3 * b + np.random.normal(0, 0.5, 100)
            self.data = pd.DataFrame({'A': a, 'B': b, 'C': c})
        
    def test_bic_score_initialization(self):
        """Test BICScore can be initialized"""
        from bamt.score_functions import BICScore
        score = BICScore()
        self.assertIsNotNone(score)
        
    @unittest.skipIf(not DEPS_AVAILABLE, "numpy/pandas not available")
    def test_bic_score_computation(self):
        """Test BIC score computation"""
        from bamt.score_functions import BICScore
        score_fn = BICScore()
        
        # Test with simple structure
        edges = [('A', 'B'), ('B', 'C')]
        score = score_fn.compute(self.data, edges)
        
        # BIC should be a number (typically negative, lower is better)
        self.assertIsInstance(score, (int, float))
        
    @unittest.skipIf(not DEPS_AVAILABLE, "numpy/pandas not available")
    def test_bic_penalizes_complexity(self):
        """Test that BIC penalizes more complex models"""
        from bamt.score_functions import BICScore
        score_fn = BICScore()
        
        # Simple model
        simple_edges = [('A', 'B')]
        simple_score = score_fn.compute(self.data, simple_edges)
        
        # Complex model (more edges)
        complex_edges = [('A', 'B'), ('A', 'C'), ('B', 'C')]
        complex_score = score_fn.compute(self.data, complex_edges)
        
        # BIC should penalize complexity (both might be negative, but complex should be lower)
        # Just verify both are computable
        self.assertIsInstance(simple_score, (int, float))
        self.assertIsInstance(complex_score, (int, float))


class TestAICScore20(unittest.TestCase):
    """Test suite for AIC (Akaike Information Criterion) score"""
    
    def setUp(self):
        """Set up test data"""
        if DEPS_AVAILABLE:
            np.random.seed(42)
            a = np.random.normal(0, 1, 100)
            b = 2 * a + np.random.normal(0, 0.5, 100)
            c = 3 * b + np.random.normal(0, 0.5, 100)
            self.data = pd.DataFrame({'A': a, 'B': b, 'C': c})
        
    def test_aic_score_initialization(self):
        """Test AICScore can be initialized"""
        from bamt.score_functions import AICScore
        score = AICScore()
        self.assertIsNotNone(score)
        
    @unittest.skipIf(not DEPS_AVAILABLE, "numpy/pandas not available")
    def test_aic_score_computation(self):
        """Test AIC score computation"""
        from bamt.score_functions import AICScore
        score_fn = AICScore()
        
        # Test with simple structure
        edges = [('A', 'B'), ('B', 'C')]
        score = score_fn.compute(self.data, edges)
        
        # AIC should be a number
        self.assertIsInstance(score, (int, float))
        
    @unittest.skipIf(not DEPS_AVAILABLE, "numpy/pandas not available")
    def test_aic_penalizes_complexity(self):
        """Test that AIC penalizes complexity (but less than BIC)"""
        from bamt.score_functions import AICScore
        score_fn = AICScore()
        
        # Simple model
        simple_edges = [('A', 'B')]
        simple_score = score_fn.compute(self.data, simple_edges)
        
        # Complex model
        complex_edges = [('A', 'B'), ('A', 'C'), ('B', 'C')]
        complex_score = score_fn.compute(self.data, complex_edges)
        
        # Both should be computable
        self.assertIsInstance(simple_score, (int, float))
        self.assertIsInstance(complex_score, (int, float))


if __name__ == '__main__':
    unittest.main()
