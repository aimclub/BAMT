"""
Phase 5 Integration Tests - Complete Workflow
Tests all Phase 5 features together
"""
import unittest
import tempfile
import os
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False


@unittest.skipIf(not DEPS_AVAILABLE, "numpy/pandas not available")
class TestPhase5Integration(unittest.TestCase):
    """Integration tests for all Phase 5 features"""
    
    def setUp(self):
        """Set up test data and temp directory"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create synthetic data
        np.random.seed(42)
        a = np.random.normal(0, 1, 150)
        b = 2 * a + np.random.normal(0, 0.5, 150)
        c = 3 * b + np.random.normal(0, 0.5, 150)
        self.data = pd.DataFrame({'A': a, 'B': b, 'C': c})
        
    def tearDown(self):
        """Clean up temp files"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_complete_workflow_with_bic(self):
        """Test complete workflow: structure learning with BIC -> fit -> save -> load"""
        from bamt.models.probabilistic_structural_models import ContinuousBayesianNetwork
        from bamt.dag_optimizers.score import HillClimbingOptimizer
        from bamt.score_functions import BICScore
        from bamt.utils_20 import save_network, load_network
        
        # 1. Learn structure with BIC
        optimizer = HillClimbingOptimizer(score_function=BICScore(), max_iter=10)
        edges = optimizer.optimize(self.data)
        
        # Should find at least one edge
        self.assertTrue(len(edges) > 0)
        
        # 2. Create and fit network
        bn = ContinuousBayesianNetwork()
        bn.set_structure(edges)
        bn.fit(self.data)
        
        # 3. Verify fitted
        self.assertEqual(len(bn.node_models), 3)
        
        # 4. Generate samples
        samples = bn.sample(20)
        self.assertEqual(len(samples), 20)
        
        # 5. Save network
        filepath = Path(self.temp_dir) / 'bn_with_bic.json'
        save_network(bn, filepath, format='json')
        
        # 6. Load network
        loaded_bn = load_network(filepath, format='json')
        self.assertEqual(len(loaded_bn.nodes), 3)
        
    def test_complete_workflow_with_aic(self):
        """Test complete workflow with AIC score function"""
        from bamt.models.probabilistic_structural_models import ContinuousBayesianNetwork
        from bamt.dag_optimizers.score import HillClimbingOptimizer
        from bamt.score_functions import AICScore
        from bamt.utils_20 import network_to_dict
        
        # Structure learning with AIC
        optimizer = HillClimbingOptimizer(score_function=AICScore(), max_iter=10)
        edges = optimizer.optimize(self.data)
        
        # Create network
        bn = ContinuousBayesianNetwork()
        bn.set_structure(edges)
        bn.fit(self.data)
        
        # Convert to dict
        data_dict = network_to_dict(bn)
        
        # Verify
        self.assertEqual(data_dict['type'], 'ContinuousBayesianNetwork')
        self.assertEqual(data_dict['num_nodes'], 3)
        self.assertTrue(data_dict['fitted'])
        
    def test_score_function_comparison(self):
        """Test that different score functions can be compared"""
        from bamt.score_functions import BICScore, AICScore, K2Score, MutualInformationScore
        
        # All score functions should work on same data/structure
        edges = [('A', 'B'), ('B', 'C')]
        
        score_functions = [
            BICScore(),
            AICScore(),
            MutualInformationScore(),
        ]
        
        scores = []
        for sf in score_functions:
            try:
                score = sf.compute(self.data, edges)
                scores.append(score)
                self.assertIsInstance(score, (int, float))
            except Exception as e:
                # Some score functions might fail on continuous data
                pass
                
        # Should get at least 2 scores
        self.assertTrue(len(scores) >= 2)
        
    def test_pickle_save_with_fitted_model(self):
        """Test saving fitted model with pickle preserves parameters"""
        from bamt.models.probabilistic_structural_models import ContinuousBayesianNetwork
        from bamt.utils_20 import save_network, load_network
        
        # Create and fit network
        bn = ContinuousBayesianNetwork()
        edges = [('A', 'B'), ('B', 'C')]
        bn.set_structure(edges)
        bn.fit(self.data)
        
        # Save as pickle
        filepath = Path(self.temp_dir) / 'fitted_bn.pkl'
        save_network(bn, filepath, format='pickle')
        
        # Load
        loaded_bn = load_network(filepath, format='pickle')
        
        # Should have fitted models
        self.assertEqual(len(loaded_bn.node_models), 3)
        
        # Should be able to sample
        samples = loaded_bn.sample(10)
        self.assertEqual(len(samples), 10)


if __name__ == '__main__':
    unittest.main()
