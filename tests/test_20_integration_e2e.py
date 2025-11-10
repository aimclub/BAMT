"""
End-to-End Integration Tests for BAMT 2.0.0
Demonstrates the complete workflow from data to trained network
"""
import unittest

try:
    import numpy as np
    import pandas as pd
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False


@unittest.skipIf(not DEPS_AVAILABLE, "numpy/pandas not available")
class TestE2EContinuousWorkflow(unittest.TestCase):
    """Test complete workflow with continuous data"""
    
    def test_continuous_workflow_with_structure_learning(self):
        """Test: Load data -> Learn structure -> Fit parameters -> Sample -> Predict"""
        # 1. Create synthetic data with known structure: A -> B -> C
        np.random.seed(42)
        a = np.random.normal(0, 1, 200)
        b = 2 * a + np.random.normal(0, 0.5, 200)
        c = 3 * b + np.random.normal(0, 0.5, 200)
        data = pd.DataFrame({'A': a, 'B': b, 'C': c})
        
        # 2. Learn structure using Hill Climbing
        from bamt.dag_optimizers.score import HillClimbingOptimizer
        from bamt.score_functions import MutualInformationScore
        
        optimizer = HillClimbingOptimizer(
            score_function=MutualInformationScore(),
            max_iter=50
        )
        learned_edges = optimizer.optimize(data)
        
        # Should find at least one edge
        self.assertTrue(len(learned_edges) > 0)
        
        # 3. Create network and set structure
        from bamt.models.probabilistic_structural_models import ContinuousBayesianNetwork
        bn = ContinuousBayesianNetwork()
        bn.set_structure(learned_edges)
        
        # 4. Fit parameters
        bn.fit(data)
        self.assertEqual(len(bn.node_models), 3)
        
        # 5. Generate samples
        samples = bn.sample(50)
        self.assertEqual(len(samples), 50)
        self.assertEqual(set(samples.columns), {'A', 'B', 'C'})
        
        # 6. Make predictions
        evidence = pd.DataFrame({'A': [0.5, 1.0]})
        predictions = bn.predict(evidence, target=['B', 'C'])
        self.assertEqual(len(predictions), 2)
        self.assertIn('B', predictions.columns)
        self.assertIn('C', predictions.columns)
        
    def test_continuous_workflow_with_known_structure(self):
        """Test workflow with user-defined structure"""
        # Create data
        np.random.seed(42)
        data = pd.DataFrame({
            'X1': np.random.normal(0, 1, 100),
            'X2': np.random.normal(0, 1, 100),
            'X3': np.random.normal(0, 1, 100)
        })
        
        # Define structure
        from bamt.models.probabilistic_structural_models import ContinuousBayesianNetwork
        bn = ContinuousBayesianNetwork()
        edges = [('X1', 'X2'), ('X2', 'X3')]
        bn.set_structure(edges)
        
        # Fit and sample
        bn.fit(data)
        samples = bn.sample(30)
        
        self.assertEqual(len(samples), 30)
        self.assertEqual(len(bn.edges), 2)


@unittest.skipIf(not DEPS_AVAILABLE, "numpy/pandas not available")
class TestE2EDiscreteWorkflow(unittest.TestCase):
    """Test complete workflow with discrete data"""
    
    def test_discrete_workflow(self):
        """Test discrete network workflow"""
        # Create discrete data
        np.random.seed(42)
        data = pd.DataFrame({
            'Weather': np.random.choice(['Sunny', 'Rainy', 'Cloudy'], 100),
            'Temperature': np.random.choice(['Hot', 'Mild', 'Cold'], 100),
            'Activity': np.random.choice(['Indoor', 'Outdoor'], 100)
        })
        
        # Create network
        from bamt.models.probabilistic_structural_models import DiscreteBayesianNetwork
        bn = DiscreteBayesianNetwork()
        edges = [('Weather', 'Activity'), ('Temperature', 'Activity')]
        bn.set_structure(edges)
        
        # Fit
        bn.fit(data)
        self.assertEqual(len(bn.node_models), 3)
        
        # Sample
        samples = bn.sample(20)
        self.assertEqual(len(samples), 20)


@unittest.skipIf(not DEPS_AVAILABLE, "numpy/pandas not available")
class TestE2EHybridWorkflow(unittest.TestCase):
    """Test complete workflow with mixed data types"""
    
    def test_hybrid_workflow(self):
        """Test hybrid network with continuous and discrete variables"""
        # Create hybrid data
        np.random.seed(42)
        data = pd.DataFrame({
            'Age': np.random.normal(40, 10, 100),
            'Income': np.random.normal(50000, 15000, 100),
            'Education': np.random.choice(['HS', 'BS', 'MS', 'PhD'], 100),
            'Employed': np.random.choice(['Yes', 'No'], 100)
        })
        
        # Create network
        from bamt.models.probabilistic_structural_models import HybridBayesianNetwork
        bn = HybridBayesianNetwork()
        edges = [('Education', 'Income'), ('Age', 'Employed')]
        bn.set_structure(edges)
        
        # Fit
        bn.fit(data)
        
        # Check type inference
        self.assertEqual(bn.node_types['Age'], 'continuous')
        self.assertEqual(bn.node_types['Income'], 'continuous')
        self.assertEqual(bn.node_types['Education'], 'discrete')
        self.assertEqual(bn.node_types['Employed'], 'discrete')
        
        # Sample
        samples = bn.sample(15)
        self.assertEqual(len(samples), 15)
        self.assertEqual(len(samples.columns), 4)


@unittest.skipIf(not DEPS_AVAILABLE, "numpy/pandas not available")
class TestE2ESklearnLikeAPI(unittest.TestCase):
    """Test that the API follows sklearn conventions"""
    
    def test_sklearn_like_fit_predict(self):
        """Test sklearn-like fit/predict pattern"""
        np.random.seed(42)
        X_train = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'target': np.random.normal(0, 1, 100)
        })
        
        from bamt.models.probabilistic_structural_models import ContinuousBayesianNetwork
        
        # sklearn-like usage
        model = ContinuousBayesianNetwork()
        model.set_structure([('feature1', 'target'), ('feature2', 'target')])
        model.fit(X_train)
        
        # Predict
        X_test = pd.DataFrame({
            'feature1': [0.5],
            'feature2': [-0.3]
        })
        predictions = model.predict(X_test, target=['target'])
        
        self.assertIsInstance(predictions, pd.DataFrame)
        self.assertIn('target', predictions.columns)


if __name__ == '__main__':
    unittest.main()
