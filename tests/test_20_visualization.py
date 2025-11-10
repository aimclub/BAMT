"""
Tests for BAMT 2.0.0 Visualization utilities
"""
import unittest

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for testing
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class TestVisualization20(unittest.TestCase):
    """Test suite for visualization utilities"""
    
    @unittest.skipIf(not MATPLOTLIB_AVAILABLE, "matplotlib not available")
    def test_plot_structure(self):
        """Test basic structure plotting"""
        from bamt.visualization import plot_structure
        
        edges = [('A', 'B'), ('B', 'C'), ('A', 'C')]
        fig = plot_structure(edges, title="Test Network")
        
        self.assertIsNotNone(fig)
        
    @unittest.skipIf(not MATPLOTLIB_AVAILABLE, "matplotlib not available")
    def test_plot_network_info(self):
        """Test network info plotting"""
        from bamt.visualization import plot_network_info
        from bamt.models.probabilistic_structural_models import ContinuousBayesianNetwork
        
        # Create a simple network
        bn = ContinuousBayesianNetwork()
        edges = [('X1', 'X2'), ('X2', 'X3')]
        bn.set_structure(edges)
        
        fig = plot_network_info(bn, title="Test Info")
        
        self.assertIsNotNone(fig)


if __name__ == '__main__':
    unittest.main()
