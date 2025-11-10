"""
Tests for BAMT 2.0.0 utility functions
"""
import unittest
import tempfile
import os
from pathlib import Path


class TestNetworkUtils20(unittest.TestCase):
    """Test suite for network save/load utilities"""
    
    def setUp(self):
        """Set up temp directory for test files"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up temp files"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_save_load_json(self):
        """Test saving and loading network in JSON format"""
        from bamt.models.probabilistic_structural_models import ContinuousBayesianNetwork
        from bamt.utils_20 import save_network, load_network
        
        # Create a network
        bn = ContinuousBayesianNetwork()
        edges = [('A', 'B'), ('B', 'C')]
        bn.set_structure(edges)
        
        # Save to JSON
        filepath = Path(self.temp_dir) / 'test_network.json'
        save_network(bn, filepath, format='json')
        
        # Verify file exists
        self.assertTrue(filepath.exists())
        
        # Load network
        loaded_bn = load_network(filepath, format='json')
        
        # Check structure is preserved
        self.assertEqual(len(loaded_bn.edges), 2)
        self.assertEqual(set(loaded_bn.nodes), {'A', 'B', 'C'})
        
    def test_network_to_dict(self):
        """Test converting network to dictionary"""
        from bamt.models.probabilistic_structural_models import HybridBayesianNetwork
        from bamt.utils_20 import network_to_dict
        
        # Create a hybrid network
        bn = HybridBayesianNetwork()
        edges = [('X1', 'X2'), ('X2', 'X3')]
        bn.set_structure(edges)
        
        # Convert to dict
        data = network_to_dict(bn)
        
        # Verify content
        self.assertEqual(data['type'], 'HybridBayesianNetwork')
        self.assertEqual(data['num_nodes'], 3)
        self.assertEqual(data['num_edges'], 2)
        self.assertIn('nodes', data)
        self.assertIn('edges', data)
        
    def test_save_different_network_types(self):
        """Test saving different network types"""
        from bamt.models.probabilistic_structural_models import (
            ContinuousBayesianNetwork,
            DiscreteBayesianNetwork,
            HybridBayesianNetwork
        )
        from bamt.utils_20 import save_network, load_network
        
        network_types = [
            (ContinuousBayesianNetwork(), 'continuous'),
            (DiscreteBayesianNetwork(), 'discrete'),
            (HybridBayesianNetwork(), 'hybrid')
        ]
        
        for network, name in network_types:
            edges = [('A', 'B')]
            network.set_structure(edges)
            
            filepath = Path(self.temp_dir) / f'test_{name}.json'
            save_network(network, filepath, format='json')
            
            # Load and verify
            loaded = load_network(filepath, format='json')
            self.assertEqual(len(loaded.edges), 1)


if __name__ == '__main__':
    unittest.main()
