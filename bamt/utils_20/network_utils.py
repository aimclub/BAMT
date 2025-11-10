"""
Utility functions for saving and loading Bayesian networks in BAMT 2.0.0
"""
import json
import pickle
from pathlib import Path
from typing import Union


def save_network(network, filepath: Union[str, Path], format: str = 'json'):
    """
    Save a Bayesian network to a file.
    
    Args:
        network: A BayesianNetwork object
        filepath: Path to save the network
        format: Format to use ('json' or 'pickle')
        
    Raises:
        ValueError: If format is not supported
    """
    filepath = Path(filepath)
    
    if format == 'json':
        # Save as JSON (structure only, distributions not serialized)
        data = {
            'type': network.__class__.__name__,
            'nodes': network.nodes if hasattr(network, 'nodes') else [],
            'edges': network.edges if hasattr(network, 'edges') else [],
        }
        
        # Add type information if available (HybridBN)
        if hasattr(network, 'node_types'):
            data['node_types'] = network.node_types
            
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
    elif format == 'pickle':
        # Save entire network object with pickle
        with open(filepath, 'wb') as f:
            pickle.dump(network, f)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'pickle'")


def load_network(filepath: Union[str, Path], format: str = 'json'):
    """
    Load a Bayesian network from a file.
    
    Args:
        filepath: Path to the network file
        format: Format to use ('json' or 'pickle')
        
    Returns:
        A BayesianNetwork object (structure only for JSON, full object for pickle)
        
    Raises:
        ValueError: If format is not supported
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if format == 'json':
        # Load structure from JSON
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # Import the appropriate network class
        from ..models.probabilistic_structural_models import (
            ContinuousBayesianNetwork,
            DiscreteBayesianNetwork,
            HybridBayesianNetwork
        )
        
        network_type = data.get('type', 'ContinuousBayesianNetwork')
        
        if network_type == 'ContinuousBayesianNetwork':
            network = ContinuousBayesianNetwork()
        elif network_type == 'DiscreteBayesianNetwork':
            network = DiscreteBayesianNetwork()
        elif network_type == 'HybridBayesianNetwork':
            network = HybridBayesianNetwork()
        else:
            # Default to continuous
            network = ContinuousBayesianNetwork()
            
        # Set structure
        edges = data.get('edges', [])
        if edges:
            network.set_structure([tuple(e) for e in edges])
            
        # Restore node types if available
        if 'node_types' in data and hasattr(network, 'node_types'):
            network.node_types = data['node_types']
            
        return network
        
    elif format == 'pickle':
        # Load full object from pickle
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'pickle'")


def network_to_dict(network) -> dict:
    """
    Convert a network to a dictionary representation.
    
    Args:
        network: A BayesianNetwork object
        
    Returns:
        Dictionary with network information
    """
    data = {
        'type': network.__class__.__name__,
        'nodes': network.nodes if hasattr(network, 'nodes') else [],
        'edges': network.edges if hasattr(network, 'edges') else [],
        'num_nodes': len(network.nodes) if hasattr(network, 'nodes') else 0,
        'num_edges': len(network.edges) if hasattr(network, 'edges') else 0,
    }
    
    if hasattr(network, 'node_types'):
        data['node_types'] = network.node_types
        
    if hasattr(network, 'node_models'):
        data['fitted'] = len(network.node_models) > 0
        
    return data
