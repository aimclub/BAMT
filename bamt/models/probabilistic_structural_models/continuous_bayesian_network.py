from typing import List, Tuple, Optional, Union
try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None

from .bayesian_network import BayesianNetwork


class ContinuousBayesianNetwork(BayesianNetwork):
    """
    Bayesian Network for continuous variables using the 2.0.0 architecture.
    Implements fit, predict, and sample methods with sklearn-like interface.
    """
    
    def __init__(self):
        super().__init__()
        self.nodes = []
        self.edges = []
        self.node_models = {}  # Maps node name to fitted distribution/model
        
    def set_structure(self, edges: List[Tuple[str, str]]):
        """
        Set the network structure from a list of edges.
        
        Args:
            edges: List of tuples (parent, child) defining the DAG structure
        """
        self.edges = edges
        # Extract unique nodes
        nodes_set = set()
        for parent, child in edges:
            nodes_set.add(parent)
            nodes_set.add(child)
        self.nodes = list(nodes_set)
        
    def fit(self, data):
        """
        Fit the network parameters given data and structure.
        
        Args:
            data: DataFrame with columns matching node names
        """
        if pd is None:
            raise ImportError("pandas is required for fit method")
        
        # Import here to avoid circular dependencies
        from ...core.node_models import ContinuousDistribution
            
        # Build parent map
        parent_map = {node: [] for node in self.nodes}
        for parent, child in self.edges:
            parent_map[child].append(parent)
            
        # Fit each node
        for node in self.nodes:
            parents = parent_map[node]
            if len(parents) == 0:
                # Root node - fit unconditional distribution
                dist = ContinuousDistribution()
                dist.fit(data[node].values)
                self.node_models[node] = {
                    'type': 'root',
                    'distribution': dist,
                    'parents': []
                }
            else:
                # Child node - for now, fit simple conditional model
                # In full implementation, this would use regression or conditional distribution
                dist = ContinuousDistribution()
                dist.fit(data[node].values)
                self.node_models[node] = {
                    'type': 'child',
                    'distribution': dist,
                    'parents': parents
                }

    def predict(self, evidence, target: Optional[List[str]] = None):
        """
        Predict target variables given evidence.
        
        Args:
            evidence: DataFrame with evidence variables
            target: List of target variable names to predict
            
        Returns:
            DataFrame with predictions for target variables
        """
        if pd is None:
            raise ImportError("pandas is required for predict method")
            
        if target is None:
            # Predict all nodes not in evidence
            target = [n for n in self.nodes if n not in evidence.columns]
            
        # Simple prediction: use mean of fitted distribution
        # In full implementation, this would do proper inference
        predictions = {}
        for node in target:
            if node in self.node_models:
                model = self.node_models[node]
                dist = model['distribution']
                # Sample from distribution as prediction
                pred = dist.sample(len(evidence))
                if hasattr(pred, '__iter__'):
                    predictions[node] = pred
                else:
                    # Single value, replicate it
                    predictions[node] = [pred] * len(evidence)
                
        return pd.DataFrame(predictions)

    def sample(self, num_samples: int):
        """
        Generate samples from the network.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            DataFrame with sampled data
        """
        if pd is None or np is None:
            raise ImportError("pandas and numpy are required for sample method")
            
        samples = {}
        
        # Topological sort to sample in order
        from collections import defaultdict, deque
        
        in_degree = defaultdict(int)
        adj_list = defaultdict(list)
        
        for parent, child in self.edges:
            adj_list[parent].append(child)
            in_degree[child] += 1
            
        # All nodes with no parents
        queue = deque([n for n in self.nodes if in_degree[n] == 0])
        topo_order = []
        
        while queue:
            node = queue.popleft()
            topo_order.append(node)
            for neighbor in adj_list[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
                    
        # Sample in topological order
        for node in topo_order:
            if node in self.node_models:
                model = self.node_models[node]
                dist = model['distribution']
                samples[node] = dist.sample(num_samples)
            else:
                # Not fitted yet - use default values
                import warnings
                warnings.warn(f"Node {node} not fitted, using default sampling")
                samples[node] = np.zeros(num_samples)
                
        return pd.DataFrame(samples)

    def __str__(self):
        return f"Continuous Bayesian Network (2.0) with {len(self.nodes)} nodes and {len(self.edges)} edges"

