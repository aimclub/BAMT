from typing import List, Tuple, Optional
try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None

from .bayesian_network import BayesianNetwork


class HybridBayesianNetwork(BayesianNetwork):
    """
    Hybrid Bayesian Network supporting both continuous and discrete variables.
    Uses the 2.0.0 architecture with sklearn-like interface.
    """
    
    def __init__(self):
        super().__init__()
        self.nodes = []
        self.edges = []
        self.node_models = {}  # Maps node name to fitted distribution/model
        self.node_types = {}  # Maps node name to 'continuous' or 'discrete'
        
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
        
    def _infer_node_type(self, data, node: str) -> str:
        """Infer if a node is continuous or discrete based on data."""
        col = data[node]
        if col.dtype in [np.float64, np.float32, np.int64, np.int32]:
            # Check if it's actually discrete (few unique values)
            if col.nunique() < 20:  # Heuristic: < 20 unique values = discrete
                return 'discrete'
            return 'continuous'
        return 'discrete'
        
    def fit(self, data):
        """
        Fit the network parameters given data and structure.
        
        Args:
            data: DataFrame with columns matching node names
        """
        if pd is None:
            raise ImportError("pandas is required for fit method")
        
        # Import distributions
        from ...core.node_models import ContinuousDistribution, EmpiricalDistribution
            
        # Infer node types
        for node in self.nodes:
            self.node_types[node] = self._infer_node_type(data, node)
            
        # Build parent map
        parent_map = {node: [] for node in self.nodes}
        for parent, child in self.edges:
            parent_map[child].append(parent)
            
        # Fit each node
        for node in self.nodes:
            parents = parent_map[node]
            node_type = self.node_types[node]
            
            if len(parents) == 0:
                # Root node
                if node_type == 'continuous':
                    dist = ContinuousDistribution()
                    dist.fit(data[node].values)
                else:
                    dist = EmpiricalDistribution()
                    dist.fit(data[node].values)
                    
                self.node_models[node] = {
                    'type': 'root',
                    'distribution': dist,
                    'parents': [],
                    'node_type': node_type
                }
            else:
                # Child node
                if node_type == 'continuous':
                    dist = ContinuousDistribution()
                    dist.fit(data[node].values)
                else:
                    dist = EmpiricalDistribution()
                    dist.fit(data[node].values)
                    
                self.node_models[node] = {
                    'type': 'child',
                    'distribution': dist,
                    'parents': parents,
                    'node_type': node_type
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
            
        predictions = {}
        for node in target:
            if node in self.node_models:
                model = self.node_models[node]
                dist = model['distribution']
                pred = dist.sample(len(evidence))
                predictions[node] = pred
                
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
                # Not fitted yet
                samples[node] = np.random.normal(0, 1, num_samples)
                
        return pd.DataFrame(samples)

    def __str__(self):
        return f"Hybrid Bayesian Network (2.0) with {len(self.nodes)} nodes ({len([n for n in self.node_types.values() if n == 'continuous'])} continuous, {len([n for n in self.node_types.values() if n == 'discrete'])} discrete)"
