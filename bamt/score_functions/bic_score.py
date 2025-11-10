from typing import List, Tuple
try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None

from .score_function import ScoreFunction


class BICScore(ScoreFunction):
    """
    BIC (Bayesian Information Criterion) score function for Bayesian networks.
    BIC = log-likelihood - (k/2) * log(n)
    where k is the number of parameters and n is the sample size.
    
    Lower BIC is better (penalizes model complexity more than AIC).
    """
    
    def __init__(self):
        super().__init__()
        
    def compute(self, data, edges: List[Tuple[str, str]]) -> float:
        """
        Compute BIC score for a given DAG structure.
        
        Args:
            data: DataFrame with the data
            edges: List of tuples (parent, child) representing the DAG
            
        Returns:
            float: BIC score (lower is better, typically negative)
        """
        if pd is None or np is None:
            raise ImportError("pandas and numpy required for BICScore")
            
        n = len(data)
        
        # Build parent map
        parent_map = {}
        nodes = set(data.columns)
        for node in nodes:
            parent_map[node] = []
        for parent, child in edges:
            parent_map[child].append(parent)
            
        log_likelihood = 0.0
        num_parameters = 0
        
        # Compute log-likelihood and count parameters for each node
        for node in nodes:
            parents = parent_map[node]
            
            if len(parents) == 0:
                # Root node - Gaussian distribution
                # Parameters: mean and variance
                num_parameters += 2
                
                # Log-likelihood for Gaussian
                mean = data[node].mean()
                var = data[node].var()
                if var > 0:
                    log_likelihood += -0.5 * n * (np.log(2 * np.pi * var) + 1)
            else:
                # Child node - Linear Gaussian model
                # Parameters: coefficients for each parent + intercept + variance
                num_parameters += len(parents) + 2
                
                # Simple Gaussian log-likelihood
                # In full implementation, would fit linear regression
                var = data[node].var()
                if var > 0:
                    log_likelihood += -0.5 * n * (np.log(2 * np.pi * var) + 1)
                    
        # BIC = log-likelihood - (k/2) * log(n)
        bic = log_likelihood - (num_parameters / 2.0) * np.log(n)
        
        return bic
        
    def estimate(self):
        """Legacy method for compatibility"""
        pass
