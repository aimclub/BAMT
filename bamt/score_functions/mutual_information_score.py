from typing import List, Tuple
try:
    import pandas as pd
    import numpy as np
    from sklearn.metrics import mutual_info_score as sk_mutual_info
    SK_AVAILABLE = True
except ImportError:
    pd = None
    np = None
    SK_AVAILABLE = False

from .score_function import ScoreFunction


class MutualInformationScore(ScoreFunction):
    """
    Mutual Information score function for structure learning.
    Measures the dependency between variables.
    """
    
    def __init__(self):
        super().__init__()
        
    def compute(self, data, edges: List[Tuple[str, str]]) -> float:
        """
        Compute MI-based score for a given DAG structure.
        
        Args:
            data: DataFrame with the data
            edges: List of tuples (parent, child) representing the DAG
            
        Returns:
            float: MI score (higher indicates stronger dependencies)
        """
        if not SK_AVAILABLE:
            raise ImportError("sklearn required for MutualInformationScore")
            
        score = 0.0
        
        # Compute mutual information for each edge
        for parent, child in edges:
            # Discretize continuous variables for MI computation
            parent_data = data[parent]
            child_data = data[child]
            
            # Simple binning for continuous data
            if parent_data.dtype in [np.float64, np.float32]:
                parent_data = pd.cut(parent_data, bins=10, labels=False)
            if child_data.dtype in [np.float64, np.float32]:
                child_data = pd.cut(child_data, bins=10, labels=False)
                
            # Compute mutual information
            mi = sk_mutual_info(parent_data, child_data)
            score += mi
            
        return score
        
    def estimate(self):
        """Legacy method for compatibility"""
        pass
