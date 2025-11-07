from typing import List, Tuple, Optional, Set
try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None

from ..dag_optimizer import DAGOptimizer
from ...score_functions.score_function import ScoreFunction


class HillClimbingOptimizer(DAGOptimizer):
    """
    Hill Climbing structure learning optimizer for Bayesian networks.
    Uses a score function to guide the search.
    """
    
    def __init__(self, score_function: Optional[ScoreFunction] = None, max_iter: int = 100):
        super().__init__()
        self.score_function = score_function
        self.max_iter = max_iter
        
    def optimize(self, data, init_edges: Optional[List[Tuple[str, str]]] = None) -> List[Tuple[str, str]]:
        """
        Find optimal DAG structure using Hill Climbing.
        
        Args:
            data: DataFrame with the data
            init_edges: Optional initial edge list
            
        Returns:
            List of edges representing the learned DAG structure
        """
        if pd is None:
            raise ImportError("pandas required for optimize method")
            
        if self.score_function is None:
            raise ValueError("Score function must be provided")
            
        nodes = list(data.columns)
        current_edges = list(init_edges) if init_edges else []
        current_score = self.score_function.compute(data, current_edges)
        
        improved = True
        iteration = 0
        
        while improved and iteration < self.max_iter:
            improved = False
            iteration += 1
            
            # Try all possible single edge additions, deletions, and reversals
            best_edges = current_edges.copy()
            best_score = current_score
            
            # Try adding edges
            for parent in nodes:
                for child in nodes:
                    if parent != child:
                        edge = (parent, child)
                        if edge not in current_edges:
                            # Check if adding this edge would create a cycle
                            test_edges = list(current_edges) + [edge]
                            if not self._has_cycle(test_edges, nodes):
                                score = self.score_function.compute(data, test_edges)
                                if score > best_score:
                                    best_score = score
                                    best_edges = test_edges
                                    improved = True
                                    
            # Try removing edges
            for edge in current_edges:
                test_edges = [e for e in current_edges if e != edge]
                score = self.score_function.compute(data, test_edges)
                if score > best_score:
                    best_score = score
                    best_edges = test_edges
                    improved = True
                    
            # Try reversing edges
            for edge in current_edges:
                parent, child = edge
                reversed_edge = (child, parent)
                test_edges = [e for e in current_edges if e != edge] + [reversed_edge]
                if not self._has_cycle(test_edges, nodes):
                    score = self.score_function.compute(data, test_edges)
                    if score > best_score:
                        best_score = score
                        best_edges = test_edges
                        improved = True
                        
            current_edges = best_edges
            current_score = best_score
            
        return current_edges
        
    def _has_cycle(self, edges: List[Tuple[str, str]], nodes: List[str]) -> bool:
        """Check if the edge list contains a cycle using DFS."""
        from collections import defaultdict, deque
        
        # Build adjacency list
        adj = defaultdict(list)
        for parent, child in edges:
            adj[parent].append(child)
            
        # Check for cycle using DFS
        visited = set()
        rec_stack = set()
        
        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in adj[node]:
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
                    
            rec_stack.remove(node)
            return False
            
        for node in nodes:
            if node not in visited:
                if dfs(node):
                    return True
                    
        return False
