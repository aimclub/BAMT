"""
Hybrid DAG optimizer that combines constraint-based and score-based methods.

This module implements hybrid structure learning that combines constraint-based
methods (like PC algorithm) with score-based methods (like Hill Climbing).
"""

from typing import List, Optional

import networkx as nx
import pandas as pd

from bamt.dag_optimizers.constraint.pc_algorithm import PCAlgorithm
from bamt.dag_optimizers.score.hill_climbing import HillClimbing
from bamt.score_functions import ScoreFunction
from bamt.dag_optimizers.dag_optimizer import DAGOptimizer


class HybridDAGOptimizer(DAGOptimizer):
    """
    Hybrid structure optimizer combining PC and Hill Climbing.

    This optimizer uses a two-phase approach:
    1. Phase 1 (PC Algorithm): Identifies skeleton and initial edge directions
       using conditional independence tests
    2. Phase 2 (Hill Climbing): Refines the structure using a scoring function

    This approach combines the strengths of both methods:
    - PC provides a good initial structure based on independence
    - Hill Climbing optimizes the score within constraints

    Attributes:
        constraint_optimizer: Constraint-based optimizer (e.g., PC Algorithm)
        score_optimizer: Score-based optimizer (e.g., Hill Climbing)
    """

    def __init__(
        self,
        score_function: ScoreFunction,
        significance_level: float = 0.05,
        max_iter: int = 200,
        max_parents: int = 3,
    ):
        """
        Initialize Hybrid optimizer.

        Args:
            score_function: Scoring function for Hill Climbing phase
            significance_level: Significance level for PC algorithm (default: 0.05)
            max_iter: Maximum iterations for Hill Climbing (default: 200)
            max_parents: Maximum parents per node (default: 3)
        """
        super().__init__()
        self.constraint_optimizer = PCAlgorithm(significance_level=significance_level)
        self.score_optimizer = HillClimbing(
            score_function=score_function,
            max_iter=max_iter,
            max_parents=max_parents,
        )

    def optimize(
        self,
        data: pd.DataFrame,
        node_names: Optional[List[str]] = None,
        **kwargs
    ) -> nx.DiGraph:
        """
        Find the optimal DAG structure using hybrid approach.

        Phase 1: Use PC algorithm to get initial structure
        Phase 2: Refine with Hill Climbing

        Args:
            data: The dataset to learn structure from
            node_names: Names of nodes (defaults to column names)
            **kwargs: Additional parameters

        Returns:
            nx.DiGraph: The learned DAG structure
        """
        if node_names is None:
            node_names = list(data.columns)

        # Phase 1: PC Algorithm to get initial structure and whitelist
        print("Phase 1: Running PC Algorithm to identify skeleton...")
        pc_graph = self.constraint_optimizer.optimize(data, node_names, **kwargs)

        # Convert PC graph edges to whitelist for Hill Climbing
        # PC gives undirected edges and some directed edges
        # We'll use both as constraints for Hill Climbing
        init_edges = []
        for u, v in pc_graph.edges():
            # PC returns directed edges, we keep them as-is
            init_edges.append((u, v))

        # Phase 2: Hill Climbing with PC-derived constraints
        print(f"Phase 2: Running Hill Climbing with {len(init_edges)} initial edges...")

        # Use PC edges as initial structure
        # Allow Hill Climbing to modify them
        self.score_optimizer.init_edges = [
            (node_names.index(u), node_names.index(v)) for u, v in init_edges
        ]

        # Run Hill Climbing
        final_graph = self.score_optimizer.optimize(data, node_names, **kwargs)

        return final_graph
