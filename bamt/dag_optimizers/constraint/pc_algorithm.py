"""
PC (Peter-Clark) Algorithm for structure learning in Bayesian Networks.

The PC algorithm is a constraint-based structure learning algorithm that
uses conditional independence tests to learn the DAG structure.
"""

from typing import Optional, List, Tuple, Set
import pandas as pd
import networkx as nx
from bamt.dag_optimizers.constraint.constraint_dag_optimizer import (
    ConstraintDAGOptimizer,
)


class PCAlgorithm(ConstraintDAGOptimizer):
    """
    PC Algorithm for constraint-based structure learning.

    The algorithm works in two phases:
    1. Skeleton construction: Find undirected edges using conditional independence tests
    2. Edge orientation: Orient edges to form a DAG

    Example:
        >>> from bamt.dag_optimizers.constraint.pc_algorithm import PCAlgorithm
        >>> optimizer = PCAlgorithm(alpha=0.05)
        >>> structure = optimizer.optimize(data)
    """

    def __init__(
        self,
        alpha: float = 0.05,
        independence_test: str = "chi_square",
        max_cond_vars: int = 5,
    ):
        """
        Initialize PC Algorithm.

        Args:
            alpha: Significance level for independence tests
            independence_test: Type of independence test ('chi_square', 'g_sq', 'pearson')
            max_cond_vars: Maximum number of conditioning variables to test
        """
        super().__init__()
        self.alpha = alpha
        self.independence_test = independence_test
        self.max_cond_vars = max_cond_vars

    def optimize(self, data: pd.DataFrame, **kwargs) -> nx.DiGraph:
        """
        Learn DAG structure using PC algorithm.

        Args:
            data: Training data as pandas DataFrame
            **kwargs: Additional parameters

        Returns:
            Learned DAG structure as networkx DiGraph

        Note:
            This implementation uses pgmpy's PC algorithm internally.
        """
        try:
            from pgmpy.estimators import PC
        except ImportError:
            raise ImportError(
                "pgmpy is required for PC algorithm. Install with: pip install pgmpy"
            )

        # Use pgmpy's PC implementation
        pc = PC(data)

        # Run skeleton construction and edge orientation
        dag = pc.estimate(
            variant="stable",
            ci_test=self.independence_test,
            max_cond_vars=self.max_cond_vars,
            significance_level=self.alpha,
            return_type="dag",
        )

        return dag

    def __repr__(self) -> str:
        return (
            f"PCAlgorithm(alpha={self.alpha}, "
            f"independence_test='{self.independence_test}', "
            f"max_cond_vars={self.max_cond_vars})"
        )

    def __str__(self) -> str:
        return f"PC Algorithm (α={self.alpha})"
