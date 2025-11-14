"""Base class for constraint-based DAG optimizers."""

from bamt.dag_optimizers.dag_optimizer import DAGOptimizer


class ConstraintDAGOptimizer(DAGOptimizer):
    """
    Base class for constraint-based DAG structure learning algorithms.

    Constraint-based algorithms use conditional independence tests to
    learn the network structure.
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Constraint-based DAG Optimizer"
