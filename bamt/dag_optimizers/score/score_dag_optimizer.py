from bamt.dag_optimizers.dag_optimizer import DAGOptimizer


class ScoreDAGOptimizer(DAGOptimizer):
    def __init__(self):
        super().__init__()

    def optimize(self, data, scorer, formatter=None):
        pass
