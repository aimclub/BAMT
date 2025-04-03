from bamt.dag_optimizers.dag_optimizer import DAGOptimizer
from pgmpy.estimators import HillClimbSearch
from bamt.core.graph.dag import DirectedAcyclicGraph


class HillClimbingOptimizer(DAGOptimizer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def to_bamt_dag(pgmpy_dag):
        """todo: Temporary method"""
        dag = DirectedAcyclicGraph()
        dag.from_pgmpy(pgmpy_dag)
        return dag

    def optimize(self, data, scorer, formatter=None):
        # todo: params loading
        optimizer = HillClimbSearch(data)
        G = optimizer.estimate(scoring_method=scorer)
        # todo: remove it
        our_dag = self.to_bamt_dag(G)

        if formatter:
            return formatter(G)

        return our_dag
