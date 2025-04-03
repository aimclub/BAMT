from bamt.score_functions.score_function import ScoreFunction
from pgmpy.estimators import K2Score as pgmpy_K2
from pgmpy.models.BayesianModel import BayesianNetwork
from pgmpy.base.DAG import DAG


# todo: is it correct to pass entire data into scorer??? (according pgmpy)

class K2Score(ScoreFunction):
    def __init__(self, data):
        super().__init__()
        self.scorer_mimic = pgmpy_K2(data)
        start_dag = DAG()
        start_dag.add_nodes_from(data.columns)

        self.model = BayesianNetwork(start_dag)

    def score(self, data):
        return self.scorer_mimic.score(self.model)

    def local_score(self, variable, parents):
        return self.scorer_mimic.local_score(variable, parents)
