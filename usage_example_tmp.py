
from bamt.models.probabilistic_structural_models import DiscreteBayesianNetwork
from bamt.score_functions.k2_score import K2Score
from bamt.dag_optimizers.score.hill_climbing import HillClimbingOptimizer

import numpy as np
import pandas as pd
np.random.seed(40)
df = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD'))

# define optimizers and score functions
dag_score_function = K2Score(df)
dag_optimizer = HillClimbingOptimizer()

# get a structure, maybe in networkx format?
G = dag_optimizer.optimize(df, scorer=dag_score_function.scorer_mimic)


# def my_formatter(dag):
#     class AbracadabraNet:
#         def __init__(self, nodes):
#             self.curious_nodes = nodes
#
#         def __repr__(self):
#             return f"Magic Network: {self.curious_nodes}"
#
#     return AbracadabraNet(nodes=dag.nodes)
#
#
# G = dag_optimizer.optimize(df,
#                            scorer=dag_score_function.scorer_mimic,
#                            formatter=my_formatter)

# print(G)
# print(type(G))
# define parameters estimator and BN
# parameters_estimator = ParametersEstimator(**parameters)
bn = DiscreteBayesianNetwork().from_dag(df, G)

# # fit the bn
bn.fit(df)
print(bn)

print(bn.nodes)
print(bn.edges)
# bn.sample(1000)
# bn.predict(data.drop[["col1", "col2"]])
