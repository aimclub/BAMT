from pomegranate import BayesianNetwork, DiscreteDistribution
import numpy as np

"""
Function for generating synthetic data 
from BN model


Input:
-bn
BayesianNetwork object

-n
Sample size

Output:
dictionary where:

key - number of the node
values - sample array with size n
"""

def generate_synthetics(bn: BayesianNetwork, n: int = 100) -> dict:
    
    states = np.arange(bn.node_count())
    data = dict([(n, []) for n in states])

    init_idx = []
    for i, edge in enumerate(bn.structure):
        if not(bool(edge)):
            init_idx.append(i)

    init_data = dict([(n, []) for n in init_idx])
    for i in init_idx:
        init_data[i] = bn.marginal()[i].sample(n)
        data[i] = init_data[i]
    
    proba = bn.predict_proba(init_data)
    for i in states:
        if i not in init_idx:
            data[i] = DiscreteDistribution(proba[i].parameters[0]).sample(n)
    return data


    

