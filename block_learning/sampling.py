from pomegranate import*
import numpy as np



def sample(bn,n=100):
    states = np.arange(bn.node_count())
    data = dict([(n,[]) for n in states])

    init_idx = []
    for i,edge in enumerate(bn.structure):
        if not(bool(edge)):
            init_idx.append(i)
    init_data = dict([(n,[]) for n in init_idx])
    for i in init_idx:
        init_data[i] = bn.marginal()[i].sample(n)
    for i in init_idx:
        data[i] = init_data[i]
    proba = bn.predict_proba(init_data)
    for i in states:
        if i not in init_idx:
            data[i] = DiscreteDistribution(proba[i].parameters[0]).sample(n)
    return data


    

