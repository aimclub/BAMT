import numpy as np
import math
import itertools

"""
Function for quantifying the joint 
distribution of synthetic data

SRMSE = sqrt(SUM(f - f')^2/M1*M2*...Mn)

where f and f' - are numbers of particular 
combinations of specific attribute values 
appears in the validation dataset and 
synthesized data.

M1, M2,...Mn - are the numbers of 
attributes levels.


Input:
-synth_data
Synthetic dataset with size N

-test_data
Validation dataset with size N

-attributes
List of attributes by which to calculate 
the metric and which represents 
the numbers of nodes in the network




Output:
Metric value


"""





def SRMSE(synth_data: np.ndarray, test_data: np.ndarray, attributes: list) -> float:
    summa = 0
    values = []
    M = 1
    for atr in attributes:
        if len(np.unique(synth_data[:,atr])) > len(np.unique(test_data[:,atr])):
             values.append(np.unique(synth_data[:,atr]))
             M *= len(np.unique(synth_data[:,atr]))
        else:
            values.append(np.unique(test_data[:,atr]))
            M *= len(np.unique(test_data[:,atr]))
    combinations = []
    for xs in itertools.product(*values):
        combinations.append(list(xs))
    
    synth_data = synth_data[:, attributes]
    test_data = test_data[:, attributes]
    for comb in combinations:
        f1_ = (synth_data == comb).all(axis=1).sum() 
        f1 = (test_data == comb).all(axis=1).sum()
        summa += (((f1 - f1_)**2) / M)
    srmse = math.sqrt(summa)
    return (srmse)


    


