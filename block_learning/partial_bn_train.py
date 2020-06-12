from pomegranate import BayesianNetwork
from block_learning.train_bn import train_model
import numpy as np

"""
Function for adding new BN and 
trainig new connected BN with 
hidden vars

Input:
-bn1
Pre-trained BN to join

-data
New source dataset with discretized variables

-init_nodes
list of initial nodes which
can't have parents

-cluster
Number of clusters for KMeans to fill
the hidden var

Output:
BN with partial training

"""



def partial_model_train(bn1: BayesianNetwork, data: np.ndarray, init_nodes: list = None, clusters: int = 5) -> BayesianNetwork:
    
    hidden_input_var = np.array(bn1.marginal()[-1].sample(data.shape[0]))
    new_data = np.column_stack((hidden_input_var, data))
    bn = train_model(new_data, clusters = clusters, init_nodes = init_nodes)
    return(bn)
    

    
    
  
    

    



    

