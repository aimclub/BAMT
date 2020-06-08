from pomegranate import *
import json
from sklearn.cluster import KMeans
import numpy as np
from pyBN.learning.structure.score.random_restarts import hc_rr
from pyBN.classes.bayesnet import BayesNet
"""
Function for connection between pre-trained
BN and new BN via the hidden var

*Idea*
BN + hidden var + new BN
All new BNs connect to previous BN via hidden var
and learn aditional hidden output var for
further connection

Input:
-bn1
Pre-trained BN to join
-data
Source dataset with discretized variables
-restrict
list of all nodes except init nodes
-cluster
Number of clusters for KMeans to init
-name
The name of BN model

"""



def connect_models(bn1,data,restrict,clusters=5, name=None):
    bn = BayesNet()
    hidden_input_var = np.array(bn1.marginal()[-1].sample(data.shape[0]))
    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(data)
    labels = kmeans.labels_
    hidden_dist = DiscreteDistribution.from_samples(labels)
    hidden_output_var = np.array(hidden_dist.sample(data.shape[0]))
    new_data_in = np.column_stack((hidden_input_var,data))
    new_data_out = np.column_stack((new_data_in,hidden_output_var))
    bn = hc_rr(new_data_out,latent=[new_data_out.shape[1]-1],restriction=restrict)
    structure = []
    nodes = sorted(list(bn.nodes()))
    for rv in nodes:
        structure.append(tuple(bn.F[rv]['parents']))
    structure = tuple(structure)
    bn = BayesianNetwork.from_structure(new_data_out,structure)
    bn.bake()
    hidden_output_var = np.array([np.nan]*(data.shape[0]))
    new_data_out = np.column_stack((new_data_in,hidden_output_var))
    bn.predict(new_data_out)
    bn.fit(new_data_out)
    
    if name==None:
        name='connected_models'
    with open('models/'+name+'.json', 'w+') as f:
        json.dump(bn.to_json(), f)
  
    

    



    

