from pyBN.learning.structure.score.random_restarts import hc_rr
from pomegranate import*
from pyBN.classes.bayesnet import BayesNet
import json
from sklearn.cluster import KMeans
import numpy as np
"""
Function for learning a BN model
and saving this model to json file
*Idea*
Each BN has output hidden var, which aggregates
info from BN

Input:
-data
Source dataset with discretized variables
-cluster
Number of clusters for KMeans to init
the hidden var
-restrict
list of all nodes except init nodes
-name
The name of BN model

Output:
Saving trained BN in a json file


"""

def save_model(data,clusters=5, restrict=None,name=None):
    bn = BayesNet()
    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(data)
    labels = kmeans.labels_
    hidden_dist = DiscreteDistribution.from_samples(labels)
    hidden_var = np.array(hidden_dist.sample(data.shape[0]))
    new_data = np.column_stack((data,hidden_var))
    if restrict==None:
        bn = hc_rr(new_data,latent=[new_data.shape[1]-1])
    else:
        bn = hc_rr(new_data,latent=[new_data.shape[1]-1],restriction=restrict)
    structure = []
    nodes = sorted(list(bn.nodes()))
    for rv in nodes:
        structure.append(tuple(bn.F[rv]['parents']))
    structure = tuple(structure)
    bn = BayesianNetwork.from_structure(new_data,structure)
    bn.bake()
    hidden_var = np.array([np.nan]*(data.shape[0]))
    new_data = np.column_stack((data,hidden_var))
    bn.predict(new_data)
    bn.fit(new_data)
    
    if name==None:
        name='BN'
    with open('models/'+name+'.json', 'w+') as f:
        json.dump(bn.to_json(), f)
    

    

    


    