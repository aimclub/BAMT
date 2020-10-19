from pyBN.learning.structure.score.random_restarts import hc_rr
from pomegranate import BayesianNetwork, DiscreteDistribution
from pyBN.classes.bayesnet import BayesNet
from sklearn.cluster import KMeans
import numpy as np
from copy import copy
import pandas as pd
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import K2Score, BicScore
from pgmpy.models import BayesianModel

"""
Function for trainig the structure
and parameters of BN from data


Input:
-data
Source dataset with discretized variables

-cluster
Number of clusters for KMeans to fill
the hidden var

-init_nodes
list of initial nodes which
can't have parents

-algorithm
Algorithm for structure 
learning
-MI
-K2


Output:
BayesianNetwork object


"""

def train_model(data: pd.DataFrame, algorithm: str, clusters: int = 5, init_nodes: list = None) -> BayesianNetwork:
    
    bn = BayesianNetwork()
    #Сluster the initial data in order to fill in a hidden variable based on the distribution of clusters
    kmeans = KMeans(n_clusters = clusters, random_state = 0).fit(data.values)
    labels = kmeans.labels_
    hidden_dist = DiscreteDistribution.from_samples(labels)
    hidden_var = np.array(hidden_dist.sample(data.shape[0]))

    #new_data = np.column_stack((data, hidden_var))
    new_data = copy(data)
    new_data['hidden_output'] = hidden_var
    latent = (new_data.shape[1])-1

    #Train the network structure on data taking into account a hidden variable
    if algorithm == 'MI':
        bn = hc_rr(new_data.values, latent = latent, init_nodes = init_nodes)
        structure = []
        nodes = sorted(list(bn.nodes()))
        for rv in nodes:
            structure.append(tuple(bn.F[rv]['parents']))
        structure = tuple(structure)
        bn = BayesianNetwork.from_structure(new_data, structure)
        bn.bake()
    if algorithm == 'K2':
        datacol = new_data.columns.to_list()
        if init_nodes: 
            init_nodes_named = [new_data.columns.to_list()[i] for i in init_nodes]
            blacklist = [(x, y) for x in datacol for y in init_nodes_named if x != y]
        latent_list = [(datacol[latent], x) for x in datacol if x != datacol[latent]]
        blacklist_total = list(set(blacklist + latent_list))
        
        hc_K2Score = HillClimbSearch(new_data, scoring_method=BicScore(new_data))
        best_model_K2Score = hc_K2Score.estimate(black_list=blacklist_total)
        structure = dict([(n, []) for n in new_data.columns])
        for edge in best_model_K2Score.edges():
            structure[edge[1]].append(edge[0])
        column_dict = dict([(n, i) for i, n in enumerate(new_data.columns)])
        structure2 = []
        for n in new_data.columns:
            l = structure[n]
            l1 = []
            for elem in l:
                l1.append(column_dict[elem])
            structure2.append(tuple(l1))
        structure2 = tuple(structure2)
        bn = BayesianNetwork.from_structure(new_data.values, structure2)
        bn.bake()
    #Learn a hidden variable
 
    hidden_var = np.array([np.nan] * (data.shape[0]))
    new_data = np.column_stack((data.values, hidden_var))
    new_data = bn.predict(new_data)
    bn.fit(new_data)
    bn.bake()
    return (bn)