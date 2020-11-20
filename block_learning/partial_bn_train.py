from pomegranate import DiscreteDistribution
from block_learning.train_bn import parameter_learning
import numpy as np
from copy import copy
import pandas as pd
from libpgm.hybayesiannetwork import HyBayesianNetwork
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import K2Score, BicScore
from pgmpy.base import DAG
import networkx as nx
from data_process.preprocessing import get_nodes_type
import json
from libpgm.nodedata import NodeData
from libpgm.graphskeleton import GraphSkeleton
from sklearn.cluster import KMeans
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


def connect_bn(bn1: HyBayesianNetwork, bn2: HyBayesianNetwork, data: pd.DataFrame, names: list) -> HyBayesianNetwork:
    skeleton = dict()
    # data1 = param_data[bn1.V]
    # kmeans = KMeans(n_clusters=3, random_state=0).fit(data1)
    latent_sample = np.random.normal(0, 1, data.shape[0])
    latent_sample = pd.DataFrame([int(x) for x in latent_sample], columns=[names[1]])
    new_data = pd.concat([data,latent_sample], join_axes=[data.index], axis=1)
    skeleton['V'] = new_data.columns.to_list()
    dag = [x for x in bn1.E] + [x for x in bn2.E] + [[names[0], names[1]]] + [[names[1], names[2]]]
    #s_dag = DAG([(x, name) for x in bn1.V] + [(name, x) for x in bn2.V])
    # nodes_type = get_nodes_type(param_data)
    # white_list = []
    # black_list = []
    # for node in bn1.V:
    #     white_list.append((node,name))
    #     white_list.append((name,node))
    #     if(nodes_type[node] == 'cont'):
    #         black_list.append((node,name))
    # for node in bn2.V:
    #     white_list.append((name,node))
    #     white_list.append((node,name))
    #     if(nodes_type[node] == 'cont'):
    #         black_list.append((node,name))
    
    # hc_K2Score = HillClimbSearch(new_data, scoring_method=K2Score(new_data))
    # best_model_K2Score = hc_K2Score.estimate(white_list=white_list, fixed_edges=dag, black_list=black_list)
    # structure = [list(x) for x in list(best_model_K2Score.edges())]
    skeleton['E'] = dag
    # param_new = pd.concat([param_data,latent_sample], axis=1)
    # print(param_new)
    nodes_type = get_nodes_type(new_data)
    param_dict = parameter_learning(new_data, nodes_type, skeleton)
    json.dump(skeleton, open("skeleton.txt",'w'))
    skel = GraphSkeleton()
    skel.load("skeleton.txt")
    skel.toporder()
    json.dump(param_dict, open("node.txt",'w'))
    nd = NodeData()
    nd.load("node.txt")
    nd.entriestoinstances()
    hybn = HyBayesianNetwork(skel, nd)
    return hybn




    



# def partial_model_train(bn1: BayesianNetwork, data: pd.DataFrame, algorithm: str, init_nodes: list = None, clusters: int = 5) -> BayesianNetwork:
    
#     hidden_input_var = np.array(bn1.marginal()[-1].sample(data.shape[0]))
#     new_data = pd.DataFrame()
#     new_data['hidden_input'] = hidden_input_var
#     new_data = pd.concat([new_data, data], axis=1)
#     #new_data = np.column_stack((hidden_input_var, data))
#     bn = train_model(new_data, algorithm = algorithm, clusters = clusters, init_nodes = init_nodes)
#     return(bn)
    

    
    
  
    

    



    

