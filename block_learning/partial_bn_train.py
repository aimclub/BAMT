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
from block_learning.save_bn import save_structure, save_params
from block_learning.read_bn import read_structure, read_params
from kmodes.kmodes import KModes



def connect_partial_bn(bn1: HyBayesianNetwork, bn2: HyBayesianNetwork, data: pd.DataFrame, name: str, n_clusters: int = 3) -> HyBayesianNetwork:
    type1 = get_nodes_type(data[bn1.V])
    type2 = get_nodes_type(data[bn2.V])
    hybn = HyBayesianNetwork()
    if ('disc' not in type1.values()) & (('disc' not in type2.values())):
        latent_sample = np.random.normal(0, 1, data.shape[0])
        latent_sample = [x for x in latent_sample]
        data[name] = latent_sample
        skeleton = dict()
        skeleton['V'] = data.columns.to_list()
        dag = [x for x in bn1.E] + [x for x in bn2.E] + [[x, name] for x in bn1.V] + [[name, x] for x in bn2.V]
        skeleton['E'] = dag
        nodes_type = get_nodes_type(data)
        param_dict = parameter_learning(data, nodes_type, skeleton)
        save_structure(skeleton, 'Structure_with_'+name)
        skel = read_structure('Structure_with_'+name)
        save_params(param_dict, 'Params_with_' + name)
        params = read_params('Params_with_' + name)
        hybn = HyBayesianNetwork(skel, params)
    else:
        km = KModes(n_clusters=n_clusters, init='Huang', n_init=5)
        clusters = km.fit_predict(data)
        latent_sample = [int(x) for x in clusters]
        data[name] = latent_sample
        skeleton = dict()
        skeleton['V'] = data.columns.to_list()
        dag = [x for x in bn1.E] + [x for x in bn2.E] + [[x, name] for x in bn1.V if type1[x] != 'cont'] + [[name, x] for x in bn2.V if len(bn2.getparents(x))==0]
        skeleton['E'] = dag
        nodes_type = get_nodes_type(data)
        param_dict = parameter_learning(data, nodes_type, skeleton)
        save_structure(skeleton, 'Structure_with_'+name)
        skel = read_structure('Structure_with_'+name)
        save_params(param_dict, 'Params_with_' + name)
        params = read_params('Params_with_' + name)
        hybn = HyBayesianNetwork(skel, params)
    return hybn





    



# def partial_model_train(bn1: BayesianNetwork, data: pd.DataFrame, algorithm: str, init_nodes: list = None, clusters: int = 5) -> BayesianNetwork:
    
#     hidden_input_var = np.array(bn1.marginal()[-1].sample(data.shape[0]))
#     new_data = pd.DataFrame()
#     new_data['hidden_input'] = hidden_input_var
#     new_data = pd.concat([new_data, data], axis=1)
#     #new_data = np.column_stack((hidden_input_var, data))
#     bn = train_model(new_data, algorithm = algorithm, clusters = clusters, init_nodes = init_nodes)
#     return(bn)
    

    
    
  
    

    



    

