from pomegranate import DiscreteDistribution
from block_learning.train_bn import parameter_learning
import numpy as np
from copy import copy
import pandas as pd
from libpgm.hybayesiannetwork import HyBayesianNetwork
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import K2Score, BicScore
from pgmpy.estimators import MmhcEstimator
from pgmpy.base import DAG
import networkx as nx
from data_process.preprocessing import get_nodes_type
import json
from libpgm.nodedata import NodeData
from libpgm.graphskeleton import GraphSkeleton
from sklearn.cluster import KMeans
from block_learning.save_bn import save_structure, save_params
from block_learning.read_bn import read_structure, read_params
from block_learning.sampling import generate_synthetics
from kmodes.kmodes import KModes



def connect_partial_bn(bn1: HyBayesianNetwork, bn2: HyBayesianNetwork, data: pd.DataFrame, name: str, n_clusters: int = 3) -> (HyBayesianNetwork, dict):
    """Functtion for connection two BNs via latent variable with discrete dist

    Args:
        bn1 (HyBayesianNetwork): input BN
        bn2 (HyBayesianNetwork): output BN
        data (pd.DataFrame): data for parameter learning for conncted BN
        name (str): name of latent var
        n_clusters (int, optional): number of clusters for latent var distribution. Defaults to 3.

    Returns:
        HyBayesianNetwork: connected BN
        dict: latent node type
    """    
    
    hybn = HyBayesianNetwork()
    input_nodes = []
    output_nodes = []
    for v in bn1.V:
        if len(bn1.getchildren(v)) == 0:
            input_nodes.append(v)
    for v in bn2.V:
        if len(bn2.getparents(v)) == 0:
            output_nodes.append(v)
    type1 = get_nodes_type(data[input_nodes])
    type2 = get_nodes_type(data[output_nodes])
    latent_type = dict()

    if (('disc' not in type1.values()) & ('disc' not in type2.values())) | (('disc' not in type1.values()) & (('disc' in type2.values()) & ('cont' in type2.values()))):
        latent_sample = np.random.normal(0, 1, data.shape[0])
        latent_sample = [x for x in latent_sample]
        data[name] = latent_sample
        latent_type[name] = 'cont'
        skeleton = dict()
        skeleton['V'] = data.columns.to_list()
        dag = [x for x in bn1.E] + [x for x in bn2.E] + [[x, name] for x in input_nodes] + [[name, x] for x in output_nodes if type2[x] == 'cont']
        skeleton['E'] = dag
        nodes_type = get_nodes_type(data)
        param_dict = parameter_learning(data, nodes_type, skeleton)
        save_structure(skeleton, 'Structure_with_'+name)
        skel = read_structure('Structure_with_'+name)
        save_params(param_dict, 'Params_with_' + name)
        params = read_params('Params_with_' + name)
        hybn = HyBayesianNetwork(skel, params)
    elif ('disc' not in type1.values()) & ('cont' not in type2.values()):
        flag_disc = False
        while not flag_disc:
            parent_of_children = []
            for node in input_nodes:
                parent_of_children = parent_of_children + bn1.getparents(node)
            parent_of_children = list(set(parent_of_children))
            parents_type = get_nodes_type(data[parent_of_children])
            if 'disc' in parents_type:
                flag_disc=True
                input_nodes = parent_of_children
                type1 = parents_type
            else:
                input_nodes = parent_of_children

        km = KModes(n_clusters=n_clusters, init='Huang', n_init=5)
        clusters = km.fit_predict(data[input_nodes+output_nodes])
        latent_sample = [int(x) for x in clusters]
        data[name] = latent_sample
        latent_type[name] = 'disc'
        skeleton = dict()
        skeleton['V'] = data.columns.to_list()
        dag = [x for x in bn1.E] + [x for x in bn2.E] + [[x, name] for x in input_nodes if type1[x] != 'cont'] + [[name, x] for x in output_nodes]
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
        clusters = km.fit_predict(data[input_nodes+output_nodes])
        latent_sample = [int(x) for x in clusters]
        data[name] = latent_sample
        latent_type[name] = 'disc'
        skeleton = dict()
        skeleton['V'] = data.columns.to_list()
        dag = [x for x in bn1.E] + [x for x in bn2.E] + [[x, name] for x in input_nodes if type1[x] != 'cont'] + [[name, x] for x in output_nodes]
        skeleton['E'] = dag
        nodes_type = get_nodes_type(data)
        param_dict = parameter_learning(data, nodes_type, skeleton)
        save_structure(skeleton, 'Structure_with_'+name)
        skel = read_structure('Structure_with_'+name)
        save_params(param_dict, 'Params_with_' + name)
        params = read_params('Params_with_' + name)
        hybn = HyBayesianNetwork(skel, params)

    return hybn, latent_type


def hierarchical_train (hybns: list, data: pd.DataFrame, latent_node_type: dict) -> HyBayesianNetwork:
    
    edge_union = set()
    for bn in hybns:
        edge_union = edge_union.union(set(tuple(i) for i in bn.E))
    edge_union = list(edge_union)
    edge_union = [list(x) for x in edge_union]
    skeleton = dict()
    skeleton['V'] = data.columns.to_list()
    skeleton['E'] = edge_union
    nodes_type = get_nodes_type(data)
    new_nodes_type = {**nodes_type, **latent_node_type}
    param_dict = parameter_learning(data, new_nodes_type, skeleton)
    save_structure(skeleton, 'Hierarchial_structure')
    skel = read_structure('Hierarchial_structure')
    save_params(param_dict, 'Hierarchial_params')
    params = read_params('Hierarchial_params')
    hybn = HyBayesianNetwork(skel, params)

    return hybn


def direct_connect (hybns: list, data: pd.DataFrame, node_type: dict):
    white_list = []
    fixed_list = []
    nets_connection = dict()
    G = nx.DiGraph()
    for i in range(len(hybns)):
        G.add_node(i)
    for bn in hybns:
        fixed_list = fixed_list + [tuple(E) for E in bn.E]
    for bn1 in hybns:
        for bn2 in hybns:
            if bn1 != bn2:
                nets_connection[str(hybns.index(bn1))+" "+str(hybns.index(bn2))] = 0
                for x in bn1.V:
                    for y in bn2.V:
                        white_list = white_list + [(x, y), (y, x)]                 
    hc_K2Score = HillClimbSearch(data, scoring_method=K2Score(data))
    best_model_K2Score = hc_K2Score.estimate(white_list=white_list, fixed_edges=fixed_list)
    structure = [list(x) for x in list(best_model_K2Score.edges())] 
    # for edge in structure:
    #     if (node_type[edge[0]] == 'cont') & (node_type[edge[1]] == 'disc'):
    #         structure.remove(edge)
    for edge in structure:
        i1 = 0
        i2 = 0
        for bn in hybns:
            if edge[0] in bn.V:
                i1 = hybns.index(bn)
            if edge[1] in bn.V:
                i2 = hybns.index(bn)
        if i1 != i2:
            nets_connection[str(i1)+" "+str(i2)] += 1
    print(nets_connection)
    list_of_connect = []
    for key in nets_connection.keys():
        if nets_connection[key] != 0:
            if nets_connection[key] >= nets_connection[key.split(' ')[1]+" "+key.split(' ')[0]]:
                G.add_edge(int(key.split(' ')[0]), int(key.split(' ')[1]))
                if nx.is_directed_acyclic_graph(G):
                    list_of_connect.append(key)
                else:
                    G.remove_edge(int(key.split(' ')[0]), int(key.split(' ')[1]))
            elif nets_connection[key] < nets_connection[key.split(' ')[1]+" "+key.split(' ')[0]]:
                G.add_edge(int(key.split(' ')[1]), int(key.split(' ')[0]))
                if nx.is_directed_acyclic_graph(G):
                    list_of_connect.append(key.split(' ')[1]+" "+key.split(' ')[0])
                else:
                    G.remove_edge(int(key.split(' ')[1]), int(key.split(' ')[0]))
            
                
    list_of_connect = list(dict.fromkeys(list_of_connect))

    return list_of_connect


def direct_train (hybns: list, data: pd.DataFrame, direct_connect_list: list) -> HyBayesianNetwork:
    list_of_pairs = []
    latent_node_type = dict()
    for pair in direct_connect_list:
        bn1 = hybns[int(pair.split()[0])]
        bn2 = hybns[int(pair.split()[1])]
        name = "L " + pair.split()[0] +"_"+pair.split()[1]
        pair_bn, l = connect_partial_bn(bn1, bn2, data[bn1.V + bn2.V], name)
        latent_node_type[name] = l[name]
        sample = generate_synthetics(pair_bn, data.shape[0])
        sample[name] = sample[name].astype('float')
        data[name] = sample[name]
        list_of_pairs.append(pair_bn)
    bn = hierarchical_train(list_of_pairs, data, latent_node_type)
    return bn
    




    

    
    
  
    

    



    

