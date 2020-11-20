
from pomegranate import BayesianNetwork, DiscreteDistribution, ConditionalProbabilityTable
from pyBN.classes.bayesnet import BayesNet
from sklearn.cluster import KMeans
import numpy as np
from copy import copy
import pandas as pd
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import K2Score, BicScore
from pgmpy.models import BayesianModel
from pgmpy.estimators import MmhcEstimator
from sklearn import linear_model
from scipy.stats import norm
import itertools
from pyBN.learning.structure.score.hill_climbing import hc as hc_method


def structure_learning(data: pd.DataFrame, algorithm: str, node_type: dict, init_nodes: list = None) -> dict:
    """Function for bayesian network structure learning from data

    Args:
        data (pd.DataFrame): input dataset
        algorithm (str): algorithm of structure learning
        node_type (dict): dictionary with node types (discrete or continuous)
        init_nodes (list, optional): List of nodes without parents. Defaults to None.

    Returns:
        dict: structure with list of nodes and list of edges
    """    
    blacklist = []
    datacol = data.columns.to_list()
    if init_nodes: 
        blacklist = [(x, y) for x in datacol for y in init_nodes if x != y]
    for x in datacol:
        for y in datacol:
            if x != y:
                if (node_type[x] == 'cont') & (node_type[y] == 'disc'):
                    blacklist.append((x, y))
    skeleton = dict()
    skeleton['V'] = datacol
    


    if algorithm == "MI":
        column_name_dict = dict([(n,i) for i, n in enumerate(datacol)])
        blacklist_new = []
        for pair in blacklist:
            blacklist_new.append((column_name_dict[pair[0]], column_name_dict[pair[1]]))
        bn = hc_method(data.values, restriction = blacklist_new)
        structure = []
        nodes = sorted(list(bn.nodes()))
        for rv in nodes:
            for pa in bn.F[rv]['parents']:
                structure.append([list(column_name_dict.keys())[list(column_name_dict.values()).index(pa)], list(column_name_dict.keys())[list(column_name_dict.values()).index(rv)]])
        skeleton['E'] = structure
    if algorithm == "K2":
        hc_K2Score = HillClimbSearch(data, scoring_method=K2Score(data))
        best_model_K2Score = hc_K2Score.estimate(black_list=blacklist)
        structure = [list(x) for x in list(best_model_K2Score.edges())]
        skeleton['E'] = structure
    if algorithm == "Bic":
        hc_Bic_Score = HillClimbSearch(data, scoring_method=BicScore(data))
        best_model_Bic_Score =  hc_Bic_Score.estimate(black_list=blacklist)
        structure = [list(x) for x in list(best_model_Bic_Score.edges())]
        skeleton['E'] = structure
    if algorithm == "MMHC":
        mmhc = MmhcEstimator(data)
        skel = mmhc.mmpc()
        hc = HillClimbSearch(data, scoring_method=K2Score(data))
        model = hc.estimate(tabu_length=10, white_list=skel.to_directed().edges(), black_list=blacklist)
        structure = [list(x) for x in list(model.edges())]
        skeleton['E'] = structure

    return skeleton

def parameter_learning (data: pd.DataFrame, node_type: dict, skeleton: dict) -> dict:
    """Function for parameter learning for hybrid BN

    Args:
        data (pd.DataFrame): input dataset
        node_type (dict): dictionary with node types (discrete or continuous)
        skeleton (dict): structure of BN

    Returns:
        dict: dictionary with parameters of distributions in nodes
    """    
    datacol = data.columns.to_list()
    node_data = dict()
    node_data['Vdata'] = dict()
    for node in datacol:
        children = []
        parents = []
        for edge in skeleton['E']:
            if (node in edge):
                if edge.index(node) == 0:
                    children.append(edge[1])
                if edge.index(node) == 1:
                    parents.append(edge[0])          
        if (node_type[node] == "disc") & (len(parents) == 0):
            numoutcomes = int(len(data[node].unique()))
            dist = DiscreteDistribution.from_samples(data[node].values)
            vals = [str(x) for x in list(dist.parameters[0].keys())]
            cprob =  list(dist.parameters[0].values())
            if (len(children) != 0):
                node_data['Vdata'][node] = {"numoutcomes":numoutcomes, "cprob": cprob, "parents": None, "vals":vals, "type": "discrete",  "children": children}
            else:
                node_data['Vdata'][node] = {"numoutcomes":numoutcomes, "cprob": cprob, "parents": None, "vals":vals, "type": "discrete",  "children": None}
        if (node_type[node] == "disc") & (len(parents) != 0):
            numoutcomes = int(len(data[node].unique()))
            dist = DiscreteDistribution.from_samples(data[node].values)
            vals = [str(x) for x in list(dist.parameters[0].keys())]
            dist = ConditionalProbabilityTable.from_samples(data[parents+[node]].values)
            params = dist.parameters[0]
            cprob = dict()
            for i in range(0, len(params), len(vals)):
                probs = []
                for j in range (i, (i + len(vals))):
                    probs.append(params[j][-1])
                combination = [str(x) for x in params[i][0:len(parents)]]
                cprob[str(combination)] = probs
            if (len(children) != 0):
                node_data['Vdata'][node] = {"numoutcomes":numoutcomes, "cprob": cprob, "parents": parents, "vals":vals, "type": "discrete",  "children": children}
            else:
                node_data['Vdata'][node] = {"numoutcomes":numoutcomes, "cprob": cprob, "parents": parents, "vals":vals, "type": "discrete",  "children": None}
        if (node_type[node] == "cont") & (len(parents) == 0):
            mean_base, std = norm.fit(data[node].values)
            variance = std**2
            if (len(children) != 0):
                node_data['Vdata'][node] = {"mean_base":mean_base, "mean_scal": [], "parents": None, "variance":variance, "type": "lg",  "children": children}
            else:
                node_data['Vdata'][node] = {"mean_base":mean_base, "mean_scal": [], "parents": None, "variance":variance, "type": "lg",  "children": None}
        if (node_type[node] == "cont") & (len(parents) != 0):
            disc_parents = []
            cont_parents = []
            for parent in parents:
                if node_type[parent] == 'disc':
                    disc_parents.append(parent)
                else:
                    cont_parents.append(parent)

            if(len(disc_parents) == 0):
                mean_base, std = norm.fit(data[node].values)
                variance = std**2
                model = linear_model.BayesianRidge()
                if len(parents) == 1:
                    model.fit(np.transpose([data[parents[0]].values]), data[node].values)
                else:
                    model.fit(data[parents].values, data[node].values)
                if (len(children) != 0):
                    node_data['Vdata'][node] = {"mean_base":mean_base, "mean_scal": list(model.coef_), "parents": parents, "variance":variance, "type": "lg",  "children": children}
                else:
                    node_data['Vdata'][node] = {"mean_base":mean_base, "mean_scal": list(model.coef_), "parents": parents, "variance":variance, "type": "lg",  "children": None}
            if(len(disc_parents) != 0) &  (len(cont_parents) != 0):
                hycprob = dict()
                values = []
                combinations = []
                for d_p in disc_parents:
                    values.append(np.unique(data[d_p].values))
                for xs in itertools.product(*values):
                    combinations.append(list(xs))
                for comb in combinations:
                    mask = np.full(len(data), True)
                    for col, val in zip(disc_parents, comb):
                         mask = (mask) & (data[col] == val)
                    new_data = data[mask]
                    mean_base, std = norm.fit(new_data[node].values)
                    variance = std**2
                    if new_data.shape[0] != 0:
                        model = linear_model.BayesianRidge()
                        if len(cont_parents) == 1:
                           model.fit(np.transpose([new_data[cont_parents[0]].values]), new_data[node].values)
                        else:
                           model.fit(new_data[cont_parents].values, new_data[node].values)
                        key_comb = [str(x) for x in comb]
                        hycprob[str(key_comb)] = {'variance':variance, 'mean_base':mean_base, 'mean_scal': list(model.coef_)}
                    else:
                        scal = list(np.full(len(cont_parents), 0.0))
                        key_comb = [str(x) for x in comb]
                        hycprob[str(key_comb)] = {'variance':variance, 'mean_base':mean_base, 'mean_scal': scal}
                      
                    
                if (len(children) != 0):
                    node_data['Vdata'][node] = {"parents": parents, "type": "lgandd",  "children": children, "hybcprob":hycprob}
                else:
                    node_data['Vdata'][node] = {"parents": parents, "type": "lgandd",  "children": None, "hybcprob":hycprob}        
            if(len(disc_parents) != 0) &  (len(cont_parents) == 0):
                hycprob = dict()
                values = []
                combinations = []
                for d_p in disc_parents:
                    values.append(np.unique(data[d_p].values))
                for xs in itertools.product(*values):
                    combinations.append(list(xs))
                for comb in combinations:
                    mask = np.full(len(data), True)
                    for col, val in zip(disc_parents, comb):
                         mask = (mask) & (data[col] == val)
                    new_data = data[mask]
                    mean_base, std = norm.fit(new_data[node].values)
                    variance = std**2
                    key_comb = [str(x) for x in comb]
                    hycprob[str(key_comb)] = {'variance':variance, 'mean_base':mean_base, 'mean_scal': []}
                if (len(children) != 0):
                    node_data['Vdata'][node] = {"parents": parents, "type": "lgandd",  "children": children, "hybcprob":hycprob}
                else:
                    node_data['Vdata'][node] = {"parents": parents, "type": "lgandd",  "children": None, "hybcprob":hycprob}    
                    
    return node_data



            
    




# def train_model(data: pd.DataFrame, algorithm: str, clusters: int = 5, init_nodes: list = None) -> BayesianNetwork:
    
#     bn = BayesianNetwork()
#     #Сluster the initial data in order to fill in a hidden variable based on the distribution of clusters
#     kmeans = KMeans(n_clusters = clusters, random_state = 0).fit(data.values)
#     labels = kmeans.labels_
#     hidden_dist = DiscreteDistribution.from_samples(labels)
#     hidden_var = np.array(hidden_dist.sample(data.shape[0]))

#     #new_data = np.column_stack((data, hidden_var))
#     new_data = copy(data)
#     new_data['hidden_output'] = hidden_var
#     latent = (new_data.shape[1])-1

#     #Train the network structure on data taking into account a hidden variable
#     if algorithm == 'MI':
#         bn = hc_rr(new_data.values, latent = latent, init_nodes = init_nodes)
#         structure = []
#         nodes = sorted(list(bn.nodes()))
#         for rv in nodes:
#             structure.append(tuple(bn.F[rv]['parents']))
#         structure = tuple(structure)
#         bn = BayesianNetwork.from_structure(new_data, structure)
#         bn.bake()
#     if algorithm == 'K2':
#         datacol = new_data.columns.to_list()
#         if init_nodes: 
#             init_nodes_named = [new_data.columns.to_list()[i] for i in init_nodes]
#             blacklist = [(x, y) for x in datacol for y in init_nodes_named if x != y]
#         latent_list = [(datacol[latent], x) for x in datacol if x != datacol[latent]]
#         blacklist_total = list(set(blacklist + latent_list))
        
#         hc_K2Score = HillClimbSearch(new_data, scoring_method=BicScore(new_data))
#         best_model_K2Score = hc_K2Score.estimate(black_list=blacklist_total)
#         structure = dict([(n, []) for n in new_data.columns])
#         for edge in best_model_K2Score.edges():
#             structure[edge[1]].append(edge[0])
#         column_dict = dict([(n, i) for i, n in enumerate(new_data.columns)])
#         structure2 = []
#         for n in new_data.columns:
#             l = structure[n]
#             l1 = []
#             for elem in l:
#                 l1.append(column_dict[elem])
#             structure2.append(tuple(l1))
#         structure2 = tuple(structure2)
#         bn = BayesianNetwork.from_structure(new_data.values, structure2)
#         bn.bake()
#     #Learn a hidden variable
 
#     hidden_var = np.array([np.nan] * (data.shape[0]))
#     new_data = np.column_stack((data.values, hidden_var))
#     new_data = bn.predict(new_data)
#     bn.fit(new_data)
#     bn.bake()
#     return (bn)