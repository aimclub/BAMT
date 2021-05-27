import itertools
from copy import copy
import math
from preprocess.discretization import get_nodes_type
import numpy as np
import pandas as pd
from pgmpy.base import DAG
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import K2Score, PC
from pomegranate import DiscreteDistribution, ConditionalProbabilityTable
from scipy.stats import norm
from sklearn import linear_model
from bayesian.structure_score import MIG, LLG, BICG, AICG
from sklearn.mixture import GaussianMixture
from bayesian.redef_HC import hc as hc_method
from bayesian.redef_info_scores import info_score
from bayesian.mi_entropy_gauss import mi
import datetime
import random
from functools import partial
from networkx.algorithms.cycles import simple_cycles
from pgmpy.estimators import K2Score
from pgmpy.models import BayesianModel
from fedot.core.chains.chain_convert import chain_as_nx_graph
from fedot.core.chains.chain_validation import has_no_self_cycled_nodes
from fedot.core.chains.graph import GraphObject
from fedot.core.chains.graph_node import PrimaryGraphNode, SecondaryGraphNode
from fedot.core.composer.gp_composer.gp_composer import ChainGenerationParams, GPComposerRequirements
from fedot.core.composer.optimisers.gp_comp.gp_optimiser import (
    GPChainOptimiser,
    GPChainOptimiserParameters,
    GeneticSchemeTypesEnum)
from fedot.core.composer.optimisers.gp_comp.operators.mutation import MutationTypesEnum
from fedot.core.log import default_log





random.seed(1)
np.random.seed(1)


def k2_metric(network: GraphObject, data: pd.DataFrame):
    nodes = data.columns.to_list()
    graph, labels = chain_as_nx_graph(network)
    struct = []
    for pair in graph.edges():
        struct.append([str(labels[pair[0]]), str(labels[pair[1]])])
    bn_model = BayesianModel(struct)
    no_nodes = []
    for node in nodes:
        if node not in bn_model.nodes():
            no_nodes.append(node)

    #return [random.random()]
    density = 10000*(2*(len(struct)) / ((len(nodes) - 1)*len(nodes)))
    nodes_have = 10000*(len(no_nodes) / len(nodes))
    score = K2Score(data).score(bn_model) #- density
    #score = score + nodes_have
    return [score]


def _has_no_duplicates(graph):
    _, labels = chain_as_nx_graph(graph)
    list_of_nodes = []
    for node in labels.values():
        list_of_nodes.append(str(node))
    if len(list_of_nodes) != len(set(list_of_nodes)):
        raise ValueError('Chain has duplicates')
    return True


def _has_disc_parents(graph):
    """node_types = {'Tectonic regime': 'disc',
                  'Period': 'disc',
                  'Lithology': 'disc',
                  'Structural setting': 'disc',
                  'Hydrocarbon type': 'disc',
                  'Gross': 'cont',
                  'Netpay': 'cont',
                  'Porosity': 'cont',
                  'Permeability': 'cont',
                  'Depth': 'cont'}"""
    graph, labels = chain_as_nx_graph(graph)
    global node_type
    for pair in graph.edges():
        if (node_type[str(labels[pair[1]])] == 'disc') & (node_type[str(labels[pair[0]])] == 'cont'):
            raise ValueError(f'Discrete node has cont parent')
    return True

# def _has_many_parents(graph, nodes):
#     graph, labels = chain_as_nx_graph(graph)
#     struct = []
#     for pair in graph.edges():
#         struct.append([str(labels[pair[0]]), str(labels[pair[1]])])
#     children_parent = dict()
#     for node in nodes:
#         for pair in struct:
#             if node in pair:





def _has_no_cycle(graph: GraphObject):
    nx_graph, _ = chain_as_nx_graph(graph)
    cycled = list(simple_cycles(nx_graph))
    if len(cycled) > 0:
        raise ValueError('Chain has cycle')
    return True


def mi_metric(network: GraphObject, data: pd.DataFrame):
    nodes = data.columns.to_list()
    graph, labels = chain_as_nx_graph(network)
    struct = []
    for pair in graph.edges():
        struct.append([str(labels[pair[0]]), str(labels[pair[1]])])
    bn_model = BayesianModel(struct)
    no_nodes = []
    for node in nodes:
        if node not in bn_model.nodes():
            no_nodes.append(node)

    #return [random.random()]
   
    density = (2*(len(struct)) / ((len(nodes) - 1)*len(nodes)))
    # if mi(struct, data) >= 0:
    #     score = mi(struct, data) - 10*density
    # else:
    #     score = mi(struct, data) - 100*density
    score = mi(struct, data, method='BIC') #- 100*density
    #score = mi(struct, data) - 10*(2*(len(struct)) / ((len(nodes) - 1)*len(nodes)))#+ 100*(len(no_nodes) / len(nodes))
    return [score]

def info_metric(network: GraphObject, data: pd.DataFrame, method='LL'):
    nodes = data.columns.to_list()
    graph, labels = chain_as_nx_graph(network)
    struct = []
    for pair in graph.edges():
        struct.append([str(labels[pair[0]]), str(labels[pair[1]])])
    # no_nodes = []
    # for node in nodes:
    #     if node not in bn_model.nodes():
    #         no_nodes.append(node)

    #return [random.random()]
    score = info_score(struct, data, method) #+ 100*(len(no_nodes) / len(nodes))
    return [score]


def run_bayesian_K2(data: pd.DataFrame, max_lead_time: datetime.timedelta = datetime.timedelta(minutes=5)):
    #data = pd.read_csv(f'{project_root()}\\data\\geo_encoded.csv')
    nodes_types = ['Tectonic regime', 'Period', 'Lithology',
                   'Structural setting', 'Hydrocarbon type', 'Gross', 'Netpay',
                   'Porosity', 'Permeability', 'Depth']
    rules = [has_no_self_cycled_nodes, _has_no_cycle, _has_no_duplicates, _has_disc_parents]

    requirements = GPComposerRequirements(
        primary=nodes_types,
        secondary=nodes_types, max_arity=4,
        max_depth=3, pop_size=50, num_of_generations=50,
        crossover_prob=0.8, mutation_prob=0.9, max_lead_time=max_lead_time)

    optimiser_parameters = GPChainOptimiserParameters(
        genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
        mutation_types=[
            MutationTypesEnum.simple,
            MutationTypesEnum.reduce,
            MutationTypesEnum.growth,
            MutationTypesEnum.local_growth])

    chain_generation_params = ChainGenerationParams(
        chain_class=GraphObject,
        primary_node_func=PrimaryGraphNode,
        secondary_node_func=SecondaryGraphNode,
        rules_for_constraint=rules)

    optimizer = GPChainOptimiser(
        chain_generation_params=chain_generation_params,
        metrics=[],
        parameters=optimiser_parameters,
        requirements=requirements, initial_chain=None,
        log=default_log(logger_name='Bayesian', verbose_level=4))

    optimized_network = optimizer.optimise(partial(k2_metric, data=data))

    return optimized_network


def run_bayesian_MI(data: pd.DataFrame,  node_types: dict, max_lead_time: datetime.timedelta = datetime.timedelta(minutes=15)):
    #data = pd.read_csv(f'{project_root()}\\data\\geo_encoded.csv')
    nodes_types = data.columns.to_list()
    global node_type
    node_type = copy(node_types)
    rules = [has_no_self_cycled_nodes, _has_no_cycle, _has_no_duplicates, _has_disc_parents]

    requirements = GPComposerRequirements(
        primary=nodes_types,
        secondary=nodes_types, max_arity=4,
        max_depth=3, pop_size=50, num_of_generations=30,
        crossover_prob=0.8, mutation_prob=0.9, max_lead_time=max_lead_time)

    optimiser_parameters = GPChainOptimiserParameters(
        genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
        mutation_types=[
            MutationTypesEnum.simple,
            MutationTypesEnum.reduce,
            MutationTypesEnum.growth,
            MutationTypesEnum.local_growth])

    chain_generation_params = ChainGenerationParams(
        chain_class=GraphObject,
        primary_node_func=PrimaryGraphNode,
        secondary_node_func=SecondaryGraphNode,
        rules_for_constraint=rules)

    optimizer = GPChainOptimiser(
        chain_generation_params=chain_generation_params,
        metrics=[],
        parameters=optimiser_parameters,
        requirements=requirements, initial_chain=None,
        log=default_log(logger_name='Bayesian', verbose_level=4))

    optimized_network = optimizer.optimise(partial(mi_metric, data=data))

    return optimized_network

def run_bayesian_info(data: pd.DataFrame, max_lead_time: datetime.timedelta = datetime.timedelta(minutes=5), method = 'LL'):
    #data = pd.read_csv(f'{project_root()}\\data\\geo_encoded.csv')
    nodes_types = data.columns.to_list()
    global node_types
    node_types = get_nodes_type(data)
    rules = [has_no_self_cycled_nodes, _has_no_cycle, _has_no_duplicates, _has_disc_parents]

    requirements = GPComposerRequirements(
        primary=nodes_types,
        secondary=nodes_types, max_arity=5,
        max_depth=3, pop_size=50, num_of_generations=30,
        crossover_prob=0.8, mutation_prob=0.9, max_lead_time=max_lead_time)

    optimiser_parameters = GPChainOptimiserParameters(
        genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
        mutation_types=[
            MutationTypesEnum.simple,
            MutationTypesEnum.reduce,
            MutationTypesEnum.growth,
            MutationTypesEnum.local_growth])

    chain_generation_params = ChainGenerationParams(
        chain_class=GraphObject,
        primary_node_func=PrimaryGraphNode,
        secondary_node_func=SecondaryGraphNode,
        rules_for_constraint=rules)

    optimizer = GPChainOptimiser(
        chain_generation_params=chain_generation_params,
        metrics=[],
        parameters=optimiser_parameters,
        requirements=requirements, initial_chain=None,
        log=default_log(logger_name='Bayesian', verbose_level=4))

    optimized_network = optimizer.optimise(partial(info_metric, data=data))

    return optimized_network

def structure_learning(data: pd.DataFrame, search: str, node_type: dict, score: str = 'MI', init_nodes: list = None,
                       white_list: list = None,
                       init_edges: list = None, remove_init_edges: bool = True, black_list: list = None) -> dict:
    """Function for bayesian networks structure learning

    Args:
        data (pd.DataFrame): input encoded and discretized data
        search (str): search strategy (HC, evo)
        score (str): algorith of learning (K2, MI, MI_mixed)
        node_type (dict): dictionary with node types (discrete or continuous)
        init_nodes (list, optional): nodes with no parents. Defaults to None.
        white_list (list, optional): allowable edges. Defaults to None.
        init_edges (list, optional): start edges of graph (set user). Defaults to None.
        remove_init_edges (bool, optional): flag that allow to delete start edges (or not allow). Defaults to True.
        black_list (list, optional): forbidden edges. Defaults to None.

    Returns:
        dict: dictionary with structure (values are lists of nodes and edges)
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
    if black_list:
        blacklist = blacklist + black_list

    skeleton = dict()
    skeleton['V'] = datacol

    if search == 'HC':
        if (score == "MI") | (score == "LL") | (score == "BIC") | (score == "AIC"):
            column_name_dict = dict([(n, i) for i, n in enumerate(datacol)])
            blacklist_new = []
            for pair in blacklist:
                blacklist_new.append((column_name_dict[pair[0]], column_name_dict[pair[1]]))
            if white_list:
                white_list_old = copy(white_list)
                white_list = []
                for pair in white_list_old:
                    white_list.append((column_name_dict[pair[0]], column_name_dict[pair[1]]))
            if init_edges:
                init_edges_old = copy(init_edges)
                init_edges = []
                for pair in init_edges_old:
                    init_edges.append((column_name_dict[pair[0]], column_name_dict[pair[1]]))
            bn = hc_method(data, metric=score, restriction=white_list, init_edges=init_edges, remove_geo_edges=remove_init_edges, black_list=blacklist_new)
            structure = []
            nodes = sorted(list(bn.nodes()))
            for rv in nodes:
                for pa in bn.F[rv]['parents']:
                    structure.append([list(column_name_dict.keys())[list(column_name_dict.values()).index(pa)],
                                  list(column_name_dict.keys())[list(column_name_dict.values()).index(rv)]])
            skeleton['E'] = structure
            
        if score == "K2":
            hc_K2Score = HillClimbSearch(data, scoring_method=K2Score(data))
            if init_edges == None:
                best_model_K2Score = hc_K2Score.estimate(black_list=blacklist, white_list=white_list)
            else:
                if remove_init_edges:
                    startdag = DAG()
                    startdag.add_nodes_from(nodes=datacol)
                    startdag.add_edges_from(ebunch=init_edges)
                    best_model_K2Score = hc_K2Score.estimate(black_list=blacklist, white_list=white_list,
                                                         start_dag=startdag)
                else:
                    best_model_K2Score = hc_K2Score.estimate(black_list=blacklist, white_list=white_list,
                                                         fixed_edges=init_edges)
            structure = [list(x) for x in list(best_model_K2Score.edges())]
            skeleton['E'] = structure


        if score == 'MI_mixed':
            hc_mi_mixed = HillClimbSearch(data, scoring_method=MIG(data=data))
            if init_edges == None:
                best_model_mi_mixed = hc_mi_mixed.estimate(black_list=blacklist, white_list=white_list)
            else:
                if remove_init_edges:
                    startdag = DAG()
                    startdag.add_nodes_from(nodes=datacol)
                    startdag.add_edges_from(ebunch=init_edges)
                    best_model_mi_mixed = hc_mi_mixed.estimate(black_list=blacklist, white_list=white_list,
                                                         start_dag=startdag)
                else:
                    best_model_mi_mixed = hc_mi_mixed.estimate(black_list=blacklist, white_list=white_list,
                                                         fixed_edges=init_edges)
            structure = [list(x) for x in list(best_model_mi_mixed.edges())]
            skeleton['E'] = structure
        
        if (score == 'LL_mixed') | (score == 'BIC_mixed') | (score == 'AIC_mixed'):
            if score == 'LL_mixed':
                hc_mi_mixed = HillClimbSearch(data, scoring_method=LLG(data=data))
            if score == 'BIC_mixed':
                hc_mi_mixed = HillClimbSearch(data, scoring_method=BICG(data=data))
            if score == 'AIC_mixed':
                hc_mi_mixed = HillClimbSearch(data, scoring_method=AICG(data=data))

            if init_edges == None:
                best_model_mi_mixed = hc_mi_mixed.estimate(black_list=blacklist, white_list=white_list)
            else:
                if remove_init_edges:
                    startdag = DAG()
                    startdag.add_nodes_from(nodes=datacol)
                    startdag.add_edges_from(ebunch=init_edges)
                    best_model_mi_mixed = hc_mi_mixed.estimate(black_list=blacklist, white_list=white_list,
                                                         start_dag=startdag)
                else:
                    best_model_mi_mixed = hc_mi_mixed.estimate(black_list=blacklist, white_list=white_list,
                                                         fixed_edges=init_edges)
            structure = [list(x) for x in list(best_model_mi_mixed.edges())]
            skeleton['E'] = structure


    if search == 'evo':
        if score == "MI":
            chain = run_bayesian_MI(data, node_types = node_type)
            graph, labels = chain_as_nx_graph(chain)
            struct = []
            for pair in graph.edges():
                struct.append([str(labels[pair[0]]), str(labels[pair[1]])])
            skeleton['E'] = struct
       
        if score == "K2":
            chain = run_bayesian_K2(data)
            graph, labels = chain_as_nx_graph(chain)
            struct = []
            for pair in graph.edges():
                struct.append([str(labels[pair[0]]), str(labels[pair[1]])])
            skeleton['E'] = struct

        if (score == "LL") | (score == "BIC") | (score == "AIC"):
            chain = run_bayesian_info(data, method = score)
            graph, labels = chain_as_nx_graph(chain)
            struct = []
            for pair in graph.edges():
                struct.append([str(labels[pair[0]]), str(labels[pair[1]])])
            skeleton['E'] = struct

    if search == 'PC':
        pc_search = PC(data)
        if init_edges == None:
            best_model_pc = pc_search.estimate(black_list=blacklist, white_list=white_list)
        else:
            if remove_init_edges:
                startdag = DAG()
                startdag.add_nodes_from(nodes=datacol)
                startdag.add_edges_from(ebunch=init_edges)
                best_model_pc = pc_search.estimate(black_list=blacklist, white_list=white_list,
                                                         start_dag=startdag)
            else:
                best_model_pc = pc_search.estimate(black_list=blacklist, white_list=white_list,
                                                         fixed_edges=init_edges)
        structure = [list(x) for x in list(best_model_pc.edges())]
        skeleton['E'] = structure

    return skeleton

def parameter_learning(data: pd.DataFrame, node_type: dict, skeleton: dict) -> dict:
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
            vals = sorted([str(x) for x in list(dist.parameters[0].keys())])
            cprob = list(dict(sorted(dist.items())).values())
            if (len(children) != 0):
                node_data['Vdata'][node] = {"numoutcomes": numoutcomes, "cprob": cprob, "parents": None, "vals": vals,
                                            "type": "discrete", "children": children}
            else:
                node_data['Vdata'][node] = {"numoutcomes": numoutcomes, "cprob": cprob, "parents": None, "vals": vals,
                                            "type": "discrete", "children": None}
        if (node_type[node] == "disc") & (len(parents) != 0):
            numoutcomes = int(len(data[node].unique()))
            dist = DiscreteDistribution.from_samples(data[node].values)
            vals = sorted([str(x) for x in list(dist.parameters[0].keys())])
            dist = ConditionalProbabilityTable.from_samples(data[parents + [node]].values)
            params = dist.parameters[0]
            cprob = dict()
            for i in range(0, len(params), len(vals)):
                probs = []
                for j in range(i, (i + len(vals))):
                    probs.append(params[j][-1])
                combination = [str(x) for x in params[i][0:len(parents)]]
                cprob[str(combination)] = probs
            if (len(children) != 0):
                node_data['Vdata'][node] = {"numoutcomes": numoutcomes, "cprob": cprob, "parents": parents,
                                            "vals": vals, "type": "discrete", "children": children}
            else:
                node_data['Vdata'][node] = {"numoutcomes": numoutcomes, "cprob": cprob, "parents": parents,
                                            "vals": vals, "type": "discrete", "children": None}
        if (node_type[node] == "cont") & (len(parents) == 0):
            # mean_base, std = norm.fit(data[node].values)
            # variance = std ** 2
            mean_base = np.mean(data[node].values)
            variance = np.var(data[node].values)
            if (len(children) != 0):
                node_data['Vdata'][node] = {"mean_base": mean_base, "mean_scal": [], "parents": None,
                                            "variance": variance, "type": "lg", "children": children}
            else:
                node_data['Vdata'][node] = {"mean_base": mean_base, "mean_scal": [], "parents": None,
                                            "variance": variance, "type": "lg", "children": None}
        if (node_type[node] == "cont") & (len(parents) != 0):
            disc_parents = []
            cont_parents = []
            for parent in parents:
                if node_type[parent] == 'disc':
                    disc_parents.append(parent)
                else:
                    cont_parents.append(parent)

            if (len(disc_parents) == 0):
                # mean_base, std = norm.fit(data[node].values)
                # variance = std ** 2
                # mean_base = np.mean(data[node].values)
                # variance = np.var(data[node].values)
                #model = linear_model.BayesianRidge()
                model = linear_model.LinearRegression()
                predict = []
                if len(parents) == 1:
                    model.fit(np.transpose([data[parents[0]].values]), data[node].values)
                    predict = model.predict(np.transpose([data[parents[0]].values]))
                else:
                    model.fit(data[parents].values, data[node].values)
                    predict = model.predict(data[parents].values)
                variance = (RSE(data[node].values, predict)) ** 2
                if (len(children) != 0):
                    node_data['Vdata'][node] = {"mean_base": model.intercept_, "mean_scal": list(model.coef_),
                                                "parents": parents, "variance": variance, "type": "lg",
                                                "children": children}
                else:
                    node_data['Vdata'][node] = {"mean_base": model.intercept_, "mean_scal": list(model.coef_),
                                                "parents": parents, "variance": variance, "type": "lg",
                                                "children": None}
            if (len(disc_parents) != 0) & (len(cont_parents) != 0):
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
                    mean_base = np.nan
                    variance = np.nan
                    predict = []
                    # if new_data.shape[0] != 0:
                    #     #mean_base, std = norm.fit(new_data[node].values)
                    #     #variance = std ** 2
                    #     mean_base = np.mean(new_data[node].values)
                    #     variance = np.var(new_data[node].values)
                    if new_data.shape[0] != 0:
                        #model = linear_model.BayesianRidge()
                        model = linear_model.LinearRegression()
                        if len(cont_parents) == 1:
                            model.fit(np.transpose([new_data[cont_parents[0]].values]), new_data[node].values)
                            predict = model.predict(np.transpose([new_data[cont_parents[0]].values]))
                        else:
                            model.fit(new_data[cont_parents].values, new_data[node].values)
                            predict = model.predict(new_data[cont_parents].values)
                        key_comb = [str(x) for x in comb]
                        mean_base = model.intercept_
                        variance = (RSE(new_data[node].values, predict)) ** 2
                        hycprob[str(key_comb)] = {'variance': variance, 'mean_base': mean_base,
                                                  'mean_scal': list(model.coef_)}
                    else:
                        scal = list(np.full(len(cont_parents), np.nan))
                        key_comb = [str(x) for x in comb]
                        hycprob[str(key_comb)] = {'variance': variance, 'mean_base': mean_base, 'mean_scal': scal}
                if (len(children) != 0):
                    node_data['Vdata'][node] = {"parents": parents, "type": "lgandd", "children": children,
                                                "hybcprob": hycprob}
                else:
                    node_data['Vdata'][node] = {"parents": parents, "type": "lgandd", "children": None,
                                                "hybcprob": hycprob}
            if (len(disc_parents) != 0) & (len(cont_parents) == 0):
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
                    if new_data.shape[0] != 0:
                        #mean_base, std = norm.fit(new_data[node].values)
                        #variance = std ** 2
                        mean_base = np.mean(new_data[node].values)
                        variance = np.var(new_data[node].values)
                        key_comb = [str(x) for x in comb]
                        hycprob[str(key_comb)] = {'variance': variance, 'mean_base': mean_base, 'mean_scal': []}
                    else:
                        mean_base = np.nan
                        variance = np.nan
                        key_comb = [str(x) for x in comb]
                        hycprob[str(key_comb)] = {'variance': variance, 'mean_base': mean_base, 'mean_scal': []}
                if (len(children) != 0):
                    node_data['Vdata'][node] = {"parents": parents, "type": "lgandd", "children": children,
                                                "hybcprob": hycprob}
                else:
                    node_data['Vdata'][node] = {"parents": parents, "type": "lgandd", "children": None,
                                                "hybcprob": hycprob}

    return node_data


def parameter_learning_mix(data: pd.DataFrame, node_type: dict, skeleton: dict) -> dict:
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
            vals = sorted([str(x) for x in list(dist.parameters[0].keys())])
            cprob = list(dict(sorted(dist.items())).values())
            if (len(children) != 0):
                node_data['Vdata'][node] = {"numoutcomes": numoutcomes, "cprob": cprob, "parents": None, "vals": vals,
                                            "type": "discrete", "children": children}
            else:
                node_data['Vdata'][node] = {"numoutcomes": numoutcomes, "cprob": cprob, "parents": None, "vals": vals,
                                            "type": "discrete", "children": None}
        if (node_type[node] == "disc") & (len(parents) != 0):
            numoutcomes = int(len(data[node].unique()))
            dist = DiscreteDistribution.from_samples(data[node].values)
            vals = sorted([str(x) for x in list(dist.parameters[0].keys())])
            dist = ConditionalProbabilityTable.from_samples(data[parents + [node]].values)
            params = dist.parameters[0]
            cprob = dict()
            for i in range(0, len(params), len(vals)):
                probs = []
                for j in range(i, (i + len(vals))):
                    probs.append(params[j][-1])
                combination = [str(x) for x in params[i][0:len(parents)]]
                cprob[str(combination)] = probs
            if (len(children) != 0):
                node_data['Vdata'][node] = {"numoutcomes": numoutcomes, "cprob": cprob, "parents": parents,
                                            "vals": vals, "type": "discrete", "children": children}
            else:
                node_data['Vdata'][node] = {"numoutcomes": numoutcomes, "cprob": cprob, "parents": parents,
                                            "vals": vals, "type": "discrete", "children": None}
        if (node_type[node] == "cont") & (len(parents) == 0):
            # mean_base, std = norm.fit(data[node].values)
            # variance = std ** 2
            #mean_base = np.mean(data[node].values)
            #variance = np.var(data[node].values)
            n_comp = n_component(data, [node])
            gmm = GaussianMixture(n_components=n_comp)
            gmm.fit(np.transpose([data[node].values]))
            means = gmm.means_
            means_list = []
            for el in means:
                means_list.append(el[0])
            var = gmm.covariances_
            var_list = []
            for v in var:
                var_list.append(v[0][0])
            cat = gmm.predict(np.transpose([data[node].values]))
            dis = str(DiscreteDistribution.from_samples(cat))
            if (len(children) != 0):
                node_data['Vdata'][node] = {"mean_base": means_list, "mean_scal": dis, "parents": None,
                                            "variance": var_list, "type": "lg", "children": children}
            else:
                node_data['Vdata'][node] = {"mean_base": means_list, "mean_scal": dis, "parents": None,
                                            "variance": var_list, "type": "lg", "children": None}
        if (node_type[node] == "cont") & (len(parents) != 0):
            disc_parents = []
            cont_parents = []
            for parent in parents:
                if node_type[parent] == 'disc':
                    disc_parents.append(parent)
                else:
                    cont_parents.append(parent)
            

            if (len(disc_parents) == 0) & (len(cont_parents) != 0):
                # mean_base, std = norm.fit(data[node].values)
                # variance = std ** 2
                # mean_base = np.mean(data[node].values)
                # variance = np.var(data[node].values)
                #model = linear_model.BayesianRidge()
                model = linear_model.LinearRegression()
                #model = SVR(kernel='linear', C=100, gamma='auto')
                #gmm = mixture.GaussianMixture(n_components=10, covariance_type='full')
                
                predict = []
                if len(parents) == 1:
                    model.fit(np.transpose([data[parents[0]].values]), data[node].values)
                    predict = model.predict(np.transpose([data[parents[0]].values]))
                else:
                    model.fit(data[parents].values, data[node].values)
                    predict = model.predict(data[parents].values)
               
                #gmm.fit(data[parents+[node]].values)

                variance = (RSE(data[node].values, predict)) ** 2
                # means_parent = [[] for i in range (N)]#[list(l)[0:-1] for l in list(gmm.means_)]
                # for c in range(N):
                #     for c_p in parents:
                #         means_parent[c].append(gmm_params[c_p][c])

                # mean_node = gmm_params[node]#[list(l)[-1] for l in list(gmm.means_)]
                #mean_node = []
                #means_parent = []
                #labels = gmm.predict(data[parents+[node]].values)
                # labels = gmm.predict(data[cont_columns].values)
                # diff_data = copy(data)
                # diff_data['labels'] = labels
                # variances = []
                # for i in range(0,N,1):
                #     sample = diff_data.loc[diff_data['labels'] == i]
                #     variances.append(np.var(sample[node].values))
                #     mean_node.append(np.mean(sample[node].values))
                #     parent_one = []
                #     for p in parents:
                #         parent_one.append(np.mean(sample[p].values))
                #     means_parent.append(parent_one)
                # for var in variances:
                #     if (str(var) == 'nan'):
                #         variances[variances.index(var)] = 0

                # if (len(children) != 0):
                #     node_data['Vdata'][node] = {"mean_base": mean_node, "mean_scal": means_parent,
                #                                 "parents": parents, "variance": variances, "type": "lg",
                #                                 "children": children}
                # else:
                #     node_data['Vdata'][node] = {"mean_base": mean_node, "mean_scal": means_parent,
                #                                 "parents": parents, "variance": variances, "type": "lg",
                #                                 "children": None}



                if (len(children) != 0):
                    node_data['Vdata'][node] = {"mean_base": model.intercept_, "mean_scal": list(model.coef_),
                                                "parents": parents, "variance": variance, "type": "lg",
                                                "children": children}
                else:
                    node_data['Vdata'][node] = {"mean_base": model.intercept_, "mean_scal": list(model.coef_),
                                                "parents": parents, "variance": variance, "type": "lg",
                                                "children": None}
            if (len(disc_parents) != 0) & (len(cont_parents) != 0):
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
                    predict = []
                    # if new_data.shape[0] != 0:
                    #     #mean_base, std = norm.fit(new_data[node].values)
                    #     #variance = std ** 2
                    #     mean_base = np.mean(new_data[node].values)
                    #     variance = np.var(new_data[node].values)
                    key_comb = [str(x) for x in comb]
                    if new_data.shape[0] != 0:
                        #model = linear_model.BayesianRidge()
                        model = linear_model.LinearRegression()
                        #model = SVR(kernel='linear', C=100, gamma='auto')
                        #if new_data.shape[0] > N:
                            #gmm = mixture.GaussianMixture(n_components=10, covariance_type='full')
                            
                        if len(cont_parents) == 1:
                            model.fit(np.transpose([new_data[cont_parents[0]].values]), new_data[node].values)
                            predict = model.predict(np.transpose([new_data[cont_parents[0]].values]))
                        else:
                            model.fit(new_data[cont_parents].values, new_data[node].values)
                            predict = model.predict(new_data[cont_parents].values)
                        mean_base = model.intercept_
                        variance = (RSE(new_data[node].values, predict)) ** 2
                            #gmm.fit(new_data[cont_parents+[node]].values)
                        # means_parent = []#[list(l)[0:-1] for l in list(gmm.means_)]
                        # mean_node = []#[list(l)[-1] for l in list(gmm.means_)]
                        # labels = gmm.predict(new_data[cont_columns].values)
                        # sample = copy(new_data)
                        # sample['labels'] = labels
                        # variances = []
                        # for i in range(0,N,1):
                        #     sample_small = sample.loc[sample['labels'] == i]
                        #     variances.append(np.var(sample_small[node].values))
                        #     mean_node.append(np.mean(sample_small[node].values))
                        #     parent_one = []
                        #     for p in cont_parents:
                        #         parent_one.append(np.mean(sample_small[p].values))
                        #     means_parent.append(parent_one)
                        # for var in variances:
                        #     if (str(var) == 'nan'):
                        #         variances[variances.index(var)] = 0
                        hycprob[str(key_comb)] = {'variance': variance, 'mean_base': mean_base, 'mean_scal': list(model.coef_)}
                        #if new_data.shape[0] <= N:
                            # model = linear_model.LinearRegression()
                            # predict = []
                            # if len(cont_parents) == 1:
                            #     model.fit(np.transpose([new_data[cont_parents[0]].values]), new_data[node].values)
                            #     predict = model.predict(np.transpose([new_data[cont_parents[0]].values]))
                            # else:
                            #     model.fit(new_data[cont_parents].values, new_data[node].values)
                            #     predict = model.predict(new_data[cont_parents].values)
                            # mean_base = model.intercept_
                            # variance = (RSE(new_data[node].values, predict)) ** 2
                            # hycprob[str(key_comb)] = {'variance': variance, 'mean_base': mean_base, 'mean_scal': list(model.coef_)}
                    else:
                        mean_base = np.nan
                        variance = np.nan
                        scal = list(np.full(len(cont_parents), np.nan))
                        #key_comb = [str(x) for x in comb]
                        hycprob[str(key_comb)] = {'variance': variance, 'mean_base': mean_base, 'mean_scal': scal}
                if (len(children) != 0):
                    node_data['Vdata'][node] = {"parents": parents, "type": "lgandd", "children": children,
                                                "hybcprob": hycprob}
                else:
                    node_data['Vdata'][node] = {"parents": parents, "type": "lgandd", "children": None,
                                                "hybcprob": hycprob}
            if (len(disc_parents) != 0) & (len(cont_parents) == 0):
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
                    key_comb = [str(x) for x in comb]
                    if new_data.shape[0] != 0:
                        #mean_base, std = norm.fit(new_data[node].values)
                        #variance = std ** 2
                        # mean_base = np.mean(new_data[node].values)
                        # variance = np.var(new_data[node].values)
                        # if new_data.shape[0] > 6:
                        #     gmm = mixture.GaussianMixture(n_components=5, covariance_type='spherical')
                            
                        #     gmm.fit(np.transpose([new_data[node].values]))
                        #     mean_node = [list(l)[0] for l in list(gmm.means_)]
                        #     labels = gmm.predict(np.transpose([new_data[node].values]))
                        #     sample = copy(new_data)
                        #     sample['labels'] = labels
                        #     variances = []
                        #     for i in range(0,5,1):
                        #         sample_small = sample.loc[sample['labels'] == i]
                        #         variances.append(sample_small[node].var())
                            
                        #     hycprob[str(key_comb)] = {'variance': variances, 'mean_base': mean_node, 'mean_scal': []}
                        # else:
                        mean_base = np.mean(new_data[node].values)
                        variance = np.var(new_data[node].values)
                            #key_comb = [str(x) for x in comb]
                        hycprob[str(key_comb)] = {'variance': variance, 'mean_base': mean_base, 'mean_scal': []}
                        
                    else:
                        mean_base = np.nan
                        variance = np.nan
                        #key_comb = [str(x) for x in comb]
                        hycprob[str(key_comb)] = {'variance': variance, 'mean_base': mean_base, 'mean_scal': []}
                if (len(children) != 0):
                    node_data['Vdata'][node] = {"parents": parents, "type": "lgandd", "children": children,
                                                "hybcprob": hycprob}
                else:
                    node_data['Vdata'][node] = {"parents": parents, "type": "lgandd", "children": None,
                                                "hybcprob": hycprob}

    return node_data



def RSE(y_true, y_predicted):
   
    y_true = np.array(y_true)
    y_predicted = np.array(y_predicted)
    RSS = np.sum(np.square(y_true - y_predicted))

    rse = math.sqrt(RSS / (len(y_true)))
    return rse

def n_component(data: pd.DataFrame, columns: list):
    bic = -1000000000000000
    n = 0
    for i in range(1, 20, 1):
        gm = GaussianMixture(n_components=i, random_state=0)
        if len(columns) == 1:
            x = np.transpose([data[columns[0]].values])
        else:
            x = data[columns].values
        gm.fit(x)
        ll_current = np.sum(gm.score_samples(x))  
        bic_current = (BIC(ll_current, i, data.shape[0]))
        if bic_current > bic:
            bic = bic_current
            n = i
        else:
            break
    return(n)

def BIC (LL, n_components, n):
    return (LL - ((0.5)*math.log(n)*(n_components-1 + 2*n_components)))