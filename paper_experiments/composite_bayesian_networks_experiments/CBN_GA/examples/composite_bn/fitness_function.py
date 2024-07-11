import pandas as pd
from golem.core.dag.convert import graph_structure_as_nx_graph
from bamt.networks.continuous_bn import ContinuousBN
import math
import numpy as np
from numpy import std, mean, log, isnan, isinf
from sklearn.metrics import mean_squared_error
from math import log10
from scipy.stats import norm
from ML import ML_models
from examples.composite_bn.composite_model import CompositeModel
from examples.bn.bn_model import BNModel
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.metrics import log_likelihood_score
from pgmpy.estimators import K2Score


class FitnessFunction():



    def classical_metric(self, graph: BNModel, data_train: pd.DataFrame, data_test: pd.DataFrame, percent = 0.5):

        data = pd.concat([data_train, data_test])
        graph_nx, labels = graph_structure_as_nx_graph(graph)
        nodes = [node.content['name'] for node in graph.nodes]
        p_info = {'types': dict.fromkeys(nodes, 'cont'),
                    'signs': dict.fromkeys(nodes, 'pos')}
        struct = []
        for pair in graph_nx.edges():
            l1 = str(labels[pair[0]])
            l2 = str(labels[pair[1]])
            struct.append((l1, l2))
        
        if struct == []:
            return 0

        bn = ContinuousBN()
        bn.add_nodes(p_info)
        bn.set_structure(edges=struct)

        bn.calculate_weights(data)

        weights = bn.weights

        if weights == {}:
            weight_score = 1000
        else:
            n_compl = len(graph.nodes)*(len(graph.nodes)-1)/2
            n_largest = math.ceil(n_compl/2.5)
            keys = sorted(weights, key=weights.get, reverse=True)[:n_largest]
            max_weights = {k:weights[k] for k in keys}
            weight_score = -(np.sum(list(max_weights.values())))

        return weight_score

    def edge_reduction(self, graph: BNModel, data_train: pd.DataFrame, data_test: pd.DataFrame, percent = 0.5):

        data = pd.concat([data_train, data_test])
        graph_nx, labels = graph_structure_as_nx_graph(graph)
        nodes = [node.content['name'] for node in graph.nodes]
        p_info = {'types': dict.fromkeys(nodes, 'cont'),
                    'signs': dict.fromkeys(nodes, 'pos')}

        struct = []
        for pair in graph_nx.edges():
            l1 = str(labels[pair[0]])
            l2 = str(labels[pair[1]])
            struct.append((l1, l2))
        
        if struct == []:
            return 0

        bn = ContinuousBN()
        bn.add_nodes(p_info)
        bn.set_structure(edges=struct)

        bn.calculate_weights(data)

        weights = bn.weights
        n_compl = len(graph.nodes)*(len(graph.nodes)-1)/2
        n_least = len(struct) - math.ceil(n_compl/2.5)
        keys = sorted(weights, key=weights.get)[:n_least]
        min_weights = {k:weights[k] for k in keys}

        for edge,_ in min_weights.items():
                child = graph.get_nodes_by_name(edge[1])[0]
                parent = graph.get_nodes_by_name(edge[0])[0]
                child.nodes_from.remove(parent)

        # for edge,weight in weights.items():
        #     if weight < threshold:
        #         child = graph.get_nodes_by_name(edge[1])[0]
        #         parent = graph.get_nodes_by_name(edge[0])[0]
        #         child.nodes_from.remove(parent)

        return graph


    def classical_metric_2(self, graph: BNModel, data_train: pd.DataFrame, data_test: pd.DataFrame, percent = 0.02):
        score = 0
        edges_count = len(graph.get_edges())
        graph_nx, labels = graph_structure_as_nx_graph(graph)
        struct = []
        for pair in graph_nx.edges():
            l1 = str(labels[pair[0]])
            l2 = str(labels[pair[1]])
            struct.append([l1, l2])
        
        bn_model = BayesianNetwork(struct)
        bn_model.add_nodes_from(data_test.columns)    
        bn_model.fit(
        data=data_train,
        estimator=MaximumLikelihoodEstimator
    )

        score = log_likelihood_score(bn_model, data_test)
        score -= (edges_count*percent)*log10(len(data_test))*edges_count
        
        score = round(-score)

        return score


    def composite_metric(self, graph: CompositeModel, data_train: pd.DataFrame, data_test: pd.DataFrame, percent = 0.02):
        # data_all = data
        # data_train , data_test = train_test_split(data_all, train_size = 0.8, random_state=42, shuffle = False)
        # try:
        score, len_data = 0, len(data_train)
        for node in graph.nodes:   
            data_of_node_train = data_train[node.content['name']]
            data_of_node_test = data_test[node.content['name']]
            if node.nodes_from != [] and node.content['parent_model'] == None:
                ml_models = ML_models()
                node.content['parent_model'] = ml_models.get_model_by_children_type(node)                     

            if node.nodes_from == None or node.nodes_from == []:
                if node.content['type'] == 'cont':
                    mu, sigma = mean(data_of_node_train), std(data_of_node_train)
                    score += norm.logpdf(data_of_node_test.values, loc=mu, scale=sigma).sum()
                else:
                    count = data_of_node_train.value_counts()            
                    frequency  = log(count / len_data)
                    index = frequency.index.tolist()
                    for value in data_of_node_test:
                        if value in index:
                            score += frequency[value]

            else:
                if node.content['parent_model'] == 'XGBClassifier':
                    print('1')
                model, columns, target, idx = ML_models().dict_models[node.content['parent_model']](), [n.content['name'] for n in node.nodes_from], data_of_node_train.to_numpy(), data_train.index.to_numpy()
                setattr(model, 'max_iter', 100000)
                features = data_train[columns].to_numpy()                
                if len(set(target)) == 1:
                    continue            
                fitted_model = model.fit(features, target)

                idx=data_test.index.to_numpy()
                features=data_test[columns].to_numpy()
                target=data_of_node_test.to_numpy()            
                if node.content['type'] == 'cont':
                    predict = fitted_model.predict(features)        
                    mse =  mean_squared_error(target, predict, squared=False) + 0.0000001
                    a = norm.logpdf(target, loc=predict, scale=mse)
                    score += a.sum()                
                else:
                    predict_proba = fitted_model.predict_proba(features)
                    idx = pd.array(list(range(len(target))))
                    li = []
                    
                    for i in idx:
                        a = predict_proba[i]
                        try:
                            b = a[target[i]]
                        except:
                            b = 0.0000001
                        if b<0.0000001:
                            b = 0.0000001
                        li.append(log(b))
                    score += sum(li)

        edges_count = len(graph.get_edges())
        # score -= (edges_count*percent)*log10(len(data_test))*edges_count    

        # except Exception as ex:
        #     print(ex)
        #     print('Problem')

        try:
            score = round(-score)
        except:
            score = -score

        return score

    def classical_K2(self, graph: CompositeModel, data_train: pd.DataFrame, data_test: pd.DataFrame, percent = 0.02):
        score = 0
        graph_nx, labels = graph_structure_as_nx_graph(graph)
        struct = []
        for pair in graph_nx.edges():
            l1 = str(labels[pair[0]])
            l2 = str(labels[pair[1]])
            struct.append([l1, l2])
        
        bn_model = BayesianNetwork(struct)
        bn_model.add_nodes_from(data_train.columns)    
        
        score = K2Score(data_train).score(bn_model)

        score = round(-score, 2)

        return score