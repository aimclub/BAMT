import pandas as pd
from math import log10
from golem.core.dag.convert import graph_structure_as_nx_graph
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.metrics import log_likelihood_score
from examples.bn.bn_model import BNModel
from examples.composite_bn.fitness_function import FitnessFunction


class Likelihood():

    def likelihood_function(self, graph: BNModel, data_train: pd.DataFrame, data_val: pd.DataFrame, percent = 0.02):
        LL = 0
        edges_count = len(graph.get_edges())
        graph_nx, labels = graph_structure_as_nx_graph(graph)
        struct = []
        for pair in graph_nx.edges():
            l1 = str(labels[pair[0]])
            l2 = str(labels[pair[1]])
            struct.append([l1, l2])
        
        bn_model = BayesianNetwork(struct)
        bn_model.add_nodes_from(data_val.columns)    
        bn_model.fit(
        data=data_train,
        estimator=MaximumLikelihoodEstimator)

        LL = log_likelihood_score(bn_model, data_val)

        return LL
    
    def likelihood_function_composite(self, graph: BNModel, data_train: pd.DataFrame, data_val: pd.DataFrame, percent = 0.02):
        edges_count = len(graph.get_edges())

        LL = FitnessFunction().composite_metric(graph = graph, data_train = data_train, data_test = data_val)
        # LL -= (edges_count*percent)*log10(len(data_val))*edges_count    

        return - LL