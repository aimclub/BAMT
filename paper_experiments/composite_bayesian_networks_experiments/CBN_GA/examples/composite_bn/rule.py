from golem.core.dag.verification_rules import has_no_cycle, has_no_self_cycled_nodes
from golem.core.dag.convert import graph_structure_as_nx_graph
from math import log10


class Rule():

    def __init__(self, N):
        self.N = N    

    def has_no_duplicates(self, graph):
        _, labels = graph_structure_as_nx_graph(graph)
        if len(labels.values()) != len(set(labels.values())):
            raise ValueError('Custom graph has duplicates')
        return True    

    def has_no_more_than_logN_parents(self, graph):
        if any([node for node in graph.nodes if len(node.nodes_from) > log10(self.N)]):
            raise ValueError('Graph has more than logN parents')
        return True

    def has_no_more_than_N_parents(self, graph):
        if any([node for node in graph.nodes if len(node.nodes_from) > self.N]):
            raise ValueError('Graph has more than logN parents')
        return True


    
    def bn_rules(self):
        return [has_no_self_cycled_nodes, has_no_cycle, self.has_no_duplicates, self.has_no_more_than_N_parents] # , self.has_no_more_than_logN_parents