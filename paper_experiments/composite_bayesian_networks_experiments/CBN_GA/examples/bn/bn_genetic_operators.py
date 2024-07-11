from bn_model import BNModel
from copy import deepcopy
from math import ceil
from random import choice, sample, randint
from golem.core.dag.graph_utils import ordered_subnodes_hierarchy
from itertools import chain

def custom_crossover_exchange_edges(graph_first: BNModel, graph_second: BNModel, max_depth):

    num_cros = 100
    try:
        for _ in range(num_cros):
            old_edges1 = []
            old_edges2 = []
            new_graph_first=deepcopy(graph_first)
            new_graph_second=deepcopy(graph_second)

            edges_1 = new_graph_first.operator.get_edges()
            edges_2 = new_graph_second.operator.get_edges()
            count = ceil(min(len(edges_1), len(edges_2))/2)
            choice_edges_1 = sample(edges_1, count)
            choice_edges_2 = sample(edges_2, count)
            
            for pair in choice_edges_1:
                new_graph_first.operator.disconnect_nodes(pair[0], pair[1], False)
            for pair in choice_edges_2:
                new_graph_second.operator.disconnect_nodes(pair[0], pair[1], False)  
            
            old_edges1 = new_graph_first.operator.get_edges()
            old_edges2 = new_graph_second.operator.get_edges()

            new_edges_2 = [[new_graph_second.get_nodes_by_name(str(i[0]))[0], new_graph_second.get_nodes_by_name(str(i[1]))[0]] for i in choice_edges_1]
            new_edges_1 = [[new_graph_first.get_nodes_by_name(str(i[0]))[0], new_graph_first.get_nodes_by_name(str(i[1]))[0]] for i in choice_edges_2]

            for pair in new_edges_1:
                if pair not in old_edges1:
                    new_graph_first.operator.connect_nodes(pair[0], pair[1])
            for pair in new_edges_2:
                if pair not in old_edges2:
                    new_graph_second.operator.connect_nodes(pair[0], pair[1])                                             
            
            if old_edges1 != new_graph_first.operator.get_edges() or old_edges2 != new_graph_second.operator.get_edges():
                break
                                  
        if old_edges1 == new_graph_first.operator.get_edges() and new_edges_1!=[] and new_edges_1!=None:
            new_graph_first = deepcopy(graph_first)
        if old_edges2 == new_graph_second.operator.get_edges() and new_edges_2!=[] and new_edges_2!=None:
            new_graph_second = deepcopy(graph_second)
    except Exception as ex:
        print(ex)
    return new_graph_first, new_graph_second

def custom_crossover_exchange_parents_both(graph_first: BNModel, graph_second: BNModel, max_depth):

    
    num_cros = 100
    try:
        for _ in range(num_cros):
            old_edges1 = []
            old_edges2 = []
            parents1 = []
            parents2 = []
            new_graph_first=deepcopy(graph_first)
            new_graph_second=deepcopy(graph_second)

            edges = new_graph_second.operator.get_edges()
            flatten_edges = list(chain(*edges))
            nodes_with_parent_or_child=list(set(flatten_edges))
            if nodes_with_parent_or_child!=[]:
                
                selected_node2=choice(nodes_with_parent_or_child)
                parents2=selected_node2.nodes_from

                selected_node1 = new_graph_first.get_nodes_by_name(str(selected_node2))[0]
                parents1=selected_node1.nodes_from
                

                if parents1:
                    for p in parents1:
                        new_graph_first.operator.disconnect_nodes(p, selected_node1, False)
                if parents2:
                    for p in parents2:
                        new_graph_second.operator.disconnect_nodes(p, selected_node2, False)

                old_edges1 = new_graph_first.operator.get_edges()
                old_edges2 = new_graph_second.operator.get_edges()

                if parents2!=[] and parents2!=None:
                    parents_in_first_graph=[new_graph_first.get_nodes_by_name(str(i))[0] for i in parents2]
                    for parent in parents_in_first_graph:
                        if [parent, selected_node1] not in old_edges1:
                            new_graph_first.operator.connect_nodes(parent, selected_node1)

                if parents1!=[] and parents1!=None:
                    parents_in_second_graph=[new_graph_second.get_nodes_by_name(str(i))[0] for i in parents1]
                    for parent in parents_in_second_graph:
                        if [parent, selected_node2] not in old_edges2:
                            new_graph_second.operator.connect_nodes(parent, selected_node2)            


            if old_edges1 != new_graph_first.operator.get_edges() or old_edges2 != new_graph_second.operator.get_edges():
                break    
        
        if old_edges1 == new_graph_first.operator.get_edges() and parents2!=[] and parents2!=None:
            new_graph_first = deepcopy(graph_first)                
        if old_edges2 == new_graph_second.operator.get_edges() and parents1!=[] and parents1!=None:
            new_graph_second = deepcopy(graph_second)       

    except Exception as ex:
        print(ex)    
    return new_graph_first, new_graph_second


def custom_crossover_exchange_parents_one(graph_first, graph_second, max_depth):
    

    num_cros = 100
    try:
        for _ in range(num_cros):

            old_edges1 = []
            new_graph_first=deepcopy(graph_first)

            edges = graph_second.operator.get_edges()
            flatten_edges = list(chain(*edges))
            nodes_with_parent_or_child=list(set(flatten_edges))
            if nodes_with_parent_or_child!=[]:
                
                selected_node=choice(nodes_with_parent_or_child)
                parents=selected_node.nodes_from
                
                node_from_first_graph=new_graph_first.get_nodes_by_name(str(selected_node))[0]
                
                node_from_first_graph.nodes_from=[]
                old_edges1 = new_graph_first.operator.get_edges()
                
                if parents!=[] and parents!=None:
                    parents_in_first_graph=[new_graph_first.get_nodes_by_name(str(i))[0] for i in parents]
                    for parent in parents_in_first_graph:
                        if [parent, node_from_first_graph] not in old_edges1:
                            new_graph_first.operator.connect_nodes(parent, node_from_first_graph)

            if old_edges1 != new_graph_first.operator.get_edges():
                break    
        
        if old_edges1 == new_graph_first.operator.get_edges() and parents!=[] and parents!=None:
            new_graph_first = deepcopy(graph_first)                

    except Exception as ex:
        print(ex)

    return new_graph_first, graph_second


def custom_mutation_add_structure(graph: BNModel, **kwargs):
    count = 100
    try:
        for _ in range(count):
            rid = choice(range(len(graph.nodes)))
            random_node = graph.nodes[rid]
            other_random_node = graph.nodes[choice(range(len(graph.nodes)))]
            nodes_not_cycling = (random_node.descriptive_id not in
                                 [n.descriptive_id for n in ordered_subnodes_hierarchy(other_random_node)] and
                                 other_random_node.descriptive_id not in
                                 [n.descriptive_id for n in ordered_subnodes_hierarchy(random_node)])
            if nodes_not_cycling:
                other_random_node.nodes_from.append(random_node)
                break

    except Exception as ex:
        graph.log.warn(f'Incorrect connection: {ex}')
    return graph
 

def custom_mutation_delete_structure(graph: BNModel, **kwargs):
    count = 100
    try:
        for _ in range(count):
            rid = choice(range(len(graph.nodes)))
            random_node = graph.nodes[rid]
            other_random_node = graph.nodes[choice(range(len(graph.nodes)))]
            if random_node.nodes_from is not None and other_random_node in random_node.nodes_from:
                random_node.nodes_from.remove(other_random_node)
                break
    except Exception as ex:
        print(ex) 
    return graph


def custom_mutation_reverse_structure(graph: BNModel, **kwargs):
    count = 100
    try:
        for _ in range(count):
            rid = choice(range(len(graph.nodes)))
            random_node = graph.nodes[rid]
            other_random_node = graph.nodes[choice(range(len(graph.nodes)))]
            if random_node.nodes_from is not None and other_random_node in random_node.nodes_from:
                random_node.nodes_from.remove(other_random_node)
                other_random_node.nodes_from.append(random_node)
                break         
    except Exception as ex:
        print(ex)  
    return graph


def custom_mutation_change_node(graph: BNModel, **kwargs):

    count = 100
    try:
        for _ in range(count):
            rid = choice(range(len(graph.nodes)))
            random_node = graph.nodes[rid]
            other_random_node = graph.nodes[choice(range(len(graph.nodes)))]
            random_node_parents = [n for n in random_node.nodes_from]
            other_random_node_parents = [n for n in other_random_node.nodes_from]
            random_node.nodes_from = other_random_node_parents
            other_random_node.nodes_from = random_node_parents
            break


    except Exception as ex:
        graph.log.warn(f'Incorrect connection: {ex}')
    return graph