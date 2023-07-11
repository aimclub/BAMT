import random

import pandas as pd

from pgmpy.estimators import K2Score
from pgmpy.models import BayesianNetwork
from golem.core.dag.convert import graph_structure_as_nx_graph
from golem.core.dag.graph_utils import ordered_subnodes_hierarchy
from golem.core.optimisers.graph import OptGraph, OptNode


class CustomGraphModel(OptGraph):
    def evaluate(self, data: pd.DataFrame):
        nodes = data.columns.to_list()
        _, labels = graph_structure_as_nx_graph(self)
        return len(nodes)


class CustomGraphNode(OptNode):
    def __str__(self):
        return f'{self.content["name"]}'


def K2_metric(graph: CustomGraphModel, data: pd.DataFrame):
    graph_nx, labels = graph_structure_as_nx_graph(graph)
    struct = []
    for meta_edge in graph_nx.edges():
        l1 = str(labels[meta_edge[0]])
        l2 = str(labels[meta_edge[1]])
        struct.append([l1, l2])

    bn_model = BayesianNetwork(struct)
    bn_model.add_nodes_from(data.columns)

    score = K2Score(data).score(bn_model)
    return -score


def custom_mutation_add(graph: CustomGraphModel, **kwargs):
    num_mut = 100
    try:
        for _ in range(num_mut):
            rid = random.choice(range(len(graph.nodes)))
            random_node = graph.nodes[rid]
            other_random_node = graph.nodes[random.choice(range(len(graph.nodes)))]
            nodes_not_cycling = random_node.descriptive_id not in [
                n.descriptive_id for n in ordered_subnodes_hierarchy(other_random_node)
            ] and other_random_node.descriptive_id not in [
                n.descriptive_id for n in ordered_subnodes_hierarchy(random_node)
            ]
            if nodes_not_cycling:
                random_node.nodes_from.append(other_random_node)
                break

    except Exception as ex:
        print(f"Incorrect connection: {ex}")
    return graph


def custom_mutation_delete(graph: OptGraph, **kwargs):
    num_mut = 100
    try:
        for _ in range(num_mut):
            rid = random.choice(range(len(graph.nodes)))
            random_node = graph.nodes[rid]
            other_random_node = graph.nodes[random.choice(range(len(graph.nodes)))]
            if (
                random_node.nodes_from is not None
                and other_random_node in random_node.nodes_from
            ):
                random_node.nodes_from.remove(other_random_node)
                break
    except Exception as ex:
        print(ex)
    return graph


def custom_mutation_reverse(graph: OptGraph, **kwargs):
    num_mut = 100
    try:
        for _ in range(num_mut):
            rid = random.choice(range(len(graph.nodes)))
            random_node = graph.nodes[rid]
            other_random_node = graph.nodes[random.choice(range(len(graph.nodes)))]
            if (
                random_node.nodes_from is not None
                and other_random_node in random_node.nodes_from
            ):
                random_node.nodes_from.remove(other_random_node)
                other_random_node.nodes_from.append(random_node)
                break
    except Exception as ex:
        print(ex)
    return graph


def has_no_duplicates(graph):
    _, labels = graph_structure_as_nx_graph(graph)
    if len(labels.values()) != len(set(labels.values())):
        raise ValueError("Custom graph has duplicates")
    return True


def has_no_blacklist_edges(graph, blacklist):
    nx_graph, _ = graph_structure_as_nx_graph(graph)
    for edge in nx_graph.edges():
        if edge in blacklist:
            raise ValueError("Graph contains blacklisted edges")
    return True


def has_only_whitelist_edges(graph, whitelist):
    nx_graph, _ = graph_structure_as_nx_graph(graph)
    for edge in nx_graph.edges():
        if edge not in whitelist:
            raise ValueError("Graph contains non-whitelisted edges")
    return True
