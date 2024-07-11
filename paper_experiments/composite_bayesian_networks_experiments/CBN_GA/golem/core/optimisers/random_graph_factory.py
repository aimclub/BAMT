from random import randint, choices
from typing import Optional, Callable

from golem.core.constants import MAX_GRAPH_GEN_ATTEMPTS
from golem.core.dag.graph import Graph
from golem.core.dag.graph_utils import distance_to_root_level
from golem.core.dag.graph_verifier import GraphVerifier
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.opt_node_factory import OptNodeFactory
from golem.core.optimisers.optimization_parameters import GraphRequirements

RandomGraphFactory = Callable[[GraphRequirements, int], Graph]


class RandomGrowthGraphFactory(RandomGraphFactory):
    """ Default realisation of random graph factory. Generates DAG graph using random growth. """
    def __init__(self,
                 verifier: GraphVerifier,
                 node_factory: OptNodeFactory):
        self.node_factory = node_factory
        self.verifier = verifier

    def __call__(self, requirements: GraphRequirements, max_depth: Optional[int] = None) -> OptGraph:
        return random_graph(self.verifier, self.node_factory, requirements, max_depth)


def random_graph(verifier: GraphVerifier,
                 node_factory: OptNodeFactory,
                 requirements: GraphRequirements,
                 max_depth: Optional[int] = None,
                 growth_proba: float = 0.3) -> OptGraph:
    max_depth = max_depth if max_depth else requirements.max_depth
    is_correct_graph = False
    graph = None
    n_iter = 0

    while not is_correct_graph:
        graph = OptGraph()
        graph_root = node_factory.get_node()
        graph.add_node(graph_root)
        if requirements.max_depth > 1:
            graph_growth(graph, graph_root, node_factory, requirements, max_depth, growth_proba)

        is_correct_graph = verifier(graph)
        n_iter += 1
        if n_iter > MAX_GRAPH_GEN_ATTEMPTS:
            raise ValueError(f'Could not generate random graph for {n_iter} '
                             f'iterations with requirements {requirements}')
    return graph


def graph_growth(graph: OptGraph,
                 node_parent: OptNode,
                 node_factory: OptNodeFactory,
                 requirements: GraphRequirements,
                 max_depth: int,
                 growth_proba: float):
    """Function create a graph and links between nodes"""
    offspring_size = randint(requirements.min_arity, requirements.max_arity)

    for offspring_node in range(offspring_size):
        node = node_factory.get_node()
        node_parent.nodes_from.append(node)
        graph.add_node(node)
        height = distance_to_root_level(graph, node)
        is_max_depth_exceeded = height >= max_depth - 1
        if not is_max_depth_exceeded:
            # lower proba of further growth reduces time of graph generation
            if choices([0, 1], weights=[1 - growth_proba, growth_proba])[0]:
                graph_growth(graph, node, node_factory, requirements, max_depth, growth_proba)
