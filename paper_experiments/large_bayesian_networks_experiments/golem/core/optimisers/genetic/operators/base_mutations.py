from copy import deepcopy
from functools import partial
from random import choice, randint, random, sample, shuffle
from typing import TYPE_CHECKING, Optional

import numpy as np

from golem.core.adapter import register_native
from golem.core.dag.graph import ReconnectType
from golem.core.dag.graph_node import GraphNode
from golem.core.dag.graph_utils import distance_to_root_level, distance_to_primary_level, graph_has_cycle
from golem.core.optimisers.advisor import RemoveType
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.opt_node_factory import OptNodeFactory
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams, AlgorithmParameters
from golem.utilities.data_structures import ComparableEnum as Enum

if TYPE_CHECKING:
    from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters


class MutationStrengthEnum(Enum):
    weak = 0.2
    mean = 1.0
    strong = 5.0


class MutationTypesEnum(Enum):
    simple = 'simple'
    growth = 'growth'
    local_growth = 'local_growth'
    tree_growth = 'tree_growth'
    reduce = 'reduce'
    single_add = 'single_add'
    single_change = 'single_change'
    single_drop = 'single_drop'
    single_edge = 'single_edge'

    none = 'none'

    @property
    def __name__(self):
        return self.name


def get_mutation_prob(mut_id: MutationStrengthEnum, node: Optional[GraphNode],
                      default_mutation_prob: float = 0.7) -> float:
    """ Function returns mutation probability for certain node in the graph

    :param mut_id: MutationStrengthEnum mean weak or strong mutation
    :param node: root node of the graph
    :param default_mutation_prob: mutation probability used when mutation_id is invalid or graph has cycles
    :return mutation_prob: mutation probability
    """
    mutation_prob = default_mutation_prob
    graph_cycled = node is None
    if node:
        graph_cycled = distance_to_primary_level(node) < 0
    if mut_id in list(MutationStrengthEnum) and not graph_cycled:
        mutation_strength = mut_id.value
        mutation_prob = mutation_strength / (distance_to_primary_level(node) + 1)
    return mutation_prob


@register_native
def simple_mutation(graph: OptGraph,
                    requirements: GraphRequirements,
                    graph_gen_params: GraphGenerationParams,
                    parameters: 'GPAlgorithmParameters'
                    ) -> OptGraph:
    """
    This type of mutation is passed over all nodes of the tree started from the root node and changes
    nodes’ operations with probability - 'node mutation probability'
    which is initialised inside the function

    :param graph: graph to mutate
    """
    exchange_node = graph_gen_params.node_factory.exchange_node
    visited_nodes = set()

    def replace_node_to_random_recursive(node: OptNode) -> OptGraph:
        if node not in visited_nodes and random() < node_mutation_probability:
            new_node = exchange_node(node)
            if new_node:
                graph.update_node(node, new_node)
            # removed node must not be visited because it's outdated
            visited_nodes.add(node)
            # new node must not mutated if encountered further during traverse
            visited_nodes.add(new_node)
            for parent in node.nodes_from:
                replace_node_to_random_recursive(parent)

    root_nodes = graph.root_nodes()
    root_node = choice(root_nodes) if root_nodes else None
    node_mutation_probability = get_mutation_prob(mut_id=parameters.mutation_strength,
                                                  node=root_node)

    root_node = root_node or choice(graph.nodes)
    replace_node_to_random_recursive(root_node)

    return graph


@register_native
def single_edge_mutation(graph: OptGraph,
                         requirements: GraphRequirements,
                         graph_gen_params: GraphGenerationParams,
                         parameters: 'GPAlgorithmParameters'
                         ) -> OptGraph:
    """
    This mutation adds new edge between two random nodes in graph.

    :param graph: graph to mutate
    """

    def nodes_not_cycling(source_node: OptNode, target_node: OptNode):
        parents = source_node.nodes_from
        while parents:
            if target_node not in parents:
                grandparents = []
                for parent in parents:
                    grandparents.extend(parent.nodes_from)
                parents = grandparents
            else:
                return False
        return True

    for _ in range(parameters.max_num_of_operator_attempts):
        if len(graph.nodes) < 2 or graph.depth > requirements.max_depth:
            return graph

        source_node, target_node = sample(graph.nodes, 2)
        if source_node not in target_node.nodes_from:
            if graph_has_cycle(graph):
                graph.connect_nodes(source_node, target_node)
                break
            else:
                if nodes_not_cycling(source_node, target_node):
                    graph.connect_nodes(source_node, target_node)
                    break
    return graph


@register_native
def add_intermediate_node(graph: OptGraph,
                          node_factory: OptNodeFactory) -> OptGraph:
    nodes_with_parents = [node for node in graph.nodes if node.nodes_from]
    if len(nodes_with_parents) > 0:
        shuffle(nodes_with_parents)
        for node_to_mutate in nodes_with_parents:
            # add between node and parent
            new_node = node_factory.get_parent_node(node_to_mutate, is_primary=False)
            if not new_node:
                continue

            # rewire old children to new parent
            new_node.nodes_from = node_to_mutate.nodes_from
            node_to_mutate.nodes_from = [new_node]

            # add new node to graph
            graph.add_node(new_node)
            break
    return graph


@register_native
def add_separate_parent_node(graph: OptGraph,
                             node_factory: OptNodeFactory) -> OptGraph:
    node_idx = np.arange(len(graph.nodes))
    shuffle(node_idx)
    for idx in node_idx:
        node_to_mutate = graph.nodes[idx]
        # add as separate parent
        new_node = node_factory.get_parent_node(node_to_mutate, is_primary=True)
        if not new_node:
            # there is no possible operators
            continue
        if node_to_mutate.nodes_from:
            node_to_mutate.nodes_from.append(new_node)
        else:
            node_to_mutate.nodes_from = [new_node]
        graph.nodes.append(new_node)
        break
    return graph


@register_native
def add_as_child(graph: OptGraph,
                 node_factory: OptNodeFactory) -> OptGraph:
    node_idx = np.arange(len(graph.nodes))
    shuffle(node_idx)
    for idx in node_idx:
        node_to_mutate = graph.nodes[idx]
        # add as child
        old_node_children = graph.node_children(node_to_mutate)
        new_node_child = choice(old_node_children) if old_node_children else None
        new_node = node_factory.get_node(is_primary=False)
        if not new_node:
            continue
        graph.add_node(new_node)
        graph.connect_nodes(node_parent=node_to_mutate, node_child=new_node)
        if new_node_child:
            graph.connect_nodes(node_parent=new_node, node_child=new_node_child)
            graph.disconnect_nodes(node_parent=node_to_mutate, node_child=new_node_child,
                                   clean_up_leftovers=True)
        break
    return graph


@register_native
def single_add_mutation(graph: OptGraph,
                        requirements: GraphRequirements,
                        graph_gen_params: GraphGenerationParams,
                        parameters: AlgorithmParameters
                        ) -> OptGraph:
    """
    Add new node between two sequential existing modes

    :param graph: graph to mutate
    """
    if graph.depth >= requirements.max_depth:
        # add mutation is not possible
        return graph

    new_graph = deepcopy(graph)
    single_add_strategies = [add_as_child, add_separate_parent_node, add_intermediate_node]
    shuffle(single_add_strategies)
    for strategy in single_add_strategies:
        new_graph = strategy(new_graph, graph_gen_params.node_factory)
        # maximum three equality check
        if new_graph == graph:
            continue
        break
    return new_graph


@register_native
def single_change_mutation(graph: OptGraph,
                           requirements: GraphRequirements,
                           graph_gen_params: GraphGenerationParams,
                           parameters: AlgorithmParameters
                           ) -> OptGraph:
    """
    Change node between two sequential existing modes.

    :param graph: graph to mutate
    """
    node_idx = np.arange(len(graph.nodes))
    shuffle(node_idx)
    for idx in node_idx:
        node = graph.nodes[idx]
        new_node = graph_gen_params.node_factory.exchange_node(node)
        if not new_node:
            continue
        graph.update_node(node, new_node)
        break
    return graph


@register_native
def single_drop_mutation(graph: OptGraph,
                         requirements: GraphRequirements,
                         graph_gen_params: GraphGenerationParams,
                         parameters: AlgorithmParameters
                         ) -> OptGraph:
    """
    Drop single node from graph.

    :param graph: graph to mutate
    """
    if len(graph.nodes) < 2:
        return graph
    node_to_del = choice(graph.nodes)
    node_name = node_to_del.name
    removal_type = graph_gen_params.advisor.can_be_removed(node_to_del)
    if removal_type == RemoveType.with_direct_children:
        # TODO refactor workaround with data_source
        graph.delete_node(node_to_del)
        nodes_to_delete = \
            [n for n in graph.nodes
             if n.descriptive_id.count('data_source') == 1 and node_name in n.descriptive_id]
        for child_node in nodes_to_delete:
            graph.delete_node(child_node, reconnect=ReconnectType.all)
    elif removal_type == RemoveType.with_parents:
        graph.delete_subtree(node_to_del)
    elif removal_type == RemoveType.node_rewire:
        graph.delete_node(node_to_del, reconnect=ReconnectType.all)
    elif removal_type == RemoveType.node_only:
        graph.delete_node(node_to_del, reconnect=ReconnectType.none)
    elif removal_type == RemoveType.forbidden:
        pass
    else:
        raise ValueError("Unknown advice (RemoveType) returned by Advisor ")
    return graph


@register_native
def tree_growth(graph: OptGraph,
                requirements: GraphRequirements,
                graph_gen_params: GraphGenerationParams,
                parameters: AlgorithmParameters,
                local_growth: bool = True) -> OptGraph:
    """
    This mutation selects a random node in a tree, generates new subtree,
    and replaces the selected node's subtree.

    :param graph: graph to mutate
    :param local_growth: if true then maximal depth of new subtree equals depth of tree located in
    selected random node, if false then previous depth of selected node doesn't affect to
    new subtree depth, maximal depth of new subtree just should satisfy depth constraint in parent tree
    """
    node_idx = np.arange(len(graph.nodes))
    shuffle(node_idx)
    for idx in node_idx:
        node_from_graph = graph.nodes[idx]
        if local_growth:
            max_depth = distance_to_primary_level(node_from_graph)
            is_primary_node_selected = (not node_from_graph.nodes_from) or (node_from_graph != graph.root_node and
                                                                            randint(0, 1))
        else:
            max_depth = requirements.max_depth - distance_to_root_level(graph, node_from_graph)
            is_primary_node_selected = \
                distance_to_root_level(graph, node_from_graph) >= requirements.max_depth and randint(0, 1)
        if is_primary_node_selected:
            new_subtree = graph_gen_params.node_factory.get_node(is_primary=True)
            if not new_subtree:
                continue
        else:
            new_subtree = graph_gen_params.random_graph_factory(requirements, max_depth).root_node

        graph.update_subtree(node_from_graph, new_subtree)
        break
    return graph


@register_native
def growth_mutation(graph: OptGraph,
                    requirements: GraphRequirements,
                    graph_gen_params: GraphGenerationParams,
                    parameters: AlgorithmParameters,
                    local_growth: bool = True
                    ) -> OptGraph:
    """
    This mutation adds new nodes to the graph (just single node between existing nodes or new subtree).

    :param graph: graph to mutate
    :param local_growth: if true then maximal depth of new subtree equals depth of tree located in
    selected random node, if false then previous depth of selected node doesn't affect to
    new subtree depth, maximal depth of new subtree just should satisfy depth constraint in parent tree
    """

    if random() > 0.5:
        # simple growth (one node can be added)
        return single_add_mutation(graph, requirements, graph_gen_params, parameters)
    else:
        # advanced growth (several nodes can be added)
        return tree_growth(graph, requirements, graph_gen_params, parameters, local_growth)


@register_native
def reduce_mutation(graph: OptGraph,
                    requirements: GraphRequirements,
                    graph_gen_params: GraphGenerationParams,
                    parameters: AlgorithmParameters,
                    ) -> OptGraph:
    """
    Selects a random node in a tree, then removes its subtree. If the current arity of the node's
    parent is more than the specified minimal arity, then the selected node is also removed.
    Otherwise, it is replaced by a random primary node.

    :param graph: graph to mutate
    """
    if len(graph.nodes) == 1:
        return graph

    nodes = [node for node in graph.nodes if node is not graph.root_node]
    shuffle(nodes)
    for node_to_del in nodes:
        children = graph.node_children(node_to_del)
        is_possible_to_delete = all([len(child.nodes_from) - 1 >= requirements.min_arity for child in children])
        if is_possible_to_delete:
            graph.delete_subtree(node_to_del)
        else:
            primary_node = graph_gen_params.node_factory.get_node(is_primary=True)
            if not primary_node:
                continue
            graph.update_subtree(node_to_del, primary_node)
        break
    return graph


@register_native
def no_mutation(graph: OptGraph, *args, **kwargs) -> OptGraph:
    return graph


base_mutations_repo = {
    MutationTypesEnum.none: no_mutation,
    MutationTypesEnum.simple: simple_mutation,
    MutationTypesEnum.growth: partial(growth_mutation, local_growth=False),
    MutationTypesEnum.local_growth: partial(growth_mutation, local_growth=True),
    MutationTypesEnum.tree_growth: tree_growth,
    MutationTypesEnum.reduce: reduce_mutation,
    MutationTypesEnum.single_add: single_add_mutation,
    MutationTypesEnum.single_edge: single_edge_mutation,
    MutationTypesEnum.single_drop: single_drop_mutation,
    MutationTypesEnum.single_change: single_change_mutation
}

simple_mutation_set = (
    MutationTypesEnum.tree_growth,
    MutationTypesEnum.single_add,
    MutationTypesEnum.single_change,
    MutationTypesEnum.single_drop,
    MutationTypesEnum.single_edge,
    # join nodes
    # flip edge
    # cycle edge
)

rich_mutation_set = (
    MutationTypesEnum.simple,
    MutationTypesEnum.reduce,
    MutationTypesEnum.growth,
    MutationTypesEnum.local_growth
)
