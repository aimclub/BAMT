from golem.core.log import default_log
from golem.core.dag.graph import Graph
from golem.structural_analysis.graph_sa.results.utils import get_entity_from_str


def nodes_deletion(graph: Graph, worst_result: dict) -> Graph:
    """ Extracts the node index from the entity key and removes it from the graph """

    node_to_delete = graph.nodes[int(worst_result["entity_idx"])]

    graph.delete_node(node_to_delete)
    default_log('NodeDeletion').message(f'{node_to_delete.name} was deleted')

    return graph


def nodes_replacement(graph: Graph, worst_result: dict) -> Graph:
    """ Extracts the node index and the operation to which it needs to be replaced from the entity key
    and replaces the node with a new one """

    # get the node that will be replaced
    node_to_replace = get_entity_from_str(graph=graph, entity_str=worst_result["entity_idx"])
    # get node to replace to
    new_node = graph.nodes[0].__class__(worst_result["entity_to_replace_to"])
    # new_node = graph.nodes[int(worst_result["entity_to_replace_to"])]
    new_node.nodes_from = []

    graph.update_node(old_node=node_to_replace, new_node=new_node)

    default_log('NodeReplacement').message(f'{node_to_replace.name} was replaced with {new_node.name}')

    return graph


def subtree_deletion(graph: Graph, worst_result: dict) -> Graph:
    """ Extracts the node index from the entity key and removes its subtree from the graph """

    node_to_delete = get_entity_from_str(graph=graph, entity_str=worst_result["entity_idx"])
    graph.delete_subtree(node_to_delete)
    default_log('SubtreeDeletion').message(f'{node_to_delete.name} subtree was deleted')

    return graph


def edges_deletion(graph: Graph, worst_result: dict) -> Graph:
    """ Extracts the edge's nodes indices from the entity key and removes edge from the graph """
    parent_node, child_node = get_entity_from_str(graph=graph, entity_str=worst_result["entity_idx"])

    graph.disconnect_nodes(parent_node, child_node)
    default_log('EdgeDeletion').message(f'Edge from {parent_node.name} to {child_node.name} was deleted')

    return graph


def edges_replacement(graph: Graph, worst_result: dict) -> Graph:
    """ Extracts the edge's nodes indices and the new edge to which it needs to be replaced from the entity key
    and replaces the edge with a new one """

    # get the edge that will be replaced
    parent_node, child_node = get_entity_from_str(graph=graph, entity_str=worst_result["entity_idx"])

    # get an edge to replace
    next_parent_node, next_child_node = \
        get_entity_from_str(graph=graph, entity_str=worst_result["entity_to_replace_to"])
    graph.connect_nodes(next_parent_node, next_child_node)

    graph.disconnect_nodes(parent_node, child_node, clean_up_leftovers=False)

    default_log('EdgeReplacement').message(f'Edge from {parent_node.name} to {child_node.name} was replaced with '
                                           f'edge from {next_parent_node.name} to {next_child_node.name}')

    return graph
