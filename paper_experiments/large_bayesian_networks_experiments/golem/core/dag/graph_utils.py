from typing import Sequence, List, TYPE_CHECKING, Callable, Union, Optional

from golem.utilities.data_structures import ensure_wrapped_in_sequence

if TYPE_CHECKING:
    from golem.core.dag.graph import Graph
    from golem.core.dag.graph_node import GraphNode


def distance_to_root_level(graph: 'Graph', node: 'GraphNode') -> int:
    """Gets distance to the final output node

    Args:
        graph: graph for finding the distance
        node: search starting point

    Return:
        int: distance to root level
    """

    def child_height(parent_node: 'GraphNode') -> int:
        height = 0
        for _ in range(graph.length):
            node_children = graph.node_children(parent_node)
            if node_children:
                height += 1
                parent_node = node_children[0]
            else:
                return height

    if graph_has_cycle(graph):
        return -1
    height = child_height(node)
    return height


def distance_to_primary_level(node: 'GraphNode') -> int:
    depth = node_depth(node)
    return depth - 1 if depth > 0 else -1


def nodes_from_layer(graph: 'Graph', layer_number: int) -> Sequence['GraphNode']:
    """Gets all the nodes from the chosen layer up to the surface

    Args:
        graph: graph with nodes
        layer_number: max height of diving

    Returns:
        all nodes from the surface to the ``layer_number`` layer
    """

    def get_nodes(roots: Sequence['GraphNode'], current_height: int) -> Sequence['GraphNode']:
        """Gets all the parent nodes of ``roots``

        :param roots: nodes to get all subnodes from
        :param current_height: current diving step depth

        :return: all parent nodes of ``roots`` in one sequence:69
        """
        nodes = []
        if current_height == layer_number:
            nodes.extend(roots)
        else:
            for root in roots:
                nodes.extend(get_nodes(root.nodes_from, current_height + 1))
        return nodes

    nodes = get_nodes(graph.root_nodes(), current_height=0)
    return nodes


def ordered_subnodes_hierarchy(node: 'GraphNode') -> List['GraphNode']:
    """Gets hierarchical subnodes representation of the graph starting from the bounded node

    Returns:
        List['GraphNode']: hierarchical subnodes list starting from the bounded node
    """
    started = {node}
    visited = set()

    def subtree_impl(node):
        nodes = [node]
        for parent in node.nodes_from:
            if parent in visited:
                continue
            elif parent in started:
                raise ValueError('Can not build ordered node hierarchy: graph has cycle')
            started.add(parent)
            nodes.extend(subtree_impl(parent))
            visited.add(parent)
        return nodes

    return subtree_impl(node)


def node_depth(nodes: Union['GraphNode', Sequence['GraphNode']]) -> int:
    """Gets the maximal depth among the provided ``nodes`` in the graph

    Args:
        nodes: nodes to calculate the depth for

    Returns:
        int: maximal depth
    """
    nodes = ensure_wrapped_in_sequence(nodes)
    final_depth = {}
    subnodes = set()
    for node in nodes:
        max_depth = 0
        # if node is a subnode of another node it has smaller depth
        if node.uid in subnodes:
            continue
        depth = 1
        visited = []
        if node in visited:
            return -1
        visited.append(node)
        stack = [(node, depth, iter(node.nodes_from))]
        while stack:
            curr_node, depth_now, parents = stack[-1]
            try:
                parent = next(parents)
                subnodes.add(parent.uid)
                if parent not in visited:
                    visited.append(parent)
                    if parent.uid in final_depth:
                        # depth of the parent has been already calculated
                        stack.append((parent, depth_now + final_depth[parent.uid], iter([])))
                    else:
                        stack.append((parent, depth_now + 1, iter(parent.nodes_from)))
                else:
                    return -1
            except StopIteration:
                _, depth_now, _ = stack.pop()
                visited.pop()
                max_depth = max(max_depth, depth_now)
        final_depth[node.uid] = max_depth
    return max(final_depth.values())


def map_dag_nodes(transform: Callable, nodes: Sequence) -> Sequence:
    """Maps nodes in dfs-order while respecting node edges.

    Args:
        transform: node transform function (maps node to node)
        nodes: sequence of nodes for mapping

    Returns:
        Sequence: sequence of transformed links with preserved relations
    """
    mapped_nodes = {}

    def map_impl(node):
        already_mapped = mapped_nodes.get(id(node))
        if already_mapped:
            return already_mapped
        # map node itself
        mapped_node = transform(node)
        # remember it to avoid recursion
        mapped_nodes[id(node)] = mapped_node
        # map its children
        mapped_node.nodes_from = list(map(map_impl, node.nodes_from))
        return mapped_node

    return list(map(map_impl, nodes))


def graph_structure(graph: 'Graph') -> str:
    """ Returns structural information about the graph - names and parameters of graph nodes.
    Represents graph info in easily readable way.

    Returns:
        str: graph structure
    """
    return '\n'.join([str(graph), *(f'{node.name} - {node.parameters}' for node in graph.nodes)])


def graph_has_cycle(graph: 'Graph') -> bool:
    """ Returns True if the graph contains a cycle and False otherwise. Implements Depth-First Search."""

    visited = {node.uid: False for node in graph.nodes}
    stack = []
    on_stack = {node.uid: False for node in graph.nodes}
    for node in graph.nodes:
        if visited[node.uid]:
            continue
        stack.append(node)
        while len(stack) > 0:
            cur_node = stack[-1]
            if not visited[cur_node.uid]:
                visited[cur_node.uid] = True
                on_stack[cur_node.uid] = True
            else:
                on_stack[cur_node.uid] = False
                stack.pop()
            for parent in cur_node.nodes_from:
                if not visited[parent.uid]:
                    stack.append(parent)
                elif on_stack[parent.uid]:
                    return True
    return False


def get_all_simple_paths(graph: 'Graph', source: 'GraphNode', target: 'GraphNode') \
        -> List[List[List['GraphNode']]]:
    """ Returns all simple paths from one node to another ignoring edge direction.
    Args:
        graph: graph in which to search for paths
        source: the first node of the path
        target: the last node of the path """
    paths = []
    nodes_children = {source.uid: graph.node_children(source)}
    target = {target}
    visited = dict.fromkeys([source])
    node_neighbors = set(source.nodes_from).union(nodes_children[source.uid])
    stack = [iter(node_neighbors)]

    while stack:
        neighbors = stack[-1]
        neighbor = next(neighbors, None)
        if neighbor is None:  # current path does not contain target
            stack.pop()
            visited.popitem()
        else:
            if neighbor in visited:  # path is not simple
                continue
            if neighbor in target:  # target node was reached
                path = list(visited) + [neighbor]
                pairs_list = [[path[i], path[(i + 1)]] for i in range(len(path) - 1)]
                paths.append(pairs_list)
            else:  # target node was not reached
                visited[neighbor] = None
                children = nodes_children[neighbor.uid] if neighbor.uid in nodes_children \
                    else nodes_children.setdefault(neighbor.uid, graph.node_children(neighbor))  # lazy setdefault
                node_neighbors = set(neighbor.nodes_from).union(children)
                stack.append(iter(node_neighbors))
    return paths


def get_connected_components(graph: 'Graph', nodes: Optional[List['GraphNode']]) -> List[set]:
    """ Returns list of connected components of the graph.
    Each connected component is represented as a set of its nodes.
    Args:
        graph: graph to divide into connected components
        nodes: if provided, only connected components containing these nodes are returned
    Returns:
        List of connected components"""
    def _bfs(graph: 'Graph', source: 'GraphNode'):
        seen = set()
        nextlevel = {source}
        while nextlevel:
            thislevel = nextlevel
            nextlevel = set()
            for v in thislevel:
                if v not in seen:
                    seen.add(v)
                    nextlevel.update(set(v.nodes_from).union(set(graph.node_children(v))))
        return seen
    visited = set()
    nodes = nodes or graph.nodes
    components = []
    for node in nodes:
        if node not in visited:
            c = _bfs(graph, node)
            visited.update(c)
            components.append(c)
    return components
