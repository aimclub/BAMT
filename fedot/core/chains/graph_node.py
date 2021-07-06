from copy import copy
from typing import Any, List, Optional
from itertools import groupby

from fedot.core.log import default_log


class GraphNode:
    def __init__(self, nodes_from: Optional[List['GraphNode']],
                 operation_type: Any,
                 log=None):
        self.nodes_from = nodes_from
        self.log = log
        self.operation = operation_type

        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

    @property
    def descriptive_id(self):
        return self._descriptive_id_recursive(visited_nodes=[])

    def _descriptive_id_recursive(self, visited_nodes):
        """
        Method returns verbal description of the operation in the node
        and its parameters
        """

        try:
            node_label = self.operation.description
        except AttributeError:
            node_label = self.operation

        full_path = ''
        if self in visited_nodes:
            return 'ID_CYCLED'
        visited_nodes.append(self)
        if self.nodes_from:
            previous_items = []
            for parent_node in self.nodes_from:
                previous_items.append(f'{parent_node._descriptive_id_recursive(copy(visited_nodes))};')
            previous_items.sort()
            previous_items_str = ';'.join(previous_items)

            full_path += f'({previous_items_str})'
        full_path += f'/{node_label}'
        return full_path

    def __str__(self):
        operation = f'{self.operation}'
        return operation

    def __repr__(self):
        return self.__str__()

    def ordered_subnodes_hierarchy(self, visited=None) -> List['GraphNode']:
        """if visited is None:
            visited = []
        nodes = [self]
        if self.nodes_from:
            for parent in self.nodes_from:
                if parent not in visited:
                    nodes.extend(parent.ordered_subnodes_hierarchy(visited))"""
        nodes = [self]
        def recursive_sort(parent_nodes: list):
            if parent_nodes:
                nonlocal nodes
                next = parent_nodes
                next_next = []
                for node in next:
                    if node.nodes_from:
                        next_next.extend(node.nodes_from)
                next_next = [el for el, _ in groupby(next_next)] 
                next = [node for node in next if node not in next_next]
                nodes.extend(next)
                recursive_sort(next_next)
        recursive_sort(self.nodes_from)
                    
        """if visited is None:
            visited = []
        if isinstance(self, list):
            nodes = self
        else:
            nodes = [self]
        for root in nodes:
            if root.nodes_from:
                for parent in root.nodes_from:
                    if parent not in visited:
                        nodes.extend(parent.ordered_subnodes_hierarchy(visited))"""
        if any([isinstance(node, list) for node in nodes]):
            print(nodes)
        return nodes


class PrimaryGraphNode(GraphNode):
    def __init__(self, operation_type: Any, **kwargs):
        super().__init__(nodes_from=None, operation_type=operation_type, **kwargs)


class SecondaryGraphNode(GraphNode):
    def __init__(self, operation_type: Any, nodes_from: Optional[List['GraphNode']] = None,
                 **kwargs):
        nodes_from = [] if nodes_from is None else nodes_from
        super().__init__(nodes_from=nodes_from, operation_type=operation_type,
                         **kwargs)

    def _nodes_from_with_fixed_order(self):
        if self.nodes_from is not None:
            return sorted(self.nodes_from, key=lambda node: node.descriptive_id)
