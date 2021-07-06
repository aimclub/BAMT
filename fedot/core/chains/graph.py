from copy import deepcopy
from typing import List, Optional, Union

from fedot.core.chains.graph_node import GraphNode, PrimaryGraphNode, SecondaryGraphNode
from fedot.core.composer.visualisation import ChainVisualiser
from fedot.core.log import Log, default_log
from itertools import groupby

ERROR_PREFIX = 'Invalid chain configuration:'


class GraphObject:
    """
    Base class used for composite model structure definition

    :param nodes: GraphNode object(s)
    :param log: Log object to record messages

    .. note::
        fitted_on_data stores the data which were used in last chain fitting (equals None if chain hasn't been
        fitted yet)
    """

    primary_cls = PrimaryGraphNode
    secondary_cls = PrimaryGraphNode

    def __init__(self, nodes: Optional[Union[GraphNode, List[GraphNode]]] = None,
                 log: Log = None):
        self.nodes = []
        self.log = log
        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

        if nodes:
            if isinstance(nodes, list):
                for node in nodes:
                    self.add_node(node)
            else:
                self.add_node(nodes)

    def add_node(self, new_node: GraphNode):
        """
        Add new node to the Chain

        :param new_node: new Node object
        """
        if new_node not in self.nodes:
            self.nodes.append(new_node)
            if new_node.nodes_from:
                for new_parent_node in new_node.nodes_from:
                    if new_parent_node not in self.nodes:
                        self.add_node(new_parent_node)

    def _actualise_old_node_childs(self, old_node: GraphNode, new_node: GraphNode):
        old_node_offspring = self.node_childs(old_node)
        arr = deepcopy([old_node_child.nodes_from for old_node_child in old_node_offspring])
        for old_node_child in old_node_offspring:
            """if len(set(old_node_child.nodes_from)) != len(old_node_child.nodes_from):
                print('Duplicates in _actualise_old_node_childs')"""
            if old_node in old_node_child.nodes_from:
                old_node_child.nodes_from[old_node_child.nodes_from.index(old_node)] = new_node
            elif new_node not in old_node_child.nodes_from:
                print('Houston, we have a problem in _actualise_old_node_childs')
                print(old_node, new_node)
                print(*arr)
            

    def replace_node_with_parents(self, old_node: GraphNode, new_node: GraphNode):
        """Exchange subtrees with old and new nodes as roots of subtrees"""
        new_node = deepcopy(new_node)
        self._actualise_old_node_childs(old_node, new_node)
        self.delete_subtree(old_node)
        self.add_node(new_node)
        self._sort_nodes()

    def update_node(self, old_node: GraphNode, new_node: GraphNode):
        self._actualise_old_node_childs(old_node, new_node)
        new_node.nodes_from = old_node.nodes_from
        if old_node in self.nodes:
            self.nodes.remove(old_node)
        self.nodes.append(new_node)
        self._sort_nodes()
        """if type(new_node) is not type(old_node):
            raise ValueError(f"Can't update {old_node.__class__.__name__} "
                             f"with {new_node.__class__.__name__}")

            self._actualise_old_node_childs(old_node, new_node)
            new_node.nodes_from = old_node.nodes_from
            self.nodes.remove(old_node)
            self.nodes.append(new_node)
            self._sort_nodes()"""
        

    def delete_subtree(self, subtree_root_node: GraphNode):
        """Delete node with all the parents it has"""
        """for node_child in self.node_childs(subtree_root_node):
            node_child.nodes_from.remove(subtree_root_node)"""
        for subtree_node in subtree_root_node.ordered_subnodes_hierarchy():
            for node_child in self.node_childs(subtree_node):
                node_child.nodes_from.remove(subtree_node)
            if subtree_node in self.nodes:
                self.nodes.remove(subtree_node)

    def delete_node(self, node: GraphNode):
        """ This method redirects edges of parents to
        all the childs old node had.
        PNode    PNode              PNode    PNode
            \  /                      |  \   / |
            SNode <- delete this      |   \/   |
            / \                       |   /\   |
        SNode   SNode               SNode   SNode
        """

        def make_secondary_node_as_primary(node_child):
            extracted_type = node_child.operation
            new_primary_node = self.__class__.primary_cls(extracted_type)
            this_node_children = self.node_childs(node_child)
            for node in this_node_children:
                index = node.nodes_from.index(node_child)
                node.nodes_from.remove(node_child)
                node.nodes_from.insert(index, new_primary_node)

        node_children_cached = self.node_childs(node)
        self_root_node_cached = self.root_node

        for node_child in self.node_childs(node):
            node_child.nodes_from.remove(node)

        if isinstance(node, SecondaryGraphNode) and len(node.nodes_from) > 1 \
                and len(node_children_cached) > 1:

            for child in node_children_cached:
                for node_from in node.nodes_from:
                    child.nodes_from.append(node_from)

        else:
            if isinstance(node, SecondaryGraphNode):
                for node_from in node.nodes_from:
                    node_children_cached[0].nodes_from.append(node_from)
            elif isinstance(node, PrimaryGraphNode):
                for node_child in node_children_cached:
                    if not node_child.nodes_from:
                        make_secondary_node_as_primary(node_child)
        self.nodes.clear()
        self.add_node(self_root_node_cached)

    def node_childs(self, node) -> List[Optional['GraphNode']]:
        return [other_node for other_node in self.nodes if isinstance(other_node, SecondaryGraphNode) and
                node in other_node.nodes_from]

    def _is_node_has_child(self, node) -> bool:
        return any(self.node_childs(node))

    def _sort_nodes(self):
        """layer by layer sorting"""
        nodes = self.root_node
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
        next = []
        for node in nodes:
            if node.nodes_from:
                    next.extend(node.nodes_from)
        next = [el for el, _ in groupby(next)]                     
        recursive_sort(next)
        self.nodes = nodes

        """roots = self.root_node
        if isinstance(roots, list):
            nodes = roots
        else:
            nodes = [roots]
        for root in nodes:
            nodes.extend(root.ordered_subnodes_hierarchy())
        
        self.nodes = [el for el, _ in groupby(nodes)]"""

    def show(self, path: str = None):
        ChainVisualiser().visualise(self, path)

    def __eq__(self, other) -> bool:
        return bool(set([root.descriptive_id for root in self.root_node]) - set([root.descriptive_id for root in other.root_node]))
        #return self.root_node.descriptive_id == other.root_node.descriptive_id

    def __str__(self):
        description = {
            'depth': self.depth,
            'length': self.length,
            'nodes': self.nodes,
        }
        return f'{description}'

    def __repr__(self):
        return self.__str__()

    @property
    def root_node(self) -> List[Optional[GraphNode]]:
        if len(self.nodes) == 0:
            return None
        root = [node for node in self.nodes
                if not self._is_node_has_child(node)]
        """if len(root) > 1:
            raise ValueError(f'{ERROR_PREFIX} More than 1 root_nodes in chain')"""
        #return root[0]
        return root

    @property
    def length(self) -> int:
        return len(self.nodes)

    @property
    def depth(self) -> int:
        def _depth_recursive(node):
            if node is None:
                return 0
            if isinstance(node, PrimaryGraphNode):
                return 1
            elif isinstance(node, list):
                return max([_depth_recursive(next_root) for next_root in node])
            elif not node.nodes_from:
                return 1
            else:
                return 1 + max([_depth_recursive(next_node) for next_node in node.nodes_from])

        return _depth_recursive(self.root_node)
