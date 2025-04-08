from typing import List, Dict, Optional, Any, Set, Tuple
import networkx as nx
import uuid


class Node:
    """Base class for nodes in a directed graph."""

    def __init__(self, name: str, content: Optional[Dict] = None):
        self.name = name
        self.content = content or {}
        self.parents = []  # nodes that point to this node
        self.children = []  # nodes that this node points to
        self.uid = str(uuid.uuid4())[:8]

    def add_child(self, node: "Node") -> None:
        """Add a child node to this node."""
        if node not in self.children:
            self.children.append(node)
            if self not in node.parents:
                node.parents.append(self)

    def remove_child(self, node: "Node") -> None:
        """Remove a child node from this node."""
        if node in self.children:
            self.children.remove(node)
            if self in node.parents:
                node.parents.remove(self)

    def add_parent(self, node: "Node") -> None:
        """Add a parent node to this node."""
        if node not in self.parents:
            self.parents.append(node)
            if self not in node.children:
                node.children.append(self)

    def remove_parent(self, node: "Node") -> None:
        """Remove a parent node from this node."""
        if node in self.parents:
            self.parents.remove(node)
            if self in node.children:
                node.children.remove(self)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"Node({self.name})"


class Graph:
    """Directed graph representation."""

    def __init__(self):
        self.nodes = []
        self.log = None  # For compatibility with the previous implementation

    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        if node not in self.nodes:
            self.nodes.append(node)

    def remove_node(self, node: Node) -> None:
        """Remove a node from the graph."""
        if node in self.nodes:
            # Remove all connections to this node
            for parent in node.parents[
                :
            ]:  # Create a copy of the list to modify during iteration
                parent.remove_child(node)
            for child in node.children[
                :
            ]:  # Create a copy of the list to modify during iteration
                child.remove_parent(node)
            self.nodes.remove(node)

    def add_edge(self, parent: Node, child: Node) -> None:
        """Add a directed edge from parent to child."""
        parent.add_child(child)

    def remove_edge(self, parent: Node, child: Node) -> None:
        """Remove a directed edge from parent to child."""
        parent.remove_child(child)

    def get_edges(self) -> List[Tuple[str, str]]:
        """Get all edges in the graph as tuples of node names."""
        edges = []
        for node in self.nodes:
            for child in node.children:
                edges.append((node.name, child.name))
        return edges

    def get_nodes_by_name(self, name: str) -> List[Node]:
        """Get all nodes with the given name."""
        return [node for node in self.nodes if node.name == name]

    def to_networkx(self) -> nx.DiGraph:
        """Convert the graph to a NetworkX DiGraph."""
        G = nx.DiGraph()
        for node in self.nodes:
            G.add_node(node.name, content=node.content)
        for node in self.nodes:
            for child in node.children:
                G.add_edge(node.name, child.name)
        return G

    def has_cycle(self) -> bool:
        """Check if the graph has a cycle."""
        return not nx.is_directed_acyclic_graph(self.to_networkx())

    def copy(self) -> "Graph":
        """Create a deep copy of the graph."""
        new_graph = Graph()
        # Create new nodes with the same names
        name_to_node = {}
        for node in self.nodes:
            new_node = Node(
                node.name, content=node.content.copy() if node.content else None
            )
            name_to_node[node.name] = new_node
            new_graph.add_node(new_node)

        # Add edges
        for node in self.nodes:
            for child in node.children:
                new_graph.add_edge(name_to_node[node.name], name_to_node[child.name])

        return new_graph
