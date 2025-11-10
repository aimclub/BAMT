"""
Simple plotting functions for BAMT 2.0.0 Bayesian Networks
"""
from typing import List, Tuple, Optional

try:
    import matplotlib.pyplot as plt
    import networkx as nx
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False


def plot_structure(edges: List[Tuple[str, str]], 
                   title: str = "Bayesian Network Structure",
                   node_size: int = 3000,
                   node_color: str = 'lightblue',
                   font_size: int = 12,
                   figsize: Tuple[int, int] = (10, 8),
                   save_path: Optional[str] = None):
    """
    Plot the DAG structure of a Bayesian network.
    
    Args:
        edges: List of tuples (parent, child) representing edges
        title: Plot title
        node_size: Size of nodes in the plot
        node_color: Color of nodes
        font_size: Font size for node labels
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib figure object (or None if matplotlib not available)
    """
    if not PLOT_AVAILABLE:
        raise ImportError("matplotlib and networkx required for plotting")
    
    # Create directed graph
    G = nx.DiGraph()
    G.add_edges_from(edges)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use hierarchical layout if possible, otherwise spring layout
    try:
        pos = nx.spring_layout(G, k=2, iterations=50)
    except:
        pos = nx.spring_layout(G)
    
    # Draw the graph
    nx.draw(G, pos, 
            with_labels=True,
            node_color=node_color,
            node_size=node_size,
            font_size=font_size,
            font_weight='bold',
            arrows=True,
            arrowsize=20,
            edge_color='gray',
            arrowstyle='->',
            ax=ax)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_network_info(network, 
                      title: str = "Network Information",
                      figsize: Tuple[int, int] = (12, 6)):
    """
    Plot information about a fitted Bayesian network.
    
    Args:
        network: A fitted BayesianNetwork object (with edges and node_models)
        title: Plot title
        figsize: Figure size
        
    Returns:
        matplotlib figure object (or None if matplotlib not available)
    """
    if not PLOT_AVAILABLE:
        raise ImportError("matplotlib and networkx required for plotting")
    
    if not hasattr(network, 'edges') or not hasattr(network, 'nodes'):
        raise ValueError("Network must have 'edges' and 'nodes' attributes")
    
    fig = plt.figure(figsize=figsize)
    
    # Left subplot: Network structure
    ax1 = plt.subplot(1, 2, 1)
    G = nx.DiGraph()
    G.add_edges_from(network.edges)
    
    pos = nx.spring_layout(G, k=2, iterations=50)
    nx.draw(G, pos,
            with_labels=True,
            node_color='lightgreen',
            node_size=2000,
            font_size=10,
            font_weight='bold',
            arrows=True,
            arrowsize=15,
            edge_color='gray',
            ax=ax1)
    ax1.set_title("Network Structure", fontweight='bold')
    
    # Right subplot: Network statistics
    ax2 = plt.subplot(1, 2, 2)
    ax2.axis('off')
    
    # Gather statistics
    num_nodes = len(network.nodes)
    num_edges = len(network.edges)
    
    # Compute in-degree and out-degree
    in_degrees = {}
    out_degrees = {}
    for node in network.nodes:
        in_degrees[node] = 0
        out_degrees[node] = 0
    
    for parent, child in network.edges:
        out_degrees[parent] = out_degrees.get(parent, 0) + 1
        in_degrees[child] = in_degrees.get(child, 0) + 1
    
    # Find root and leaf nodes
    root_nodes = [n for n, deg in in_degrees.items() if deg == 0]
    leaf_nodes = [n for n, deg in out_degrees.items() if deg == 0]
    
    # Display statistics
    info_text = f"""
Network Statistics
{'='*30}

Total Nodes: {num_nodes}
Total Edges: {num_edges}

Root Nodes: {len(root_nodes)}
{', '.join(root_nodes[:5])}{'...' if len(root_nodes) > 5 else ''}

Leaf Nodes: {len(leaf_nodes)}
{', '.join(leaf_nodes[:5])}{'...' if len(leaf_nodes) > 5 else ''}

Avg In-Degree: {sum(in_degrees.values())/num_nodes:.2f}
Avg Out-Degree: {sum(out_degrees.values())/num_nodes:.2f}

Max In-Degree: {max(in_degrees.values())}
Max Out-Degree: {max(out_degrees.values())}
    """
    
    ax2.text(0.1, 0.5, info_text, 
             fontsize=10,
             verticalalignment='center',
             family='monospace')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig
