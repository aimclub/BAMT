"""
Visualization utilities for BAMT 2.0.0 Bayesian Networks.

Provides plotting functionality using pyvis for interactive HTML visualizations
and matplotlib for static images.
"""

from pathlib import Path
from typing import Union, Optional, Dict, Any
import networkx as nx


class BayesianNetworkVisualizer:
    """Visualizer for Bayesian Network structures."""

    @staticmethod
    def plot(
        bn: "BayesianNetwork",
        output_path: Union[str, Path],
        notebook: bool = False,
        show_node_info: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Visualize a Bayesian Network using pyvis.

        Args:
            bn: Bayesian Network instance
            output_path: Path to save the visualization (should end with .html)
            notebook: Whether to display in a Jupyter notebook
            show_node_info: Whether to show node type information in tooltips
            **kwargs: Additional arguments passed to pyvis.Network()

        Example:
            >>> bn.plot("my_network.html")
            >>> bn.plot("my_network.html", height="600px", width="80%")
        """
        try:
            from pyvis.network import Network
        except ImportError:
            raise ImportError(
                "pyvis is required for visualization. Install with: pip install pyvis"
            )

        output_path = Path(output_path)
        if not str(output_path).endswith(".html"):
            output_path = output_path.with_suffix(".html")

        # Create pyvis network
        network_params = {
            "height": kwargs.get("height", "800px"),
            "width": kwargs.get("width", "100%"),
            "notebook": notebook,
            "directed": True,
            "layout": kwargs.get("layout", "hierarchical"),
        }

        net = Network(**network_params)

        # Add nodes with type information
        for node_name in bn.structure.nodes():
            node_info = bn.nodes_dict.get(node_name)
            if node_info and show_node_info:
                node_type = type(node_info).__name__
                title = f"{node_name}\nType: {node_type}"
            else:
                title = node_name

            # Color nodes by type if available
            color = None
            if node_info:
                node_class = type(node_info).__name__
                if "Discrete" in node_class:
                    color = "#90EE90"  # Light green
                elif "Continuous" in node_class:
                    color = "#87CEEB"  # Sky blue
                elif "Conditional" in node_class:
                    color = "#FFB6C1"  # Light pink

            net.add_node(node_name, title=title, color=color)

        # Add edges
        for edge in bn.structure.edges():
            net.add_edge(edge[0], edge[1])

        # Set physics options for better layout
        net.set_options(
            """
        {
          "physics": {
            "hierarchicalRepulsion": {
              "centralGravity": 0.0,
              "springLength": 200,
              "springConstant": 0.01,
              "nodeDistance": 150,
              "damping": 0.09
            },
            "minVelocity": 0.75,
            "solver": "hierarchicalRepulsion"
          }
        }
        """
        )

        # Save
        net.save_graph(str(output_path))

    @staticmethod
    def plot_static(
        bn: "BayesianNetwork",
        output_path: Optional[Union[str, Path]] = None,
        figsize: tuple = (12, 8),
        node_size: int = 3000,
        font_size: int = 10,
        **kwargs: Any,
    ) -> None:
        """
        Create a static visualization using matplotlib and networkx.

        Args:
            bn: Bayesian Network instance
            output_path: Path to save the image (PNG/PDF/etc.). If None, displays the plot.
            figsize: Figure size (width, height)
            node_size: Size of nodes
            font_size: Font size for labels
            **kwargs: Additional arguments passed to nx.draw()

        Example:
            >>> bn.plot_static("my_network.png")
            >>> bn.plot_static("my_network.pdf", figsize=(15, 10))
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for static plots. Install with: pip install matplotlib"
            )

        fig, ax = plt.subplots(figsize=figsize)

        # Use hierarchical layout for DAG
        pos = nx.spring_layout(bn.structure, k=2, iterations=50)

        # Try hierarchical layout if available
        try:
            pos = nx.nx_agraph.graphviz_layout(bn.structure, prog="dot")
        except:
            # Fall back to spring layout
            pass

        # Draw the graph
        nx.draw(
            bn.structure,
            pos,
            with_labels=True,
            node_size=node_size,
            node_color="lightblue",
            font_size=font_size,
            font_weight="bold",
            edge_color="gray",
            arrows=True,
            arrowsize=20,
            ax=ax,
            **kwargs,
        )

        ax.set_title(
            f"{bn.__class__.__name__} Structure", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()


def plot_bn(
    bn: "BayesianNetwork",
    output_path: Union[str, Path],
    mode: str = "interactive",
    **kwargs: Any,
) -> None:
    """
    Convenience function to plot a Bayesian Network.

    Args:
        bn: Bayesian Network instance
        output_path: Path to save the visualization
        mode: Either "interactive" (HTML with pyvis) or "static" (image with matplotlib)
        **kwargs: Additional arguments passed to the plotting function

    Example:
        >>> plot_bn(bn, "network.html")  # Interactive HTML
        >>> plot_bn(bn, "network.png", mode="static")  # Static PNG
    """
    if mode == "interactive":
        BayesianNetworkVisualizer.plot(bn, output_path, **kwargs)
    elif mode == "static":
        BayesianNetworkVisualizer.plot_static(bn, output_path, **kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'interactive' or 'static'.")
