import random
import os

import networkx as nx
import numpy as np
from pyvis.network import Network

import matplotlib.pyplot as plt
import matplotlib

from bamt.log import logger_display

from typing import Dict, Optional, List


class Display(object):
    """Display object"""

    def __init__(self, output):
        """
        :param output: output file name
        """
        self.output = output
        self.class2color = Dict
        self.name2class = Dict

    @staticmethod
    def _validate_dir(output):
        """Format validation"""
        if not output.endswith(".html"):
            logger_display.error("This version allows only html format.")
            return None
        return True

    @staticmethod
    def _make_network(nodes, edges, **kwargs):
        """Make network and graph
        :param nodes: nodes
        :param edges: edges
        :param kwargs: a dict passed to pyvis.network.Network
        """
        g = nx.DiGraph()
        if not kwargs:
            network_params = dict(
                height="800px",
                width="100%",
                notebook=True,
                directed=nx.is_directed(g),
                layout="hierarchical",
            )
        else:
            network_params = kwargs

        g.add_nodes_from(nodes)
        g.add_edges_from(edges)

        network = Network(**network_params)
        return network, g

    def _init_colors(self, nodes):
        # Qualitative class of colormaps
        q_classes = [
            "Pastel1",
            "Pastel2",
            "Paired",
            "Accent",
            "Dark2",
            "Set1",
            "Set2",
            "Set3",
            "tab10",
            "tab20",
            "tab20b",
            "tab20c",
        ]

        hex_colors = []
        for cls in q_classes:
            rgb_colors = plt.get_cmap(cls).colors
            hex_colors.extend(
                [matplotlib.colors.rgb2hex(rgb_color) for rgb_color in rgb_colors]
            )

        class_number = len(set([node.type for node in nodes]))
        hex_colors_indexes = [
            random.randint(0, len(hex_colors) - 1) for _ in range(class_number)
        ]

        hex_colors = np.array(hex_colors)

        hex_colors_picked = hex_colors[hex_colors_indexes]
        class2color = {
            cls: color
            for cls, color in zip(set([node.type for node in nodes]), hex_colors_picked)
        }
        name2class = {node.name: node.type for node in nodes}
        self.class2color = class2color
        self.name2class = name2class

    def build(self, nodes, edges, **kwargs):
        """Build and show network

        :param nodes: nodes
        :param edges: edges
        :param kwargs: a dict passed to pyvis.network.Network.add_node
        """
        if not self._validate_dir(self.output):
            logger_display.error(f"Output error: {self.output}")
            return None

        if not kwargs:
            node_params = dict(font={"size": 36}, size=45)
        else:
            node_params = kwargs

        self._init_colors(nodes)

        nodes_names = [node.name for node in nodes]

        network, g = self._make_network(nodes_names, edges)
        nodes_sorted = np.array(list(nx.topological_generations(g)), dtype=object)

        for level in range(len(nodes_sorted)):
            for node_i in range(len(nodes_sorted[level])):
                name = nodes_sorted[level][node_i]
                cls = self.name2class[name]
                color = self.class2color[cls]
                network.add_node(
                    name,
                    label=name,
                    color=color,
                    level=level,
                    title=f"{name} ({cls})",
                    **node_params,
                )

        for edge in g.edges:
            network.add_edge(edge[0], edge[1])

        network.hrepulsion(node_distance=300, central_gravity=0.5)

        if not (os.path.exists("visualization_result")):
            os.mkdir("visualization_result")

        return network.show(f"visualization_result/" + self.output)


class GraphAnalyzer(object):
    def __init__(self, bn):
        self.bn = bn

    def _isolate_structure(self, nodes):
        isolated_edges = []
        for edge in self.bn.edges:
            if edge[0] in nodes and edge[1] in nodes:
                isolated_edges.append(edge)
        return isolated_edges

    def markov_blanket(self, node_name: str, plot_to: Optional[str] = None):
        if not self.bn.nodes:
            logger_display.error("Empty nodes")
            return

        node = self.bn[node_name]

        parents = node.cont_parents + node.disc_parents
        children = node.children
        fremd_eltern = []

        for child in node.children:
            all_parents = self.bn[child].cont_parents + self.bn[child].disc_parents

            if all_parents == [node_name]:
                continue
            else:
                new = all_parents
            fremd_eltern.extend(new)

        nodes = parents + children + fremd_eltern

        edges = self._isolate_structure(nodes)
        if plot_to:
            Display(plot_to).build([self.bn[node] for node in nodes], edges)
        return {"nodes": nodes, "edges": edges}

    def _collect_height(self, node_name, height):
        nodes = []
        node = self.bn[node_name]
        if height <= 0:
            return []

        if height == 1:
            return node.disc_parents + node.cont_parents

        for parent in node.cont_parents + node.disc_parents:
            nodes.append(parent)
            nodes.extend(self._collect_height(parent, height=height - 1))
        return nodes

    def _collect_depth(self, node_name, depth):
        nodes = []
        node = self.bn[node_name]

        if depth <= 0:
            return []

        if depth == 1:
            return node.children

        for child in node.children:
            nodes.append(child)
            nodes.extend(self._collect_depth(child, depth=depth - 1))

        return nodes

    def find_family(
        self,
        node_name: str,
        with_nodes: Optional[List] = None,
        height: int = 1,
        depth: int = 1,
        plot_to: Optional[str] = None,
    ):
        if with_nodes is None:
            with_nodes = []
        nodes = (
            self._collect_depth(node_name, depth)
            + self._collect_height(node_name, height)
            + [node_name]
        )
        nodes = list(set(nodes))
        if plot_to:
            Display(plot_to).build(
                [self.bn[node] for node in nodes + with_nodes],
                self._isolate_structure(nodes + with_nodes),
            )
        return nodes
