import random
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pandas import DataFrame
from pyvis.network import Network

# from bamt.log import logger_display


class Display(object):
    """Display object"""

    def __init__(self, output):
        """
        :param output: output file name
        """
        self.output = output
        self.class2color = Dict
        self.name2class = Dict

    # @staticmethod
    # def _validate_dir(output):
    #     """Format validation"""
    #     if not output.endswith(".html"):
    #         logger_display.error("This version allows only html format.")
    #         return None
    #     return True

    @staticmethod
    def _make_network(nodes, edges, **kwargs):
        """Make network and graph
        :param nodes: nodes
        :param edges: edges
        :param kwargs: a dict passed to pyvis.network.Network
        """
        g = nx.DiGraph()
        network_params = dict(
            height=kwargs.get("height", "800px"),
            width=kwargs.get("width", "100%"),
            notebook=kwargs.get("notebook", True),
            directed=kwargs.get("directed", nx.is_directed(g)),
            layout=kwargs.get("layout", "hierarchical"),
        )

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
        # if not self._validate_dir(self.output):
        #     logger_display.error(f"Output error: {self.output}")
        #     return None

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

        return network.show(self.output)

    @staticmethod
    def get_info(bn, as_df):
        if as_df:
            names = []
            types_n = []
            types_d = []
            parents = []
            parents_types = []
            for n in bn.nodes:
                names.append(n)
                types_n.append(n.type)
                types_d.append(bn.descriptor["types"][n.name])
                parents_types.append(
                    [
                        bn.descriptor["types"][name]
                        for name in n.cont_parents + n.disc_parents
                    ]
                )
                parents.append([name for name in n.cont_parents + n.disc_parents])
            return DataFrame(
                {
                    "name": names,
                    "node_type": types_n,
                    "data_type": types_d,
                    "parents": parents,
                    "parents_types": parents_types,
                }
            )
        else:
            for n in bn.nodes:
                print(
                    f"{n.name: <20} | "
                    f"{n.type: <50} | "
                    f"{bn.descriptor['types'][n.name]: <10} | "
                    f"{str([bn.descriptor['types'][name] for name in n.cont_parents + n.disc_parents]): <50} | "
                    f"{str([name for name in n.cont_parents + n.disc_parents])}"
                )
