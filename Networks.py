import Builders
import Nodes
# import numpy as np
# from Utils import GraphUtils as gru
# import pickle
from concurrent.futures import ThreadPoolExecutor
import random
from Utils import GraphUtils as gru
# from gmr import GMM
# import itertools
# import sys
import networkx as nx
from pyvis.network import Network
import matplotlib
import matplotlib.pyplot as plt
# from matplotlib import cm

from log import logger_network


class BaseNetwork(object):
    """
    Base class for Bayesian Network
    """

    def __init__(self):
        """
        Attributes:
            nodes: a list of nodes instances
            edges: a list of edges
            distributions: a dict with "numoutcomes", "cprob","parents","type", "children"
        """
        self.type = 'Abstract'
        self._allowed_dtypes = ['Abstract']
        self.nodes = []
        self.edges = []
        self.descriptor = {}
        self.distributions = {}
        self.has_logit = False
        self.use_mixture = False

    def validate(self, descriptor):
        types = descriptor['types']
        return True if all([a in self._allowed_dtypes for a in types.values()]) else False

    def update_descriptor(self):
        new_nodes_names = [node.name for node in self.nodes]
        self.descriptor['types'] = {node: type for node, type in self.descriptor['types'].items() if
                                    node in new_nodes_names}
        self.descriptor['signs'] = {node: sign for node, sign in self.descriptor['signs'].items() if
                                    node in new_nodes_names}

    def add_nodes(self, descriptor):
        self.descriptor = descriptor
        if not self.validate(descriptor=descriptor):
            if not self.type == 'Hybrid':
                logger_network.error(
                    f"{self.type} BN does not support {'discrete' if self.type == 'Continuous' else 'continuous'} data")
                return
            else:
                logger_network.error(
                    f"Hybrid BN is not supposed to work with only one type of data. Use DiscreteBN or Continuous BN instead.")
                return
        elif ['Abstract'] in self._allowed_dtypes:
            return None
        # Stage 1
        worker_1 = Builders.VerticesDefiner(descriptor)
        self.nodes = worker_1.vertices

    def add_edges(self, data,
                  scoring_function, params=None, optimizer='HC'):
        if not self.validate(descriptor=self.descriptor):
            logger_network.error(
                f"{self.type} BN does not support {'discrete' if self.type == 'Continuous' else 'continuous'} data")
            return
        if optimizer == 'HC':
            worker = Builders.HCStructureBuilder(data=data,
                                                 descriptor=self.descriptor,
                                                 scoring_function=scoring_function,
                                                 has_logit=self.has_logit,
                                                 use_mixture=self.use_mixture)
            self.sf_name = scoring_function[0]
            worker.build(data=data, params=params)

            self.nodes = worker.skeleton['V']  # update family
            self.edges = worker.skeleton['E']

    def set_nodes(self, nodes=None, **kwargs):
        if nodes is None:
            nodes = dict()
        nodes.update(kwargs)
        for column_name, node in nodes.items():
            try:
                assert issubclass(node, Nodes.BaseNode)
            except AssertionError:
                logger_network.error(f"{node} is not an instance of {Nodes.BaseNode}")
                continue
            except TypeError:
                logger_network.error(f"Passed kwarg must be a class. Arg: {node}")
                continue

            self.nodes.append(node(name=column_name))
            self.update_descriptor()

    def fit_parameters(self, data, dropna=True):
        if dropna:
            data = data.dropna()
            data.reset_index(inplace=True, drop=True)

        if 'disc_num' in self.descriptor['types'].values():
            columns_names = [name for name, t in self.descriptor['types'].items() if t in ['disc_num']]
            data[columns_names] = data.loc[:, columns_names].astype('str')

        # Topology sorting
        ordered = gru.toporder(self.nodes, self.edges)
        notOrdered = [node.name for node in self.nodes]
        mask = [notOrdered.index(name) for name in ordered]
        self.nodes = [self.nodes[i] for i in mask]

        def worker(node):
            return node.fit_parameters(data)

        pool = ThreadPoolExecutor(3)
        for node in self.nodes:
            future = pool.submit(worker, node)
            self.distributions[node.name] = future.result()

    def get_info(self):
        for n in self.nodes:
            print(
                f"{n.name: <20} | {n.type: <30} | {self.descriptor['types'][n.name]: <10} | {str([self.descriptor['types'][name] for name in n.cont_parents + n.disc_parents]): <50} | {str([name for name in n.cont_parents + n.disc_parents])}")

    def plot(self, output):
        from numpy import array
        G = nx.DiGraph()
        nodes = [node.name for node in self.nodes]
        G.add_nodes_from(nodes)
        G.add_edges_from(self.edges)

        network = Network(height="800px", width="100%", notebook=True, directed=nx.is_directed(G),
                          layout='hierarchical')

        nodes_sorted = array(list(nx.topological_generations(G)), dtype=object)

        # Qualitative class of colormaps
        q_classes = ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20',
                     'tab20b', 'tab20c']

        hex_colors = []
        for cls in q_classes:
            rgb_colors = plt.get_cmap(cls).colors
            hex_colors.extend(
                [matplotlib.colors.rgb2hex(rgb_color) for rgb_color in rgb_colors])

        hex_colors = array(hex_colors)

        # Number_of_colors in matplotlib in Qualitative class = 144

        class_number = len(
            set([node.type for node in self.nodes])
        )
        hex_colors_indexes = [random.randint(0, len(hex_colors)-1) for _ in range(class_number)]
        hex_colors_picked = hex_colors[hex_colors_indexes]
        class2color = {cls: color for cls, color in zip(set([node.type for node in self.nodes]), hex_colors_picked)}
        name2class = {node.name: node.type for node in self.nodes}

        for level in range(len(nodes_sorted)):
            for node_i in range(len(nodes_sorted[level])):
                name = nodes_sorted[level][node_i]
                cls = name2class[name]
                color = class2color[cls]
                network.add_node(name, label=name, color=color, size=45, level = level, font={'size': 36},
                                 title=f'Узел байесовской сети {name} ({cls})')

        for edge in G.edges:
            network.add_edge(edge[0], edge[1])

        network.hrepulsion(node_distance=300, central_gravity=0.5)

        import os
        if not (os.path.exists('../visualization_result')):
            os.mkdir("../visualization_result")

        return network.show(f'../visualization_result/' + output + '.html')


class DiscreteBN(BaseNetwork):
    def __init__(self):
        super(DiscreteBN, self).__init__()
        self.type = 'Discrete'
        self.scoring_function = ""
        self._allowed_dtypes = ['disc', 'disc_num']
        self.has_logit = None
        self.use_mixture = None

    def sample(self, n, evidence=None):
        seq = []
        random.seed()
        for _ in range(n):
            output = {}
            for node in self.nodes:
                parents = node.cont_parents + node.disc_parents
                if evidence:
                    if node.name in evidence.keys():
                        output[node.name] = evidence[node.name]
                if not parents:
                    pvals = None
                else:
                    pvals = [str(output[t]) for t in parents]
                output[node.name] = node.choose(self.distributions[node.name], pvals=pvals)
            seq.append(output)

        return seq


class ContinuousBN(BaseNetwork):
    def __init__(self, use_mixture: bool = False):
        super(ContinuousBN, self).__init__()
        self.type = 'Continuous'
        self._allowed_dtypes = ['cont']
        self.has_logit = None
        self.use_mixture = use_mixture
        self.scoring_function = ""

    # TODO: Обработка случая с неудачной топологической соритровкой
    def sample(self, n, evidence=None):
        seq = []
        random.seed()
        for _ in range(n):
            output = {}
            for node in self.nodes:
                parents = node.disc_parents + node.cont_parents
                if evidence:
                    if node.name in evidence.keys():
                        output[node.name] = evidence[node.name]

                if not parents:
                    pvalues = None
                else:
                    pvalues = [output[p] for p in parents]

                sample = node.choose(pvals=pvalues, node_info=self.distributions[node.name])
                output[node.name] = sample
            seq.append(output)
        return seq


class HybridBN(BaseNetwork):
    def __init__(self, has_logit: bool = False, use_mixture: bool = False):
        super(HybridBN, self).__init__()
        self._allowed_dtypes = ['cont', 'disc', 'disc_num']
        self.type = 'Hybrid'
        self.has_logit = has_logit
        self.use_mixture = use_mixture

    def validate(self, descriptor):
        types = descriptor['types']
        s = set(types.values())
        return True if ({'cont', 'disc', 'disc_num'} == s) or ({'cont', 'disc'} == s) else False

    def sample(self, n, evidence=None):
        seq = []
        random.seed()
        for _ in range(n):
            output = {}
            for node in self.nodes:
                parents = node.disc_parents + node.cont_parents
                if evidence:
                    if node.name in evidence.keys():
                        output[node.name] = evidence[node.name]
                if not parents:
                    pvalues = None
                else:
                    pvalues = [output[p] for p in parents]
                result = node.choose(self.distributions[node.name], pvals=pvalues)

                output[node.name] = result
            seq.append(output)
        return seq
