import Builders, Nodes
# from Utils import GraphUtils as gru
# import pickle
from concurrent.futures import ThreadPoolExecutor
import random
from Utils import GraphUtils as gru
# import itertools
# import sys

# from log import logger_network


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
        # if not self.validate(descriptor=descriptor):
        #     if not self.type == 'Hybrid':
        #         logger_network.error(
        #             f"{self.type} BN does not support {'discrete' if self.type == 'Continuous' else 'continuous'} data")
        #         return
        #     else:
        #         logger_network.error(
        #             f"Hybrid BN is not supposed to work with only one type of data. Use DiscreteBN or Continuous BN instead.")
        #         return
        # elif ['Abstract'] in self._allowed_dtypes:
        #     return None
        # Stage 1
        worker_1 = Builders.VerticesDefiner(descriptor)
        self.nodes = worker_1.vertices

    def add_edges(self, data,
                  scoring_function, params, optimizer='HC'):
        # if not self.validate(descriptor=self.descriptor):
        #     logger_network.error(
        #         f"{self.type} BN does not support {'discrete' if self.type == 'Continuous' else 'continuous'} data")
        #     return
        if optimizer == 'HC':
            worker = Builders.HCStructureBuilder(data=data,
                                                 descriptor=self.descriptor,
                                                 scoring_function=scoring_function,
                                                 has_logit=self.has_logit,
                                                 use_mixture=self.use_mixture)
            self.sf_name = scoring_function[0]
            worker.build(data=data, **params)

            self.nodes = worker.skeleton['V']  # update family
            self.edges = worker.skeleton['E']

    # def set_nodes(self, nodes=None, **kwargs):
        # if nodes is None:
        #     nodes = dict()
        # nodes.update(kwargs)
        # for column_name, node in nodes.items():
        #     try:
        #         assert issubclass(node, Nodes.BaseNode)
            # except AssertionError:
            #     logger_network.error(f"{node} is not an instance of {Nodes.BaseNode}")
            #     continue
            # except TypeError:
            #     logger_network.error(f"Passed kwarg must be a class. Arg: {node}")
            #     continue

            # self.nodes.append(node(name=column_name))
            # self.update_descriptor()

    def fit_parameters(self, data, dropna=True):
        if dropna:
            data = data.dropna()
            data.reset_index(inplace=True, drop=True)

        # Topology sorting
        ordered = gru.toporder(self.edges)
        notOrdered = [node.name for node in self.nodes]
        mask = [notOrdered.index(name) for name in ordered]
        self.nodes = [self.nodes[i] for i in mask]

        def worker(node):
            return node.fit_parameters(data)
        pool = ThreadPoolExecutor(3)
        for node in self.nodes:
            future = pool.submit(worker, node)
            self.distributions[node.name] = future.result()


class DiscreteBN(BaseNetwork):
    def __init__(self):
        super(DiscreteBN, self).__init__()
        self.type = 'Discrete'
        self.scoring_function = ""
        self._allowed_dtypes = ['disc', 'disc_num']
        self.has_logit = None
        self.use_mixture = None

    def sample(self, n, evidence=None):
        output = {}
        seq = []
        random.seed()
        for _ in range(n):
            for node in self.nodes:
                parents = node.cont_parents + node.disc_parents
                if evidence:
                    if node.name in evidence.keys():
                        output[node.name] = evidence[node.name]
                if not parents:
                    output[node.name] = node.choose(self.distributions[node.name])
                else:
                    pvals = [str(output[t]) for t in parents]
                    output[node.name] = node.choose(self.distributions[node.name], pvals=pvals)
            seq.append(output)

        return seq



class ContinuousBN(BaseNetwork):
    def __init__(self, has_logit: bool = False, use_mixture: bool = False):
        super(ContinuousBN, self).__init__()
        self.type = 'Continuous'
        self._allowed_dtypes = ['cont']
        self.has_logit = has_logit
        self.use_mixture = use_mixture
        self.scoring_function = ""
        # self.distributions = {'mean': 0.0, 'variance': 1.0, 'lr_coefs': []}


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
        return True if {'cont', 'disc', 'cont'} == s else False
