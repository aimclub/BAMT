import Builders, Nodes
from pomegranate import DiscreteDistribution, ConditionalProbabilityTable
# from Utils import GraphUtils as gru
# import pickle
# from sklearn import linear_model
# import numpy as np
# import itertools
# import sys
import logging.config

from os import path

log_file_path = path.join(path.dirname(path.abspath(__file__)), 'logging.conf')

logging.config.fileConfig(log_file_path)
logger = logging.getLogger('network')


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
            logger.error(
                f"{self.type} BN does not support {'discrete' if self.type == 'Continuous' else 'continuous'} data")
            return 'Error occurred during validation. Check logs.'
        elif ['Abstract'] in self._allowed_dtypes:
            return None
        ### Stage 1 ###
        worker_1 = Builders.VerticesDefiner(descriptor)
        self.nodes = worker_1.vertices

    def add_edges(self, data,
                  scoring_function, params, optimizer='HC'):
        if not self.validate(descriptor=self.descriptor):
            logger.error(
                f"{self.type} BN does not support {'discrete' if self.type == 'Continuous' else 'continuous'} data")
            return
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

    def set_nodes(self, nodes=None, **kwargs):
        if nodes is None:
            nodes = dict()
        nodes.update(kwargs)
        for column_name, node in nodes.items():
            try:
                assert issubclass(node, Nodes.BaseNode)
            except AssertionError:
                logger.error(f"{node} is not an instance of {Nodes.BaseNode}")
                continue
            except TypeError:
                logger.error(f"Passed kwarg must be a class. Arg: {node}")
                continue

            self.nodes.append(node(name=column_name))
            self.update_descriptor()


class DiscreteBN(BaseNetwork):
    def __init__(self):
        super(DiscreteBN, self).__init__()
        self.type = 'Discrete'
        self.scoring_function = ""
        self._allowed_dtypes = ['disc', 'disc_num']
        self.has_logit = None
        self.use_mixture = None
        # self.distributions = {'probas_matrix': None}

    def fit_parameters(self, data):
        def worker(node):
            parents = node.disc_parents + node.cont_parents
            if len(parents) == 0:
                numoutcomes = int(len(data[node.name].unique()))
                dist = DiscreteDistribution.from_samples(data[node.name].values)
                vals = sorted([str(x) for x in list(dist.parameters[0].keys())])
                cprob = list(dict(sorted(dist.items())).values())
            if len(parents) != 0:
                numoutcomes = int(len(data[node.name].unique()))
                dist = DiscreteDistribution.from_samples(data[node.name].values)
                vals = sorted([str(x) for x in list(dist.parameters[0].keys())])
                dist = ConditionalProbabilityTable.from_samples(data[parents + [node.name]].values)
                params = dist.parameters[0]
                cprob = dict()
                for i in range(0, len(params), len(vals)):
                    probs = []
                    for j in range(i, (i + len(vals))):
                        probs.append(params[j][-1])
                    combination = [str(x) for x in params[i][0:len(parents)]]
                    cprob[str(combination)] = probs
            # TODO: точно ли нам нужен такой выход? в таблице только probas_matrix
            return {"numoutcomes": numoutcomes, "cprob": cprob, "vals": vals}

        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(3)
        for node in self.nodes:
            future = pool.submit(worker, node)
            self.distributions[node.name] = future.result()

    # def fit_parameters(self, data):
    #     for node in self.nodes:
    #         if len(node.parents) == 0:
    #             numoutcomes = int(len(data[node.name].unique()))
    #             dist = DiscreteDistribution.from_samples(data[node.name].values)
    #             vals = sorted([str(x) for x in list(dist.parameters[0].keys())])
    #             cprob = list(dict(sorted(dist.items())).values())
    #         if len(node.parents) != 0:
    #             numoutcomes = int(len(data[node.name].unique()))
    #             dist = DiscreteDistribution.from_samples(data[node.name].values)
    #             vals = sorted([str(x) for x in list(dist.parameters[0].keys())])
    #             dist = ConditionalProbabilityTable.from_samples(data[node.parents + [node.name]].values)
    #             params = dist.parameters[0]
    #             cprob = dict()
    #             for i in range(0, len(params), len(vals)):
    #                 probs = []
    #                 for j in range(i, (i + len(vals))):
    #                     probs.append(params[j][-1])
    #                 combination = [str(x) for x in params[i][0:len(node.parents)]]
    #                 cprob[str(combination)] = probs
    #         self.distributions[node.name] = {"numoutcomes": numoutcomes, "cprob": cprob, "parents": node.parents,
    #                                          "vals": vals, "type": "discrete", "children": node.children}


class ContinuousBN(BaseNetwork):
    def __init__(self, has_logit: bool = False, use_mixture: bool = False):
        super(ContinuousBN, self).__init__()
        self.type = 'Continuous'
        self._allowed_dtypes = ['cont']
        self.has_logit = has_logit
        self.use_mixture = use_mixture
        self.scoring_function = ""
        self.distributions = {'mean': 0.0, 'variance': 1.0, 'lr_coefs': []}

    def fit_parameters(self):
        pass


class HybridBN(BaseNetwork):
    def __init__(self, has_logit: bool = False, use_mixture: bool = False):
        super(HybridBN, self).__init__()
        self._allowed_dtypes = ['cont', 'disc', 'disc_num']
        self.type = 'Hybrid'
        self.has_logit = has_logit
        self.use_mixture = use_mixture

    # TODO: Overwrite validation
    # def validate(self, descriptor):
    #     types = descriptor['types']
    #     return True if  else False

