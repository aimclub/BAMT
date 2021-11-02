import Builders
from pomegranate import DiscreteDistribution, ConditionalProbabilityTable
import pickle


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
        self.nodes = []
        self.edges = []
        self.descriptor = None
        self.distributions = {}

    def add_nodes(self, descriptor):
        self.descriptor = descriptor
        worker = Builders.VerticesDefiner(descriptor)
        self.nodes = worker.vertices


class DiscreteBN(BaseNetwork):
    def __init__(self):
        super(DiscreteBN, self).__init__()
        self.scoring_function = ""

    def add_edges(self, data,
                  scoring_function, params, optimizer='HC'):
        if optimizer == 'HC':
            worker = Builders.HCStructureBuilder(data=data,
                                                 descriptor=self.descriptor,
                                                 scoring_function=scoring_function)
            self.sf_name = scoring_function[0]
            worker.build(data=data, **params)

            self.nodes = worker.skeleton['V']  # update family
            self.edges = worker.skeleton['E']

    def fit_parameters(self, data):
        def worker(node):
            if len(node.parents) == 0:
                numoutcomes = int(len(data[node.name].unique()))
                dist = DiscreteDistribution.from_samples(data[node.name].values)
                vals = sorted([str(x) for x in list(dist.parameters[0].keys())])
                cprob = list(dict(sorted(dist.items())).values())
            if len(node.parents) != 0:
                numoutcomes = int(len(data[node.name].unique()))
                dist = DiscreteDistribution.from_samples(data[node.name].values)
                vals = sorted([str(x) for x in list(dist.parameters[0].keys())])
                dist = ConditionalProbabilityTable.from_samples(data[node.parents + [node.name]].values)
                params = dist.parameters[0]
                cprob = dict()
                for i in range(0, len(params), len(vals)):
                    probs = []
                    for j in range(i, (i + len(vals))):
                        probs.append(params[j][-1])
                    combination = [str(x) for x in params[i][0:len(node.parents)]]
                    cprob[str(combination)] = probs
            return {"numoutcomes": numoutcomes, "cprob": cprob, "parents": node.parents,
                    "vals": vals, "type": "discrete", "children": node.children}

        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(3)
        for node in self.nodes:
            future = pool.submit(worker, node)
            self.distributions[node.name] = future.result()


class GaussianNB(BaseNetwork):
    pass
