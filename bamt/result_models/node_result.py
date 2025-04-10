from bamt.result_models.results import Results
from scipy.stats import norm
import numpy as np
from gmr import GMM

class NodeDistribution(Results):
    def __init__(self):
        self.node_type = "abstract"

    def __repr__(self):
        return f'NodeDistribution({self.node_type})'

    def get(self):
        pass


class DiscreteNodeResult(NodeDistribution):
    def __init__(self, probs, values):
        super().__init__()
        self.node_type = 'discrete'
        self.probs = np.array(probs)
        self.values = values


    def get(self):
        return self.probs, self.values


class GaussianNodeResult(NodeDistribution):
    def __init__(self, distribution):
        super().__init__()
        self.node_type = 'gaussian'

        self.mean, self.std = distribution[0], distribution[1]

    def get(self, with_gaussian: bool = False):
        if with_gaussian:
            return norm(loc=self.mean, scale=self.std)
        else:
            return self.mean, self.std

class MixtureGaussianNodeResult(NodeDistribution):
    def __init__(self, distribution, n_components):
        super().__init__()
        self.node_type = 'mixture'
        self.n_components = n_components

        self.mean, self.covars, self.priors = distribution

    def get(self, with_gaussian: bool = False):
        if with_gaussian:
            return GMM(
                n_components=self.n_components,
                priors=self.priors,
                means=self.mean,
                covariances=self.covars
                )
        else:
            return self.mean, self.covars, self.priors

class ConditionalMixtureGaussianNodeResult(MixtureGaussianNodeResult):
    def __init__(self, distribution, n_components):
        super().__init__(distribution, n_components)


class ConditionalGaussianNodeResult(GaussianNodeResult):
    def __init__(self, distribution):
        super().__init__(distribution)
        self.node_type = 'conditional_gaussian'


class LogitNodeResult(NodeDistribution):
    def __init__(self, probs, values):
        super().__init__()

        self.node_type = 'logit'
        self.probs = np.array(probs)
        self.values = values

    def get(self):
        return self.probs, self.values

class ConditionalLogitNodeResult(LogitNodeResult):
    def __init__(self, probs, values):
        super().__init__(probs, values)
        self.node_type = 'conditional_logit'

