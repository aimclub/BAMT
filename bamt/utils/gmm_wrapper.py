import numpy as np
from sklearn.mixture import GaussianMixture

class GMM:
    def __init__(self, n_components = 1, priors = None, means = None, covariances = None, random_state = None):
        self.n_components = n_components
        self.random_state = random_state
        self._gmm = None  # объект GaussianMixture или ручная сборка

        if means is not None and covariances is not None and priors is not None:
            self._manual_init(means, covariances, priors)

