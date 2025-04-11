import numpy as np
from sklearn.mixture import GaussianMixture

class GMM:
    def __init__(self, n_components=1, priors=None, means=None, covariances=None, random_state=None):
        self.n_components = n_components
        self._gmm = None
        self.random_state = random_state

        if means is not None and covariances is not None and priors is not None:
            self._manual_init(means, covariances, priors)

    def _manual_init(self, means, covariances, priors):
        self._gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='full',
            random_state=self.random_state
        )
        self._gmm.weights_ = np.array(priors)
        self._gmm.means_ = np.array(means)
        self._gmm.covariances_ = np.array(covariances)
        self._gmm.precisions_cholesky_ = GaussianMixture(
            n_components=self.n_components,
            covariance_type='full'
        )._compute_precision_cholesky(self._gmm.covariances_, 'full')

    def from_samples(self, X, n_iter=100, init_params="kmeans++"):
        self._gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='full',
            max_iter=n_iter,
            init_params=init_params,
            random_state=self.random_state
        )
        self._gmm.fit(X)
        return self

    @property
    def means(self):
        return self._gmm.means_

    @property
    def covariances(self):
        return self._gmm.covariances_

    @property
    def priors(self):
        return self._gmm.weights_

    def sample(self, n_samples):
        return self._gmm.sample(n_samples)

    def predict(self, X):
        return self._gmm.predict(X)

    def predict_proba(self, X):
        return self._gmm.predict_proba(X)

    def condition(self, given_indices, given_values):
        raise NotImplementedError("condition() not implemented yet")
