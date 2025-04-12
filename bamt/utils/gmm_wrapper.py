import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
import warnings

class GMM:
    def __init__(self, n_components = 1, priors = None, means = None, covariances = None, random_state = None):
        self.n_components = n_components
        self.random_state = random_state
        self._gmm = None  # объект GaussianMixture или ручная сборка

        if means is not None and covariances is not None and priors is not None:
            self._manual_init(means, covariances, priors)

    def _manual_init(self, means, covariances, priors):
        # Преобразуем всё в numpy-массивы
        means = np.array(means)
        covariances = np.array(covariances)
        priors = np.array(priors)

        # Инициализируем пустую GMM и вручную проставим параметры
        self._gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='full',
            random_state=self.random_state
        )

        self._gmm.weights_ = priors
        self._gmm.means_ = means
        self._gmm.covariances_ = covariances
        # Восстановим precision_cholesky вручную (обязательная часть в sklearn)
        self._gmm.precisions_cholesky_ = _compute_precision_cholesky(
            covariances, 'full'
        )


    @property
    def means(self):
        return self._gmm.means_

    @property
    def priors(self):
        return self._gmm.weights_

    @property
    def covariances(self):
        return self._gmm.covariances_

    def from_samples(self, X, n_iter=100, init_params="kmeans", sample_weight=None):

        if sample_weight is not None:
            warnings.warn(
                "sample_weight is not supported in the current sklearn build and will be ignored.",
                RuntimeWarning
            )

        self._gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='full',
            max_iter=n_iter,
            init_params=init_params,
            random_state=self.random_state
        )

        self._gmm.fit(X)
        return self

