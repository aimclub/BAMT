import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
import warnings
from scipy.stats import multivariate_normal

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
        _init_param_aliases = {
            "kmeans++": "k-means++",  # gmr-style → sklearn
            "k-means++": "k-means++",
            "kmeans": "kmeans",
            "random": "random",
            "random_from_data": "random_from_data"
        }
        if init_params not in _init_param_aliases:
            raise ValueError(
                f"Unsupported init_params '{init_params}'. "
                f"Allowed values: {list(_init_param_aliases.keys())}"
            )
        translated_init_param = _init_param_aliases[init_params]



        self._gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='full',
            max_iter=n_iter,
            init_params=translated_init_param,
            random_state=self.random_state
        )
        if X.shape[0] < 2:
            # ручная инициализация модели
            mean = X[0]
            cov = np.eye(X.shape[1]) * 1e-3  # маленькая ковариация
            self._gmm = GaussianMixture(n_components=1)
            self._gmm.weights_ = np.array([1.0])
            self._gmm.means_ = np.array([mean])
            self._gmm.covariances_ = np.array([cov])
            self._gmm.precisions_cholesky_ = _compute_precision_cholesky(self._gmm.covariances_, 'full')
            return self

        self._gmm.fit(X)
        return self

    def sample(self, n_samples):

        if self._gmm is None:
            raise RuntimeError("GMM is not initialized. Call from_samples(...) or use manual init first.")
        if n_samples == 0:
            n_features = self._gmm.means_.shape[1]
            return np.empty((0, n_features))
        samples, _ = self._gmm.sample(n_samples)
        return samples

    def predict(self, X):

        if self._gmm is None:
            raise RuntimeError("GMM is not initialized. Call from_samples(...) or use manual init first.")

        X = np.array(X)
        if X.shape[0] == 0:
            return np.empty((0,), dtype=int)

        return self._gmm.predict(X)
    def predict_conditioned(self, given_indices, given_values):
        cond_gmm = self.condition(given_indices, given_values)
        dummy_input = np.zeros((len(given_values), len(cond_gmm.means[0])))
        return cond_gmm.predict(dummy_input)

    def to_responsibilities(self, X):
        if self._gmm is None:
            raise RuntimeError("GMM is not initialized.")

        X = np.array(X)
        if X.shape[0] == 0:
            return np.empty((0, self.n_components))

        return self._gmm.predict_proba(X)

    def condition(self, given_indices, given_values) -> 'GMM':

        if self._gmm is None:
            raise RuntimeError("GMM is not initialized.")

        given_values = np.array(given_values)[0]
        total_dim = self._gmm.means_.shape[1]

        # Индексы переменных, которые НЕ заданы
        free_indices = [i for i in range(total_dim) if i not in given_indices]

        # Новые списки параметров
        new_means = []
        new_covariances = []
        new_priors = []

        for k in range(self.n_components):
            mean = self._gmm.means_[k]
            cov = self._gmm.covariances_[k]

            # Блоки матрицы
            mu_given = mean[given_indices]
            mu_free = mean[free_indices]

            cov_gg = cov[np.ix_(given_indices, given_indices)]
            cov_gf = cov[np.ix_(given_indices, free_indices)]
            cov_fg = cov[np.ix_(free_indices, given_indices)]
            cov_ff = cov[np.ix_(free_indices, free_indices)]

            # Условное среднее
            delta = given_values - mu_given
            cond_mean = mu_free + cov_fg @ np.linalg.solve(cov_gg, delta)

            # Условная ковариация
            cond_cov = cov_ff - cov_fg @ np.linalg.solve(cov_gg, cov_gf)

            new_means.append(cond_mean)
            new_covariances.append(cond_cov)

            # Обновим вес компоненты
            prior = self._gmm.weights_[k]
            pdf_val = multivariate_normal.pdf(given_values, mean=mu_given, cov=cov_gg)
            new_priors.append(prior * pdf_val)

        # Нормализация весов
        new_priors = np.array(new_priors)
        new_priors /= new_priors.sum()

        # Вернуть новый GMM
        return GMM(
            n_components=self.n_components,
            priors=new_priors.tolist(),
            means=[m.tolist() for m in new_means],
            covariances=[c.tolist() for c in new_covariances]
        )