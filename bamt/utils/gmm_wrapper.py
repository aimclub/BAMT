import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
import warnings
from scipy.stats import multivariate_normal
from typing import Optional, Union, List


class GMM:
    """
    A wrapper class around sklearn's GaussianMixture to emulate GMR's interface
    and allow conditional sampling and manual initialization.
    """

    def __init__(
        self,
        n_components: int = 1,
        priors: Optional[Union[List[float], np.ndarray]] = None,
        means: Optional[Union[List[List[float]], np.ndarray]] = None,
        covariances: Optional[Union[List[np.ndarray], np.ndarray]] = None,
        random_state: Optional[int] = None,
    ):
        """Initialize the GMM wrapper."""
        self.n_components = n_components
        self.random_state = random_state
        self._gmm = None

        if means is not None and covariances is not None and priors is not None:
            self._manual_init(means, covariances, priors)

    def _manual_init(
        self,
        means: Union[List, np.ndarray],
        covariances: Union[List, np.ndarray],
        priors: Union[List, np.ndarray],
    ):
        """Manually initialize the GMM with given parameters."""
        means = np.array(means)
        covariances = np.array(covariances)
        priors = np.array(priors)

        self._gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type="full",
            random_state=self.random_state,
        )
        self._gmm.weights_ = priors
        self._gmm.means_ = means
        self._gmm.covariances_ = covariances
        self._gmm.precisions_cholesky_ = _compute_precision_cholesky(
            covariances, "full"
        )

    @property
    def means(self) -> np.ndarray:
        """Return the mean vectors of each component."""
        if self._gmm is None:
            raise RuntimeError("GMM is not initialized.")
        return self._gmm.means_

    @property
    def priors(self) -> np.ndarray:
        """Return the prior weights of each component."""
        if self._gmm is None:
            raise RuntimeError("GMM is not initialized.")
        return self._gmm.weights_

    @property
    def covariances(self) -> np.ndarray:
        """Return the covariance matrices of each component."""
        if self._gmm is None:
            raise RuntimeError("GMM is not initialized.")
        return self._gmm.covariances_

    def from_samples(
        self,
        X: np.ndarray,
        n_iter: int = 100,
        init_params: str = "kmeans",
        sample_weight: Optional[np.ndarray] = None,
    ) -> "GMM":
        """Fit GMM to data."""
        if sample_weight is not None:
            warnings.warn(
                "sample_weight is not supported in this sklearn version and will be ignored.",
                RuntimeWarning,
            )

        _init_param_aliases = {
            "kmeans++": "k-means++",
            "k-means++": "k-means++",
            "kmeans": "kmeans",
            "random": "random",
            "random_from_data": "random_from_data",
        }

        if init_params not in _init_param_aliases:
            raise ValueError(
                f"Unsupported init_params '{init_params}'. Allowed: {list(_init_param_aliases.keys())}"
            )

        translated_init_param = _init_param_aliases[init_params]

        self._gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type="full",
            max_iter=n_iter,
            init_params=translated_init_param,
            random_state=self.random_state,
        )

        if X.shape[0] < 2:
            mean = X[0]
            cov = np.eye(X.shape[1]) * 1e-3
            self._gmm = GaussianMixture(n_components=1)
            self._gmm.weights_ = np.array([1.0])
            self._gmm.means_ = np.array([mean])
            self._gmm.covariances_ = np.array([cov])
            self._gmm.precisions_cholesky_ = _compute_precision_cholesky(
                self._gmm.covariances_, "full"
            )
            return self

        self._gmm.fit(X)
        return self

    def sample(self, n_samples: int) -> np.ndarray:
        """Generate samples from the GMM."""
        if self._gmm is None:
            raise RuntimeError("GMM is not initialized.")

        w = self._gmm.weights_
        if np.any(np.isnan(w)) or np.any(w < 0) or not np.isclose(w.sum(), 1.0):
            warnings.warn(
                f"Sampling skipped: invalid GMM weights detected (w = {w}). Returning NaNs.",
                RuntimeWarning,
            )
            return np.full((n_samples, self._gmm.means_.shape[1]), np.nan)

        if n_samples == 0:
            n_features = self._gmm.means_.shape[1]
            return np.empty((0, n_features))

        samples, _ = self._gmm.sample(n_samples)
        return samples

    def predict(self, X: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
        """Predict the most probable component for each sample."""
        if self._gmm is None:
            raise RuntimeError("GMM is not initialized.")

        X = np.array(X)
        if X.shape[0] == 0:
            return np.empty((0,), dtype=int)
        return self._gmm.predict(X)

    def predict_conditioned(
        self, given_indices: List[int], given_values: List[List[float]]
    ) -> np.ndarray:
        """Predict the component assignment based on conditioned GMM."""
        cond_gmm = self.condition(given_indices, given_values)
        dummy_input = np.zeros((len(given_values), len(cond_gmm.means[0])))
        return cond_gmm.predict(dummy_input)

    def to_responsibilities(
        self, X: Union[np.ndarray, List[List[float]]]
    ) -> np.ndarray:
        """Return the posterior probabilities (responsibilities) for each component."""
        if self._gmm is None:
            raise RuntimeError("GMM is not initialized.")

        X = np.array(X)
        if X.shape[0] == 0:
            return np.empty((0, self.n_components))

        return self._gmm.predict_proba(X)

    def condition(
        self, given_indices: List[int], given_values: List[List[float]]
    ) -> "GMM":
        """Compute the conditional GMM given a subset of observed variables."""
        if self._gmm is None:
            raise RuntimeError("GMM is not initialized.")

        given_values = np.array(given_values)[0]
        total_dim = self._gmm.means_.shape[1]
        free_indices = [i for i in range(total_dim) if i not in given_indices]

        new_means = []
        new_covariances = []
        new_priors = []

        for k in range(self.n_components):
            mean = self._gmm.means_[k]
            cov = self._gmm.covariances_[k]

            mu_given = mean[given_indices]
            mu_free = mean[free_indices]

            cov_gg = cov[np.ix_(given_indices, given_indices)]
            cov_gf = cov[np.ix_(given_indices, free_indices)]
            cov_fg = cov[np.ix_(free_indices, given_indices)]
            cov_ff = cov[np.ix_(free_indices, free_indices)]

            delta = given_values - mu_given
            cond_mean = mu_free + cov_fg @ np.linalg.solve(cov_gg, delta)
            cond_cov = cov_ff - cov_fg @ np.linalg.solve(cov_gg, cov_gf)

            new_means.append(cond_mean)
            new_covariances.append(cond_cov)

            prior = self._gmm.weights_[k]
            pdf_val = multivariate_normal.pdf(given_values, mean=mu_given, cov=cov_gg)
            new_priors.append(prior * pdf_val)

        new_priors = np.array(new_priors)
        new_priors /= new_priors.sum()

        return GMM(
            n_components=self.n_components,
            priors=new_priors.tolist(),
            means=[m.tolist() for m in new_means],
            covariances=[c.tolist() for c in new_covariances],
        )
