"""
Mixture Gaussian Distribution model for BAMT 2.0.0 architecture.

This module provides Gaussian Mixture Model (GMM) distribution support
for continuous nodes that may have multimodal distributions.
"""

from typing import Optional, List, Dict, Any

import numpy as np
from gmr import GMM

from .distribution import Distribution
from bamt.utils.math_utils import component


class MixtureGaussianDistribution(Distribution):
    """
    Gaussian Mixture Model distribution for continuous variables.

    This distribution uses GMM to model continuous variables that may have
    multiple modes. The number of components is automatically selected using
    AIC/BIC criteria.

    Example Usage:
        >>> import numpy as np
        >>> data = np.concatenate([np.random.normal(-2, 0.5, 500),
        ...                       np.random.normal(2, 0.5, 500)])
        >>> dist = MixtureGaussianDistribution()
        >>> dist.fit(data.reshape(-1, 1))
        >>> samples = dist.sample(100)
        >>> prediction = dist.predict(np.array([[0.5]]))
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        component_selection_method: str = "aic_bic_average",
        max_components: int = 10,
        **gmm_kwargs,
    ):
        """
        Initialize Mixture Gaussian Distribution.

        Args:
            n_components: Fixed number of components (if None, auto-select)
            component_selection_method: Method for selecting components
                - 'aic': Use AIC criterion
                - 'bic': Use BIC criterion
                - 'aic_bic_average': Average of AIC and BIC (default)
                - 'LRTS': Likelihood Ratio Test Statistic
            max_components: Maximum number of components to consider
            **gmm_kwargs: Additional arguments for GMM
        """
        self.n_components = n_components
        self.component_selection_method = component_selection_method
        self.max_components = max_components
        self.gmm_kwargs = gmm_kwargs

        # Fitted parameters
        self._gmm: Optional[GMM] = None
        self._means: Optional[List[List[float]]] = None
        self._covariances: Optional[List[List[List[float]]]] = None
        self._weights: Optional[List[float]] = None
        self._fitted_n_components: Optional[int] = None

    def fit(self, X: np.ndarray, parent_values: Optional[np.ndarray] = None) -> None:
        """
        Fit the GMM to the data.

        Args:
            X: Data to fit (n_samples, n_features) - typically (n_samples, 1) for univariate
            parent_values: Optional parent values for conditional distribution
                          (n_samples, n_parent_features)

        Note:
            If parent_values are provided, fits a joint GMM over [X, parent_values]
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Combine with parent values if provided
        if parent_values is not None:
            if parent_values.ndim == 1:
                parent_values = parent_values.reshape(-1, 1)
            data = np.hstack([X, parent_values])
        else:
            data = X

        # Determine number of components
        if self.n_components is None:
            n_comp = self._select_components(data)
        else:
            n_comp = self.n_components

        self._fitted_n_components = n_comp

        # Fit GMM
        self._gmm = GMM(n_components=n_comp, **self.gmm_kwargs)
        self._gmm.from_samples(data, n_iter=500, init_params="kmeans++")

        # Store parameters
        self._means = self._gmm.means.tolist()
        self._covariances = self._gmm.covariances.tolist()
        self._weights = self._gmm.priors.tolist()

    def _select_components(self, data: np.ndarray) -> int:
        """
        Select optimal number of components using specified method.

        Args:
            data: Input data

        Returns:
            Optimal number of components
        """
        import pandas as pd

        # Convert to DataFrame for component() function
        df = pd.DataFrame(data)
        columns = list(df.columns)

        if self.component_selection_method == "aic_bic_average":
            n_aic = component(df, columns, "aic")
            n_bic = component(df, columns, "bic")
            return int((n_aic + n_bic) / 2)
        elif self.component_selection_method in ["aic", "bic", "LRTS"]:
            return component(df, columns, self.component_selection_method)
        else:
            raise ValueError(
                f"Unknown component selection method: {self.component_selection_method}"
            )

    def sample(self, n_samples: int = 1, parent_values: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Sample from the fitted distribution.

        Args:
            n_samples: Number of samples to generate
            parent_values: Optional parent values for conditional sampling
                          Shape: (n_samples, n_parent_features)

        Returns:
            Samples from the distribution (n_samples, 1)
        """
        if self._gmm is None:
            raise ValueError("Distribution must be fitted before sampling")

        if parent_values is not None:
            # Conditional sampling
            if parent_values.ndim == 1:
                parent_values = parent_values.reshape(-1, 1)

            samples = []
            for pval in parent_values:
                # Condition on parent values
                cond_gmm = self._condition_on_parents(pval)
                sample = cond_gmm.sample(1)[0][0]
                samples.append(sample)
            return np.array(samples).reshape(-1, 1)
        else:
            # Unconditional sampling
            samples = self._gmm.sample(n_samples)
            return samples[:, 0].reshape(-1, 1)

    def predict(self, parent_values: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict values given parent values (or unconditionally).

        Args:
            parent_values: Parent values for prediction
                          Shape: (n_samples, n_parent_features) or None

        Returns:
            Predicted values (n_samples, 1)
        """
        if self._gmm is None:
            raise ValueError("Distribution must be fitted before prediction")

        if parent_values is not None:
            if parent_values.ndim == 1:
                parent_values = parent_values.reshape(-1, 1)

            predictions = []
            for pval in parent_values:
                if np.isnan(pval).all():
                    predictions.append(np.nan)
                else:
                    # Condition on parent values and predict
                    cond_gmm = self._condition_on_parents(pval)
                    # Use weighted mean of mixture components
                    pred = 0
                    for i, w in enumerate(cond_gmm.priors):
                        pred += w * cond_gmm.means[i][0]
                    predictions.append(pred)
            return np.array(predictions).reshape(-1, 1)
        else:
            # Unconditional prediction (weighted mean)
            prediction = 0
            for i, w in enumerate(self._weights):
                prediction += w * self._means[i][0]
            return np.array([[prediction]])

    def _condition_on_parents(self, parent_values: np.ndarray) -> GMM:
        """
        Create conditional GMM given parent values.

        Args:
            parent_values: Parent values to condition on (1D array)

        Returns:
            Conditional GMM
        """
        # Parent indices (assuming parents are features 1, 2, ..., n_parents)
        n_parents = len(parent_values)
        parent_indices = list(range(1, n_parents + 1))

        cond_gmm = self._gmm.condition(parent_indices, [parent_values])
        return cond_gmm

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get fitted parameters.

        Returns:
            Dictionary with 'mean', 'covars', 'coef' (weights)
        """
        if self._gmm is None:
            raise ValueError("Distribution must be fitted first")

        return {
            "mean": self._means,
            "covars": self._covariances,
            "coef": self._weights,
            "n_components": self._fitted_n_components,
        }

    def log_prob(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate log probability of data.

        Args:
            X: Data points (n_samples, n_features)

        Returns:
            Log probabilities (n_samples,)
        """
        if self._gmm is None:
            raise ValueError("Distribution must be fitted before computing log probability")

        # GMM doesn't have a direct log_prob, but we can compute it from components
        log_probs = []
        for x in X:
            # For each component, compute weighted probability
            component_probs = []
            for i in range(self._fitted_n_components):
                # Multivariate normal probability
                from scipy.stats import multivariate_normal
                mean = np.array(self._means[i])
                cov = np.array(self._covariances[i])
                weight = self._weights[i]

                prob = weight * multivariate_normal.pdf(x, mean=mean, cov=cov)
                component_probs.append(prob)

            total_prob = np.sum(component_probs)
            log_probs.append(np.log(total_prob + 1e-10))  # Add small epsilon to avoid log(0)

        return np.array(log_probs)

    def __str__(self) -> str:
        """String representation."""
        if self._gmm is None:
            return f"MixtureGaussianDistribution(unfitted, max_components={self.max_components})"
        return f"MixtureGaussianDistribution(n_components={self._fitted_n_components})"

    def __repr__(self) -> str:
        """Repr representation."""
        return self.__str__()
