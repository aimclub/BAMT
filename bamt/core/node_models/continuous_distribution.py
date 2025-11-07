from enum import Enum
from typing import Tuple, Optional, List, Type, Dict

try:
    import numpy as np
    from scipy import stats
    from scipy.special import kl_div
    from scipy.stats import rv_continuous
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    np = None
    stats = None

from .distribution import Distribution

if SCIPY_AVAILABLE:
    # Get all continuous distributions from scipy.stats
    _CONTINUOUS_DISTRIBUTIONS = [
        getattr(stats, name)
        for name in dir(stats)
        if isinstance(getattr(stats, name), type) and issubclass(getattr(stats, name), stats.rv_continuous)
    ]
else:
    _CONTINUOUS_DISTRIBUTIONS = []


class DistributionPool(Enum):
    """
    Enum for selecting the distribution pool.
    """

    SMALL = "small"
    LARGE = "large"
    CUSTOM = "custom"


# noinspection PyPep8Naming
class ContinuousDistribution(Distribution):
    """
    Class for continuous distributions.
    This class is a wrapper for continuous distributions from `scipy.stats` module,
    however, any custom continuous distribution can be used, as long as it implements
    `scipy.stats` interface.
    Example Usage:
    >>> data = np.random.normal(0, 1, 1000)
    >>> dist = ContinuousDistribution()
    >>> dist.fit(data, distributions_pool=DistributionPool.SMALL)
    >>> samples = dist.sample(10)
    """

    if SCIPY_AVAILABLE:
        SMALL_POOL: Tuple[Type[stats.rv_continuous], ...] = (
            stats.norm,
            stats.laplace,
            stats.t,
            stats.uniform,
            stats.rayleigh,
        )

        LARGE_POOL: List[Type[stats.rv_continuous]] = _CONTINUOUS_DISTRIBUTIONS
    else:
        SMALL_POOL = ()
        LARGE_POOL = []

    def __init__(
        self,
        distribution_model: Optional[Type] = None,
        **parameters,
    ) -> None:
        """
        Initialize the ContinuousDistribution with an optional distribution model and parameters.

        Args:
            distribution_model (Optional[Type[stats.rv_continuous]]): A specific `scipy.stats` distribution.
            **parameters: Parameters for the specified distribution model.
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for ContinuousDistribution")
        self._distribution_model = distribution_model
        self._parameters = parameters

    def fit(
        self,
        X,
        distributions_pool: DistributionPool = DistributionPool.SMALL,
        custom_pool: Optional[List[Type]] = None,
    ) -> None:
        """
        Fit the data to the best distribution within the specified pool.

        Args:
            X (np.ndarray): The data to fit.
            distributions_pool (DistributionPool): The pool of distributions to consider (small, large, custom).
            custom_pool (Optional[List[Type[stats.rv_continuous]]]): if `DistributionPool.CUSTOM` is selected.

        Raises:
            ValueError: If a custom pool is selected but not provided.
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for fit method")
            
        if self._distribution_model is None:
            pool = self._select_pool(distributions_pool, custom_pool)
            self._distribution_model, self._parameters = self._fit_best_distribution(
                X, pool
            )
        else:
            self._parameters = self._distribution_model.fit(X)

    @staticmethod
    def _select_pool(
        pool_type: DistributionPool,
        custom_pool: Optional[List[Type]],
    ) -> List[Type]:
        """
        Select the appropriate pool of distributions.

        Args:
            pool_type (DistributionPool): The type of pool to select.
            custom_pool (Optional[List[Type[stats.rv_continuous]]]): The custom pool of distributions.

        Returns:
            List[Type[stats.rv_continuous]]: The selected pool of distributions.

        Raises:
            ValueError: If a custom pool is selected but not provided.
        """
        if pool_type == DistributionPool.SMALL:
            return list(ContinuousDistribution.SMALL_POOL)
        elif pool_type == DistributionPool.LARGE:
            return ContinuousDistribution.LARGE_POOL
        elif pool_type == DistributionPool.CUSTOM:
            if custom_pool is not None:
                return custom_pool
            else:
                raise ValueError("Custom pool selected but no custom pool provided")
        else:
            raise ValueError("Invalid distribution pool type")

    @staticmethod
    def _fit_best_distribution(
        X, distribution_models_pool: List[Type]
    ) -> Tuple[Type, Dict[str, float]]:
        """
        Fit the data to the best distribution in the pool by minimizing the KL divergence.

        Args:
            X (np.ndarray): The data to fit.
            distribution_models_pool (List[Type[stats.rv_continuous]]): The pool of distributions to consider.

        Returns:
            Tuple[Optional[Type[stats.rv_continuous]], dict]: The best fitting distribution and its parameters.
        """
        best_distribution = None
        best_params = None
        min_kl_divergence: float = np.inf

        # Compute empirical histogram
        hist, bin_edges = np.histogram(X, bins="auto", density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        for distribution in distribution_models_pool:
            try:
                # Fit the distribution to the data
                params = distribution.fit(X)

                # Compute the fitted probability density function
                pdf = distribution.pdf(bin_centers, *params)

                # Compute the KL divergence between the empirical and fitted distribution
                kl_divergence = np.sum(kl_div(hist, pdf))

                # Update the best distribution if the current one is better
                if kl_divergence < min_kl_divergence:
                    min_kl_divergence = kl_divergence
                    best_distribution = distribution
                    best_params = params
            except Exception as e:
                # Handle any exceptions that occur during fitting
                continue

        return best_distribution, best_params

    def sample(self, num_samples: int):
        """
        Generate samples from the fitted distribution.

        Args:
            num_samples (int): The number of samples to generate.

        Returns:
            np.ndarray: The generated samples.

        Raises:
            ValueError: If no distribution is fitted yet.
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for sample method")
        if self._distribution_model is None:
            raise ValueError("No distribution fitted yet.")
        return self._distribution_model.rvs(*self._parameters, size=num_samples)

    def __getattr__(self, name: str):
        """
        Redirect method calls to the underlying distribution if it exists.

        Args:
            name (str): The name of the attribute or method.

        Returns:
            Any: The attribute or method from the underlying distribution.

        Raises:
            AttributeError: If the attribute or method does not exist.
        """
        if self._distribution_model:
            return getattr(self._distribution_model, name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __str__(self) -> str:
        """
        Return the string representation of the distribution.

        Returns:
            str: The name of the fitted distribution or a message indicating no distribution is fitted.
        """
        if self._distribution_model:
            return f"{self._distribution_model.name} continuous distribution"
        return "No distribution fitted yet"
