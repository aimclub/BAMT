import numpy as np
from typing import Optional, Union, List
from distribution import Distribution


# noinspection PyPep8Naming
class EmpiricalDistribution(Distribution):
    """
    Class for empirical distributions.
    This class fits an empirical distribution to the provided categorical or discrete data by calculating
    the probabilities of unique values and allows sampling from it.
    Usage example:
    ```python
    data = ['apple', 'banana', 'apple', 'orange', 'banana', 'banana', 'orange', 'apple']
    emp_dist = EmpiricalDistribution()
    emp_dist.fit(data)
    print(emp_dist)
    samples = emp_dist.sample(10)
    print(samples)
    print(emp_dist.pmf('banana'))
    ```
    """

    def __init__(self) -> None:
        """
        Initialize the EmpiricalDistribution.
        """
        self._values: Optional[np.ndarray] = None
        self._probabilities: Optional[np.ndarray] = None

    def fit(self, X: Union[np.ndarray, List[Union[str, int]]]) -> None:
        """
        Fit the empirical distribution to the categorical or discrete data by
        calculating the probabilities of unique values.

        Args:
            X (Union[np.ndarray, List[Union[str, int]]]): The categorical or discrete data to fit.
        """
        X = np.asarray(X)
        unique_values, counts = np.unique(X, return_counts=True)
        self._values = unique_values
        self._probabilities = counts / counts.sum()

    def sample(self, num_samples: int) -> np.ndarray:
        """
        Generate samples from the empirical distribution.

        Args:
            num_samples (int): The number of samples to generate.

        Returns:
            np.ndarray: The generated samples.

        Raises:
            ValueError: If no data has been fitted yet.
        """
        if self._values is None or self._probabilities is None:
            raise ValueError("No data fitted yet.")
        return np.random.choice(
            self._values, size=num_samples, p=self._probabilities, replace=True
        )

    def pmf(self, value: Union[str, int]) -> np.ndarray | float:
        """
        Return the probability mass function (PMF) for a given value.

        Args:
            value (Union[str, int]): The categorical or discrete value to get the PMF for.

        Returns:
            float: The PMF of the given value.

        Raises:
            ValueError: If no data has been fitted yet.
        """
        if self._values is None or self._probabilities is None:
            raise ValueError("No data fitted yet.")
        idx = np.where(self._values == value)
        if idx[0].size == 0:
            return 0.0
        return self._probabilities[idx][0]

    def __str__(self) -> str:
        """
        Return the string representation of the empirical distribution.

        Returns:
            str: The name of the distribution.
        """
        return "Empirical Distribution"
