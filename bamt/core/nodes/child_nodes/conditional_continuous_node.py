"""
Conditional Continuous Node for Bayesian Networks.

This module implements a conditional continuous (Gaussian) node that uses
regression models to predict continuous outcomes based on parent values.
Handles hybrid parents (discrete and continuous).
"""

import itertools
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error as rmse

from bamt.core.node_models.regressor import Regressor

from .child_node import ChildNode


class ConditionalContinuousNode(ChildNode):
    """
    Conditional Continuous Node using regression models.

    This node can have both discrete and continuous parents. For each
    combination of discrete parent values, a separate regressor is trained
    to predict the node's value based on continuous parents.

    Attributes:
        name: Node name
        regressor: Regression model (Regressor or sklearn regressor)
        disc_parents: List of discrete parent node names
        cont_parents: List of continuous parent node names
        _fitted_models: Dictionary mapping discrete parent combinations to fitted models
    """

    def __init__(
        self,
        name: str,
        regressor: Optional[object] = None,
        disc_parents: Optional[List[str]] = None,
        cont_parents: Optional[List[str]] = None,
    ):
        """
        Initialize ConditionalContinuousNode.

        Args:
            name: Node name
            regressor: Regression model (default: LinearRegression)
            disc_parents: List of discrete parent names
            cont_parents: List of continuous parent names
        """
        super().__init__()
        self.name = name
        self.disc_parents = disc_parents if disc_parents else []
        self.cont_parents = cont_parents if cont_parents else []

        if regressor is None:
            # Use default regressor
            regressor = LinearRegression()

        self.regressor = regressor
        self._fitted_models = {}
        self._model_type = type(regressor).__name__

    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the conditional continuous node.

        For each combination of discrete parent values, fit a separate
        regressor on the continuous parents.

        Args:
            data: DataFrame containing node and parent columns
        """
        self._fitted_models = {}

        # Get all combinations of discrete parent values
        if self.disc_parents:
            disc_values = []
            for d_p in self.disc_parents:
                disc_values.append(np.unique(data[d_p].values))
            combinations = list(itertools.product(*disc_values))
        else:
            # No discrete parents, single model
            combinations = [()]

        # Fit a model for each combination
        for comb in combinations:
            key_comb = str([str(x) for x in comb])

            # Filter data for this combination
            if comb:
                mask = np.full(len(data), True)
                for col, val in zip(self.disc_parents, comb):
                    mask = mask & (data[col] == val)
                subset_data = data[mask]
            else:
                subset_data = data

            if subset_data.shape[0] == 0:
                # No data for this combination
                self._fitted_models[key_comb] = {
                    "mean": np.nan,
                    "std": np.nan,
                    "model": None,
                }
                continue

            if self.cont_parents:
                # Train regressor
                model = clone(self.regressor)
                X = subset_data[self.cont_parents].values
                y = subset_data[self.name].values
                model.fit(X, y)

                # Calculate residual standard deviation
                y_pred = model.predict(X)
                std = rmse(y, y_pred)

                self._fitted_models[key_comb] = {
                    "mean": np.nan,  # Mean is predicted by model
                    "std": std,
                    "model": model,
                }
            else:
                # No continuous parents, just store mean and std
                mean = np.mean(subset_data[self.name].values)
                std = np.std(subset_data[self.name].values)

                self._fitted_models[key_comb] = {
                    "mean": mean,
                    "std": std,
                    "model": None,
                }

    def predict(self, parent_values: Dict[str, Union[str, float]]) -> float:
        """
        Predict the expected value given parent values.

        Args:
            parent_values: Dictionary mapping parent names to values

        Returns:
            Predicted continuous value
        """
        disc_vals = [str(parent_values.get(p, np.nan)) for p in self.disc_parents]
        cont_vals = [parent_values.get(p, np.nan) for p in self.cont_parents]

        key_comb = str(disc_vals)
        model_info = self._fitted_models.get(key_comb)

        if model_info is None:
            return np.nan

        if model_info["model"] is not None:
            # Use regressor to predict
            X = np.array(cont_vals).reshape(1, -1)
            pred = model_info["model"].predict(X)[0]
            return float(pred)
        else:
            # No model, return mean
            return float(model_info["mean"])

    def sample(self, parent_values: Dict[str, Union[str, float]]) -> float:
        """
        Sample a value given parent values (with Gaussian noise).

        Args:
            parent_values: Dictionary mapping parent names to values

        Returns:
            Sampled continuous value
        """
        disc_vals = [str(parent_values.get(p, np.nan)) for p in self.disc_parents]
        cont_vals = [parent_values.get(p, np.nan) for p in self.cont_parents]

        key_comb = str(disc_vals)
        model_info = self._fitted_models.get(key_comb)

        if model_info is None:
            return np.nan

        # Get mean prediction
        if model_info["model"] is not None:
            X = np.array(cont_vals).reshape(1, -1)
            mean = model_info["model"].predict(X)[0]
        else:
            mean = model_info["mean"]

        # Sample from Gaussian distribution
        std = model_info["std"]
        if np.isnan(std) or std == 0:
            return float(mean)

        sampled_value = np.random.normal(mean, std)
        return float(sampled_value)

    def get_distribution(
        self, parent_values: Dict[str, Union[str, float]]
    ) -> Tuple[float, float]:
        """
        Get Gaussian distribution parameters given parent values.

        Args:
            parent_values: Dictionary mapping parent names to values

        Returns:
            Tuple of (mean, std)
        """
        disc_vals = [str(parent_values.get(p, np.nan)) for p in self.disc_parents]
        cont_vals = [parent_values.get(p, np.nan) for p in self.cont_parents]

        key_comb = str(disc_vals)
        model_info = self._fitted_models.get(key_comb)

        if model_info is None:
            return np.nan, np.nan

        if model_info["model"] is not None:
            X = np.array(cont_vals).reshape(1, -1)
            mean = model_info["model"].predict(X)[0]
        else:
            mean = model_info["mean"]

        std = model_info["std"]
        return float(mean), float(std)

    def get_children(self):
        """Get child nodes (not implemented in this simplified version)."""
        return []

    def get_parents(self):
        """Get parent nodes."""
        return self.disc_parents + self.cont_parents

    def __str__(self):
        return f"ConditionalContinuousNode({self.name}, {self._model_type})"
