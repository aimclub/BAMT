"""
Regressor model with automatic algorithm selection.

This module provides a Regressor class that can automatically select
the best regression algorithm from a set of candidates using
cross-validation.
"""

from typing import Dict, Optional, Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

from .prediction_model import PredictionModel


class Regressor(PredictionModel):
    """
    Automatic regressor selection and prediction model.

    This class can either use a provided regressor or automatically
    select the best one from a set of candidates using cross-validation.

    Attributes:
        _regressor: The selected/provided regressor
        _parameters: Additional parameters for the regressor
        _candidate_models: Dictionary of candidate regressors to try
        _cv_folds: Number of cross-validation folds (default: 5)
        _scoring: Scoring metric for model selection (default: 'neg_mean_squared_error')
    """

    DEFAULT_REGRESSORS = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "RandomForest": RandomForestRegressor(
            n_estimators=50, max_depth=5, random_state=42
        ),
        "DecisionTree": DecisionTreeRegressor(max_depth=5, random_state=42),
    }

    def __init__(
        self,
        regressor=None,
        candidate_models: Optional[Dict[str, Any]] = None,
        cv_folds: int = 5,
        scoring: str = "neg_mean_squared_error",
        **parameters
    ):
        """
        Initialize Regressor.

        Args:
            regressor: Pre-specified regressor (if None, auto-select)
            candidate_models: Dictionary of candidate regressors to try
            cv_folds: Number of CV folds for model selection
            scoring: Scoring metric ('neg_mean_squared_error', 'r2', etc.)
            **parameters: Additional parameters
        """
        self._regressor = regressor
        self._parameters = parameters
        self._candidate_models = (
            candidate_models if candidate_models else self.DEFAULT_REGRESSORS
        )
        self._cv_folds = cv_folds
        self._scoring = scoring
        self._best_model_name = None
        self._cv_scores = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the regressor.

        If no regressor is specified, automatically selects the best one
        using cross-validation.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
        """
        if self._regressor is None:
            # Auto-select best regressor
            self._regressor = self._select_best_regressor(X, y)

        # Fit the selected regressor
        self._regressor.fit(X, y)

    def _select_best_regressor(self, X: np.ndarray, y: np.ndarray):
        """
        Select the best regressor using cross-validation.

        Args:
            X: Feature matrix
            y: Target values

        Returns:
            The best performing regressor
        """
        best_score = -np.inf
        best_model = None
        best_name = None

        # Check if we have enough samples for CV
        n_samples = X.shape[0]
        cv_folds = min(self._cv_folds, n_samples)

        # If too few samples, just use the first model
        if n_samples < 2:
            best_name = list(self._candidate_models.keys())[0]
            best_model = list(self._candidate_models.values())[0]
            self._best_model_name = best_name
            self._cv_scores[best_name] = 0.0
            return best_model

        for name, model in self._candidate_models.items():
            try:
                # Perform cross-validation
                scores = cross_val_score(
                    model, X, y, cv=cv_folds, scoring=self._scoring
                )
                mean_score = np.mean(scores)
                self._cv_scores[name] = mean_score

                if mean_score > best_score:
                    best_score = mean_score
                    best_model = model
                    best_name = name
            except Exception:
                # Skip models that fail (e.g., incompatible with data)
                self._cv_scores[name] = -np.inf
                continue

        self._best_model_name = best_name
        return best_model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predicted values (n_samples,)
        """
        if self._regressor is None:
            raise RuntimeError("Regressor not fitted. Call fit() first.")
        return self._regressor.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        For compatibility with PredictionModel interface.
        Most regressors don't have predict_proba.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predictions (same as predict for regressors)
        """
        # For regressors, this just returns predictions
        # since they don't have probability distributions
        return self.predict(X)

    def get_best_model_info(self) -> Dict[str, Any]:
        """
        Get information about the selected model.

        Returns:
            Dictionary with model name and CV scores
        """
        return {
            "best_model": self._best_model_name,
            "cv_scores": self._cv_scores,
        }

    def __str__(self):
        if self._regressor is None:
            return "Regressor (not fitted)"
        model_str = str(self._regressor)
        if self._best_model_name:
            return f"Regressor ({self._best_model_name}): {model_str}"
        return f"Regressor: {model_str}"

    def __getattr__(self, name: str):
        if self._regressor:
            return getattr(self._regressor, name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )
