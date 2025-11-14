"""
Extended ML model repository for BAMT 2.0.0 architecture.

This module provides access to a comprehensive set of machine learning models
for use in Bayesian Networks, including advanced gradient boosting models
(XGBoost, CatBoost, LightGBM) and various sklearn models.
"""

from typing import Dict, Any, Optional, List
import warnings

# Standard sklearn models
from sklearn.ensemble import (
    AdaBoostRegressor,
    AdaBoostClassifier,
    ExtraTreesRegressor,
    ExtraTreesClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    SGDRegressor,
    SGDClassifier,
)
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR

# Try to import advanced gradient boosting libraries
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    warnings.warn("CatBoost not available. Install with: pip install catboost")

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM not available. Install with: pip install lightgbm")


class MLModelsRepository:
    """
    Repository of machine learning models for Bayesian Networks.

    Provides access to a comprehensive set of classifiers and regressors,
    including standard sklearn models and advanced gradient boosting models.

    Example:
        >>> repo = MLModelsRepository()
        >>> classifiers = repo.get_classifiers()
        >>> regressors = repo.get_regressors()
        >>> model = repo.get_model('XGBRegressor', n_estimators=100)
    """

    def __init__(self):
        """Initialize the ML models repository."""
        self._initialize_models()

    def _initialize_models(self):
        """Initialize model dictionaries."""
        # Regressors - always available
        self.regressors = {
            "LinearRegression": LinearRegression,
            "Ridge": Ridge,
            "Lasso": Lasso,
            "SGDRegressor": SGDRegressor,
            "DecisionTreeRegressor": DecisionTreeRegressor,
            "RandomForestRegressor": RandomForestRegressor,
            "ExtraTreesRegressor": ExtraTreesRegressor,
            "AdaBoostRegressor": AdaBoostRegressor,
            "GradientBoostingRegressor": GradientBoostingRegressor,
            "MLPRegressor": MLPRegressor,
            "SVR": SVR,
        }

        # Classifiers - always available
        self.classifiers = {
            "LogisticRegression": LogisticRegression,
            "SGDClassifier": SGDClassifier,
            "DecisionTreeClassifier": DecisionTreeClassifier,
            "RandomForestClassifier": RandomForestClassifier,
            "ExtraTreesClassifier": ExtraTreesClassifier,
            "AdaBoostClassifier": AdaBoostClassifier,
            "GradientBoostingClassifier": GradientBoostingClassifier,
            "GaussianNB": GaussianNB,
            "BernoulliNB": BernoulliNB,
            "MultinomialNB": MultinomialNB,
            "MLPClassifier": MLPClassifier,
            "SVC": SVC,
        }

        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.regressors["XGBRegressor"] = XGBRegressor
            self.classifiers["XGBClassifier"] = XGBClassifier

        # Add CatBoost if available
        if CATBOOST_AVAILABLE:
            self.regressors["CatBoostRegressor"] = CatBoostRegressor
            self.classifiers["CatBoostClassifier"] = CatBoostClassifier

        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            self.regressors["LGBMRegressor"] = LGBMRegressor
            self.classifiers["LGBMClassifier"] = LGBMClassifier

        # Combined dictionary
        self.all_models = {**self.regressors, **self.classifiers}

    def get_regressors(self, include_advanced: bool = True) -> Dict[str, Any]:
        """
        Get dictionary of available regressors.

        Args:
            include_advanced: Include XGBoost/CatBoost/LightGBM if available

        Returns:
            Dictionary mapping model names to model classes
        """
        if include_advanced:
            return self.regressors.copy()

        # Return only standard sklearn models
        standard_models = {
            k: v
            for k, v in self.regressors.items()
            if not any(x in k for x in ["XGB", "CatBoost", "LGBM"])
        }
        return standard_models

    def get_classifiers(self, include_advanced: bool = True) -> Dict[str, Any]:
        """
        Get dictionary of available classifiers.

        Args:
            include_advanced: Include XGBoost/CatBoost/LightGBM if available

        Returns:
            Dictionary mapping model names to model classes
        """
        if include_advanced:
            return self.classifiers.copy()

        # Return only standard sklearn models
        standard_models = {
            k: v
            for k, v in self.classifiers.items()
            if not any(x in k for x in ["XGB", "CatBoost", "LGBM"])
        }
        return standard_models

    def get_model(self, model_name: str, **kwargs) -> Any:
        """
        Get instantiated model by name.

        Args:
            model_name: Name of the model
            **kwargs: Parameters for model initialization

        Returns:
            Instantiated model

        Example:
            >>> repo = MLModelsRepository()
            >>> model = repo.get_model('RandomForestRegressor', n_estimators=100)
        """
        if model_name not in self.all_models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.all_models.keys())}")

        model_class = self.all_models[model_name]
        return model_class(**kwargs)

    def get_default_params(self, model_name: str) -> Dict[str, Any]:
        """
        Get recommended default parameters for a model.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary of default parameters
        """
        defaults = {
            # Tree-based models
            "RandomForestRegressor": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
            "RandomForestClassifier": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
            "DecisionTreeRegressor": {"max_depth": 10, "random_state": 42},
            "DecisionTreeClassifier": {"max_depth": 10, "random_state": 42},
            "ExtraTreesRegressor": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
            "ExtraTreesClassifier": {"n_estimators": 100, "max_depth": 10, "random_state": 42},

            # Gradient Boosting
            "GradientBoostingRegressor": {"n_estimators": 100, "max_depth": 5, "random_state": 42},
            "GradientBoostingClassifier": {"n_estimators": 100, "max_depth": 5, "random_state": 42},
            "AdaBoostRegressor": {"n_estimators": 50, "random_state": 42},
            "AdaBoostClassifier": {"n_estimators": 50, "random_state": 42},

            # Linear models
            "LinearRegression": {},
            "LogisticRegression": {"max_iter": 1000, "random_state": 42},
            "Ridge": {"alpha": 1.0, "random_state": 42},
            "Lasso": {"alpha": 1.0, "random_state": 42},
            "SGDRegressor": {"random_state": 42},
            "SGDClassifier": {"random_state": 42},

            # Neural Networks
            "MLPRegressor": {"hidden_layer_sizes": (100,), "max_iter": 1000, "random_state": 42},
            "MLPClassifier": {"hidden_layer_sizes": (100,), "max_iter": 1000, "random_state": 42},

            # Naive Bayes
            "GaussianNB": {},
            "BernoulliNB": {},
            "MultinomialNB": {},

            # SVM
            "SVC": {"random_state": 42},
            "SVR": {},
        }

        # Advanced models
        if XGBOOST_AVAILABLE:
            defaults["XGBRegressor"] = {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1, "random_state": 42}
            defaults["XGBClassifier"] = {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1, "random_state": 42}

        if CATBOOST_AVAILABLE:
            defaults["CatBoostRegressor"] = {"iterations": 100, "depth": 5, "learning_rate": 0.1, "random_state": 42, "verbose": False}
            defaults["CatBoostClassifier"] = {"iterations": 100, "depth": 5, "learning_rate": 0.1, "random_state": 42, "verbose": False}

        if LIGHTGBM_AVAILABLE:
            defaults["LGBMRegressor"] = {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1, "random_state": 42, "verbose": -1}
            defaults["LGBMClassifier"] = {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1, "random_state": 42, "verbose": -1}

        return defaults.get(model_name, {})

    def get_fast_models(self, model_type: str = "both") -> Dict[str, Any]:
        """
        Get computationally efficient models (fast training).

        Args:
            model_type: 'classifier', 'regressor', or 'both'

        Returns:
            Dictionary of fast models
        """
        fast_regressors = {
            "LinearRegression": LinearRegression,
            "Ridge": Ridge,
            "Lasso": Lasso,
            "DecisionTreeRegressor": DecisionTreeRegressor,
        }

        fast_classifiers = {
            "LogisticRegression": LogisticRegression,
            "DecisionTreeClassifier": DecisionTreeClassifier,
            "GaussianNB": GaussianNB,
        }

        if model_type == "regressor":
            return fast_regressors
        elif model_type == "classifier":
            return fast_classifiers
        else:
            return {**fast_regressors, **fast_classifiers}

    def is_model_available(self, model_name: str) -> bool:
        """
        Check if a model is available.

        Args:
            model_name: Name of the model

        Returns:
            True if model is available, False otherwise
        """
        return model_name in self.all_models

    def list_available_models(self, model_type: Optional[str] = None) -> List[str]:
        """
        List all available models.

        Args:
            model_type: Filter by 'classifier', 'regressor', or None for all

        Returns:
            List of available model names
        """
        if model_type == "regressor":
            return list(self.regressors.keys())
        elif model_type == "classifier":
            return list(self.classifiers.keys())
        else:
            return list(self.all_models.keys())


# Global repository instance
_REPOSITORY = None


def get_ml_repository() -> MLModelsRepository:
    """
    Get the global ML models repository instance.

    Returns:
        Singleton MLModelsRepository instance
    """
    global _REPOSITORY
    if _REPOSITORY is None:
        _REPOSITORY = MLModelsRepository()
    return _REPOSITORY
