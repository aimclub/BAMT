"""
Classifier model with automatic algorithm selection.

This module provides a Classifier class that can automatically select
the best classification algorithm from a set of candidates using
cross-validation.
"""

from typing import Dict, Optional, Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from .prediction_model import PredictionModel


class Classifier(PredictionModel):
    """
    Automatic classifier selection and prediction model.

    This class can either use a provided classifier or automatically
    select the best one from a set of candidates using cross-validation.

    Attributes:
        _classifier: The selected/provided classifier
        _parameters: Additional parameters for the classifier
        _candidate_models: Dictionary of candidate classifiers to try
        _cv_folds: Number of cross-validation folds (default: 5)
        _scoring: Scoring metric for model selection (default: 'accuracy')
    """

    DEFAULT_CLASSIFIERS = {
        "LogisticRegression": LogisticRegression(
            solver="newton-cg", max_iter=100, random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=50, max_depth=5, random_state=42
        ),
        "DecisionTree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "GaussianNB": GaussianNB(),
    }

    def __init__(
        self,
        classifier=None,
        candidate_models: Optional[Dict[str, Any]] = None,
        cv_folds: int = 5,
        scoring: str = "accuracy",
        **parameters
    ):
        """
        Initialize Classifier.

        Args:
            classifier: Pre-specified classifier (if None, auto-select)
            candidate_models: Dictionary of candidate classifiers to try
            cv_folds: Number of CV folds for model selection
            scoring: Scoring metric ('accuracy', 'f1', 'roc_auc', etc.)
            **parameters: Additional parameters
        """
        self._classifier = classifier
        self._parameters = parameters
        self._candidate_models = (
            candidate_models if candidate_models else self.DEFAULT_CLASSIFIERS
        )
        self._cv_folds = cv_folds
        self._scoring = scoring
        self._best_model_name = None
        self._cv_scores = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the classifier.

        If no classifier is specified, automatically selects the best one
        using cross-validation.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)
        """
        if self._classifier is None:
            # Auto-select best classifier
            self._classifier = self._select_best_classifier(X, y)

        # Fit the selected classifier
        self._classifier.fit(X, y)

    def _select_best_classifier(self, X: np.ndarray, y: np.ndarray):
        """
        Select the best classifier using cross-validation.

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            The best performing classifier
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
        Predict class labels.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predicted class labels (n_samples,)
        """
        if self._classifier is None:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
        return self._classifier.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Class probabilities (n_samples, n_classes)
        """
        if self._classifier is None:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
        return self._classifier.predict_proba(X)

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
        if self._classifier is None:
            return "Classifier (not fitted)"
        model_str = str(self._classifier)
        if self._best_model_name:
            return f"Classifier ({self._best_model_name}): {model_str}"
        return f"Classifier: {model_str}"

    def __getattr__(self, name: str):
        if self._classifier:
            return getattr(self._classifier, name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )
