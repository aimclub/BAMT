"""
Conditional Discrete Node for Bayesian Networks.

This module implements a conditional discrete (categorical) node that uses
classification models to predict discrete outcomes based on parent values.
Handles hybrid parents (discrete and continuous).
"""

import itertools
import random
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from bamt.core.node_models.classifier import Classifier

from .child_node import ChildNode


class ConditionalDiscreteNode(ChildNode):
    """
    Conditional Discrete Node using classification models.

    This node can have both discrete and continuous parents. For each
    combination of discrete parent values, a separate classifier is trained
    to predict the node's value based on continuous parents.

    Attributes:
        name: Node name
        classifier: Classification model (Classifier or sklearn classifier)
        disc_parents: List of discrete parent node names
        cont_parents: List of continuous parent node names
        _fitted_models: Dictionary mapping discrete parent combinations to fitted models
    """

    def __init__(
        self,
        name: str,
        classifier: Optional[object] = None,
        disc_parents: Optional[List[str]] = None,
        cont_parents: Optional[List[str]] = None,
    ):
        """
        Initialize ConditionalDiscreteNode.

        Args:
            name: Node name
            classifier: Classification model (default: LogisticRegression)
            disc_parents: List of discrete parent names
            cont_parents: List of continuous parent names
        """
        super().__init__()
        self.name = name
        self.disc_parents = disc_parents if disc_parents else []
        self.cont_parents = cont_parents if cont_parents else []

        if classifier is None:
            # Use default classifier
            classifier = LogisticRegression(solver="newton-cg", max_iter=100, random_state=42)

        self.classifier = classifier
        self._fitted_models = {}
        self._model_type = type(classifier).__name__

    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the conditional discrete node.

        For each combination of discrete parent values, fit a separate
        classifier on the continuous parents.

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
                    "classes": [np.nan],
                    "model": None,
                }
                continue

            classes = np.unique(subset_data[self.name].values)

            if len(classes) <= 1:
                # Only one class, no need to train
                self._fitted_models[key_comb] = {
                    "classes": list(classes),
                    "model": None,
                }
            else:
                # Train classifier
                if self.cont_parents:
                    model = clone(self.classifier)
                    X = subset_data[self.cont_parents].values
                    y = subset_data[self.name].values
                    model.fit(X, y)
                    self._fitted_models[key_comb] = {
                        "classes": list(model.classes_),
                        "model": model,
                    }
                else:
                    # No continuous parents, just store classes
                    self._fitted_models[key_comb] = {
                        "classes": list(classes),
                        "model": None,
                    }

    def predict(self, parent_values: Dict[str, Union[str, float]]) -> str:
        """
        Predict the most likely value given parent values.

        Args:
            parent_values: Dictionary mapping parent names to values

        Returns:
            Predicted class label
        """
        disc_vals = [str(parent_values.get(p, np.nan)) for p in self.disc_parents]
        cont_vals = [parent_values.get(p, np.nan) for p in self.cont_parents]

        key_comb = str(disc_vals)
        model_info = self._fitted_models.get(key_comb)

        if model_info is None or model_info["model"] is None:
            # No model or only one class
            return str(model_info["classes"][0]) if model_info else "nan"

        # Predict using the model
        X = np.array(cont_vals).reshape(1, -1)
        pred = model_info["model"].predict(X)[0]
        return str(pred)

    def sample(self, parent_values: Dict[str, Union[str, float]]) -> str:
        """
        Sample a value given parent values (with randomness).

        Args:
            parent_values: Dictionary mapping parent names to values

        Returns:
            Sampled class label
        """
        disc_vals = [str(parent_values.get(p, np.nan)) for p in self.disc_parents]
        cont_vals = [parent_values.get(p, np.nan) for p in self.cont_parents]

        key_comb = str(disc_vals)
        model_info = self._fitted_models.get(key_comb)

        if model_info is None or model_info["model"] is None:
            # No model or only one class
            return str(model_info["classes"][0]) if model_info else "nan"

        if len(model_info["classes"]) == 1:
            return str(model_info["classes"][0])

        # Get probability distribution
        X = np.array(cont_vals).reshape(1, -1)
        probs = model_info["model"].predict_proba(X)[0]

        # Sample based on probabilities
        rand = random.random()
        cumsum = 0
        for i, p in enumerate(probs):
            cumsum += p
            if rand < cumsum:
                return str(model_info["classes"][i])

        return str(model_info["classes"][-1])

    def get_distribution(
        self, parent_values: Dict[str, Union[str, float]]
    ) -> Tuple[np.ndarray, List]:
        """
        Get probability distribution given parent values.

        Args:
            parent_values: Dictionary mapping parent names to values

        Returns:
            Tuple of (probabilities, class_labels)
        """
        disc_vals = [str(parent_values.get(p, np.nan)) for p in self.disc_parents]
        cont_vals = [parent_values.get(p, np.nan) for p in self.cont_parents]

        key_comb = str(disc_vals)
        model_info = self._fitted_models.get(key_comb)

        if model_info is None:
            return np.array([1.0]), ["nan"]

        if model_info["model"] is None or len(model_info["classes"]) == 1:
            return np.array([1.0]), model_info["classes"]

        # Get probability distribution
        X = np.array(cont_vals).reshape(1, -1)
        probs = model_info["model"].predict_proba(X)[0]
        return probs, model_info["classes"]

    def get_children(self):
        """Get child nodes (not implemented in this simplified version)."""
        return []

    def get_parents(self):
        """Get parent nodes."""
        return self.disc_parents + self.cont_parents

    def __str__(self):
        return f"ConditionalDiscreteNode({self.name}, {self._model_type})"
