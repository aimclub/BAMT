"""
Composite Bayesian Network implementation.

This module implements a Bayesian Network with advanced ML model support for both
discrete and continuous variables. Similar to HybridBN but with explicit control
over which ML models are used for each conditional node.
"""

from typing import Dict, List, Optional, Any

import networkx as nx
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression

from bamt.core.node_models.classifier import Classifier
from bamt.core.node_models.regressor import Regressor
from bamt.core.nodes.child_nodes.conditional_continuous_node import (
    ConditionalContinuousNode,
)
from bamt.core.nodes.child_nodes.conditional_discrete_node import (
    ConditionalDiscreteNode,
)
from bamt.core.nodes.root_nodes.continuous_node import ContinuousNode
from bamt.core.nodes.root_nodes.discrete_node import DiscreteNode
from bamt.parameter_estimators.maximum_likelihood_estimator import (
    MaximumLikelihoodEstimator,
)

from .bayesian_network import BayesianNetwork


class CompositeBayesianNetwork(BayesianNetwork):
    """
    Composite Bayesian Network with ML Models.

    A Bayesian Network for mixed discrete and continuous data with explicit
    control over which ML models are used for conditional nodes. This provides
    flexibility to use XGBoost, CatBoost, LightGBM, or any sklearn-compatible
    model for prediction.

    The key difference from HybridBN is that CompositeBN allows you to:
    1. Specify custom classifiers/regressors per node
    2. Use ensemble models (XGBoost, CatBoost, etc.)
    3. Fine-tune model selection for each conditional relationship

    Attributes:
        structure: DAG structure (NetworkX DiGraph)
        nodes: Dictionary mapping node names to node objects
        discrete_columns: Set of discrete column names
        continuous_columns: Set of continuous column names
        node_classifiers: Dict mapping node names to custom classifiers
        node_regressors: Dict mapping node names to custom regressors
        _fitted: Whether the network has been fitted
    """

    def __init__(
        self,
        structure: Optional[nx.DiGraph] = None,
        discrete_columns: Optional[List[str]] = None,
        continuous_columns: Optional[List[str]] = None,
    ):
        """
        Initialize Composite Bayesian Network.

        Args:
            structure: Optional DAG structure (NetworkX DiGraph)
            discrete_columns: List of discrete column names
            continuous_columns: List of continuous column names
        """
        super().__init__()
        self.structure = structure if structure else nx.DiGraph()
        self.nodes = {}
        self.discrete_columns = set(discrete_columns) if discrete_columns else set()
        self.continuous_columns = (
            set(continuous_columns) if continuous_columns else set()
        )
        self.node_classifiers: Dict[str, Any] = {}
        self.node_regressors: Dict[str, Any] = {}
        self._fitted = False
        self._data_columns = []

    def set_classifiers(self, classifiers: Dict[str, Any]) -> None:
        """
        Set custom classifiers for specific discrete nodes.

        Args:
            classifiers: Dictionary mapping node names to sklearn-compatible
                        classifier instances or dicts of candidate models

        Example:
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> from xgboost import XGBClassifier
            >>> bn.set_classifiers({
            ...     'node1': RandomForestClassifier(n_estimators=100),
            ...     'node2': {'RF': RandomForestClassifier(), 'XGB': XGBClassifier()}
            ... })
        """
        self.node_classifiers.update(classifiers)

    def set_regressors(self, regressors: Dict[str, Any]) -> None:
        """
        Set custom regressors for specific continuous nodes.

        Args:
            regressors: Dictionary mapping node names to sklearn-compatible
                       regressor instances or dicts of candidate models

        Example:
            >>> from sklearn.ensemble import RandomForestRegressor
            >>> from catboost import CatBoostRegressor
            >>> bn.set_regressors({
            ...     'node1': RandomForestRegressor(n_estimators=100),
            ...     'node2': {'RF': RandomForestRegressor(), 'CB': CatBoostRegressor()}
            ... })
        """
        self.node_regressors.update(regressors)

    def _infer_column_types(self, data: pd.DataFrame) -> None:
        """
        Infer which columns are discrete vs continuous.

        Uses heuristic: if column has ≤10 unique values or is categorical/object type,
        treat as discrete. Otherwise, treat as continuous.

        Args:
            data: The dataset
        """
        if not self.discrete_columns and not self.continuous_columns:
            for col in data.columns:
                if data[col].dtype in ["object", "category", "bool"]:
                    self.discrete_columns.add(col)
                elif data[col].nunique() <= 10:
                    self.discrete_columns.add(col)
                else:
                    self.continuous_columns.add(col)

    def fit(
        self,
        data: pd.DataFrame,
        structure: Optional[nx.DiGraph] = None,
        parameter_estimator: Optional[MaximumLikelihoodEstimator] = None,
        **kwargs
    ) -> "CompositeBayesianNetwork":
        """
        Fit the Composite Bayesian Network.

        Args:
            data: Training data (mixed discrete and continuous columns)
            structure: Optional structure (if not set during init)
            parameter_estimator: Parameter estimator (default: MLE)
            **kwargs: Additional parameters for node fitting

        Returns:
            self
        """
        if structure is not None:
            self.structure = structure

        if self.structure is None or len(self.structure.nodes) == 0:
            raise ValueError(
                "Structure must be provided either during init or in fit()"
            )

        # Infer column types if not specified
        self._infer_column_types(data)
        self._data_columns = list(data.columns)

        # Create parameter estimator if not provided
        if parameter_estimator is None:
            parameter_estimator = MaximumLikelihoodEstimator()

        # Create nodes based on structure
        self.nodes = {}
        for node_name in self.structure.nodes:
            if node_name not in data.columns:
                raise ValueError(f"Node '{node_name}' not found in data columns")

            parents = list(self.structure.predecessors(node_name))
            is_discrete = node_name in self.discrete_columns

            # Create appropriate node type
            if len(parents) == 0:
                # Root node
                if is_discrete:
                    node = DiscreteNode(name=node_name)
                else:
                    node = ContinuousNode(name=node_name)
            else:
                # Child node with parents
                if is_discrete:
                    # Get custom classifier if provided
                    if node_name in self.node_classifiers:
                        custom_model = self.node_classifiers[node_name]
                        if isinstance(custom_model, dict):
                            # Multiple candidate models
                            classifier = Classifier(candidate_models=custom_model)
                        else:
                            # Single model
                            classifier = Classifier(
                                candidate_models={"custom": custom_model}
                            )
                    else:
                        # Use default auto-selection
                        classifier = Classifier()

                    node = ConditionalDiscreteNode(
                        name=node_name, parents=parents, classifier=classifier
                    )
                else:
                    # Continuous child node
                    # Get custom regressor if provided
                    if node_name in self.node_regressors:
                        custom_model = self.node_regressors[node_name]
                        if isinstance(custom_model, dict):
                            # Multiple candidate models
                            regressor = Regressor(candidate_models=custom_model)
                        else:
                            # Single model
                            regressor = Regressor(
                                candidate_models={"custom": custom_model}
                            )
                    else:
                        # Use default auto-selection
                        regressor = Regressor()

                    node = ConditionalContinuousNode(
                        name=node_name, parents=parents, regressor=regressor
                    )

            # Fit the node
            node.fit(data, **kwargs)
            self.nodes[node_name] = node

        self._fitted = True
        return self

    def predict(
        self, data: pd.DataFrame, target_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Predict values for target columns given evidence.

        Args:
            data: Data with some columns (evidence) and missing target columns
            target_columns: Columns to predict (if None, predicts all missing)

        Returns:
            DataFrame with predictions for target columns
        """
        if not self._fitted:
            raise ValueError("Network must be fitted before prediction")

        result = data.copy()

        if target_columns is None:
            target_columns = [
                col for col in self._data_columns if col not in data.columns
            ]

        # Topological sort to ensure parent values are available
        try:
            sorted_nodes = list(nx.topological_sort(self.structure))
        except nx.NetworkXError:
            raise ValueError("Structure must be a DAG (no cycles)")

        # Predict each target in topological order
        for node_name in sorted_nodes:
            if node_name in target_columns:
                node = self.nodes[node_name]
                predictions = node.predict(result)
                result[node_name] = predictions

        return result[target_columns]

    def sample(
        self, n_samples: int = 1, evidence: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Generate samples from the Bayesian Network.

        Uses ancestral sampling: sample nodes in topological order.

        Args:
            n_samples: Number of samples to generate
            evidence: Optional dict of {node_name: value} for conditioning

        Returns:
            DataFrame with generated samples
        """
        if not self._fitted:
            raise ValueError("Network must be fitted before sampling")

        samples = pd.DataFrame()

        # Topological sort
        try:
            sorted_nodes = list(nx.topological_sort(self.structure))
        except nx.NetworkXError:
            raise ValueError("Structure must be a DAG (no cycles)")

        # Apply evidence if provided
        if evidence:
            for node_name, value in evidence.items():
                samples[node_name] = [value] * n_samples

        # Sample each node in topological order
        for node_name in sorted_nodes:
            if evidence and node_name in evidence:
                continue  # Skip evidence nodes

            node = self.nodes[node_name]
            node_samples = node.sample(n_samples, data=samples)
            samples[node_name] = node_samples

        return samples

    def __str__(self):
        """String representation."""
        if self._fitted:
            n_nodes = len(self.nodes)
            n_edges = len(self.structure.edges)
            n_discrete = len(self.discrete_columns)
            n_continuous = len(self.continuous_columns)
            n_custom = len(self.node_classifiers) + len(self.node_regressors)
            return (
                f"CompositeBayesianNetwork(nodes={n_nodes}, edges={n_edges}, "
                f"discrete={n_discrete}, continuous={n_continuous}, "
                f"custom_models={n_custom})"
            )
        else:
            return "CompositeBayesianNetwork(unfitted)"
