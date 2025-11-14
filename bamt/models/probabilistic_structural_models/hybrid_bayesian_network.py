"""
Hybrid Bayesian Network implementation.

This module implements a Bayesian Network for mixed (discrete and continuous) data.
"""

from typing import Dict, List, Optional

import networkx as nx
import pandas as pd

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


class HybridBayesianNetwork(BayesianNetwork):
    """
    Hybrid Bayesian Network.

    A Bayesian Network for modeling mixed discrete and continuous data.
    Automatically selects appropriate node types based on data types.

    Attributes:
        structure: DAG structure (NetworkX DiGraph)
        nodes: Dictionary mapping node names to node objects
        discrete_columns: Set of discrete column names
        continuous_columns: Set of continuous column names
        _fitted: Whether the network has been fitted
    """

    def __init__(
        self,
        structure: Optional[nx.DiGraph] = None,
        discrete_columns: Optional[List[str]] = None,
        continuous_columns: Optional[List[str]] = None,
    ):
        """
        Initialize Hybrid Bayesian Network.

        Args:
            structure: Optional DAG structure (NetworkX DiGraph)
            discrete_columns: List of discrete column names
            continuous_columns: List of continuous column names
        """
        super().__init__()
        self.structure = structure if structure else nx.DiGraph()
        self.nodes = {}
        self.discrete_columns = set(discrete_columns) if discrete_columns else set()
        self.continuous_columns = set(continuous_columns) if continuous_columns else set()
        self._fitted = False
        self._data_columns = []

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
                if data[col].dtype in ['object', 'category', 'bool']:
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
    ) -> "HybridBayesianNetwork":
        """
        Fit the Hybrid Bayesian Network.

        Args:
            data: Training data (mixed discrete and continuous columns)
            structure: Optional structure (if not set during init)
            parameter_estimator: Parameter estimator (default: MLE)
            **kwargs: Additional parameters

        Returns:
            self
        """
        if structure is not None:
            self.structure = structure

        if self.structure.number_of_nodes() == 0:
            raise ValueError("Network structure is empty. Provide a DAG structure.")

        if parameter_estimator is None:
            parameter_estimator = MaximumLikelihoodEstimator()

        self._data_columns = list(data.columns)

        # Infer column types if not provided
        self._infer_column_types(data)

        # Create nodes based on structure and data types
        for node_name in self.structure.nodes():
            parents = list(self.structure.predecessors(node_name))

            is_discrete_node = node_name in self.discrete_columns

            if not parents:
                # Root node - no parents
                if is_discrete_node:
                    node = DiscreteNode(name=node_name)
                else:
                    node = ContinuousNode(name=node_name)
                node_data = data[[node_name]]
            else:
                # Child node - has parents
                # Separate discrete and continuous parents
                disc_parents = [p for p in parents if p in self.discrete_columns]
                cont_parents = [p for p in parents if p in self.continuous_columns]

                if is_discrete_node:
                    # Discrete child node
                    node = ConditionalDiscreteNode(
                        name=node_name,
                        disc_parents=disc_parents,
                        cont_parents=cont_parents,
                    )
                else:
                    # Continuous child node
                    node = ConditionalContinuousNode(
                        name=node_name,
                        disc_parents=disc_parents,
                        cont_parents=cont_parents,
                    )

                cols = [node_name] + parents
                node_data = data[cols]

            # Fit the node
            node.fit(node_data)
            self.nodes[node_name] = node

        self._fitted = True
        return self

    def predict(
        self,
        data: pd.DataFrame,
        target_columns: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Predict missing values in data.

        Args:
            data: DataFrame with some missing values
            target_columns: Columns to predict (if None, predict all missing)
            **kwargs: Additional parameters

        Returns:
            DataFrame with predictions
        """
        if not self._fitted:
            raise RuntimeError("Network not fitted. Call fit() first.")

        result = data.copy()

        if target_columns is None:
            # Find columns with missing values
            target_columns = [col for col in data.columns if data[col].isna().any()]

        for target_col in target_columns:
            if target_col not in self.nodes:
                continue

            node = self.nodes[target_col]
            parents = list(self.structure.predecessors(target_col))

            # Predict for each row
            for idx in result.index:
                if pd.isna(result.loc[idx, target_col]):
                    # Get parent values
                    parent_vals = {p: result.loc[idx, p] for p in parents}

                    # Predict
                    pred = node.predict(parent_vals)
                    result.loc[idx, target_col] = pred

        return result

    def sample(
        self, n_samples: int = 1, evidence: Optional[Dict[str, any]] = None, **kwargs
    ) -> pd.DataFrame:
        """
        Sample from the Bayesian Network.

        Uses ancestral sampling (forward sampling from topological order).

        Args:
            n_samples: Number of samples to generate
            evidence: Optional evidence (fixed values for some nodes)
            **kwargs: Additional parameters

        Returns:
            DataFrame with sampled data
        """
        if not self._fitted:
            raise RuntimeError("Network not fitted. Call fit() first.")

        evidence = evidence if evidence else {}

        # Get topological order
        topo_order = list(nx.topological_sort(self.structure))

        samples = []
        for _ in range(n_samples):
            sample = {}

            for node_name in topo_order:
                if node_name in evidence:
                    # Use evidence
                    sample[node_name] = evidence[node_name]
                else:
                    # Sample from node
                    node = self.nodes[node_name]
                    parents = list(self.structure.predecessors(node_name))

                    if not parents:
                        # Root node
                        sampled_val = node.sample()
                    else:
                        # Child node - get parent values
                        parent_vals = {p: sample[p] for p in parents}
                        sampled_val = node.sample(parent_vals)

                    sample[node_name] = sampled_val

            samples.append(sample)

        return pd.DataFrame(samples)

    def get_structure(self) -> nx.DiGraph:
        """Get the network structure."""
        return self.structure

    def set_structure(self, structure: nx.DiGraph):
        """Set the network structure."""
        self.structure = structure
        self._fitted = False  # Need to refit with new structure

    def __str__(self):
        n_nodes = self.structure.number_of_nodes()
        n_edges = self.structure.number_of_edges()
        n_disc = len(self.discrete_columns)
        n_cont = len(self.continuous_columns)
        status = "fitted" if self._fitted else "not fitted"
        return f"HybridBayesianNetwork({n_nodes} nodes, {n_edges} edges, {n_disc} discrete, {n_cont} continuous, {status})"
