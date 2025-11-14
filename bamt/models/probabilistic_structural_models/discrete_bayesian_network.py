"""
Discrete Bayesian Network implementation.

This module implements a Bayesian Network for discrete (categorical) data.
"""

from typing import Dict, List, Optional

import networkx as nx
import pandas as pd

from bamt.core.nodes.child_nodes.conditional_discrete_node import (
    ConditionalDiscreteNode,
)
from bamt.core.nodes.root_nodes.discrete_node import DiscreteNode
from bamt.parameter_estimators.maximum_likelihood_estimator import (
    MaximumLikelihoodEstimator,
)

from .bayesian_network import BayesianNetwork


class DiscreteBayesianNetwork(BayesianNetwork):
    """
    Discrete Bayesian Network.

    A Bayesian Network for modeling discrete/categorical data.
    Nodes use empirical distributions (root nodes) or classification
    models (conditional nodes).

    Attributes:
        structure: DAG structure (NetworkX DiGraph)
        nodes: Dictionary mapping node names to node objects
        _fitted: Whether the network has been fitted
    """

    def __init__(self, structure: Optional[nx.DiGraph] = None):
        """
        Initialize Discrete Bayesian Network.

        Args:
            structure: Optional DAG structure (NetworkX DiGraph)
        """
        super().__init__()
        self.structure = structure if structure else nx.DiGraph()
        self.nodes = {}
        self._fitted = False
        self._data_columns = []

    def fit(
        self,
        data: pd.DataFrame,
        structure: Optional[nx.DiGraph] = None,
        parameter_estimator: Optional[MaximumLikelihoodEstimator] = None,
        **kwargs
    ) -> "DiscreteBayesianNetwork":
        """
        Fit the Discrete Bayesian Network.

        Args:
            data: Training data (discrete/categorical columns)
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

        self._data_columns = list(data.columns)

        # Create nodes based on structure
        for node_name in self.structure.nodes():
            parents = list(self.structure.predecessors(node_name))

            if not parents:
                # Root node - no parents
                node = DiscreteNode(name=node_name)
                node_data = data[[node_name]]
            else:
                # Child node - has parents
                # For discrete BN, all parents are discrete
                node = ConditionalDiscreteNode(
                    name=node_name,
                    disc_parents=parents,
                    cont_parents=[],
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
        status = "fitted" if self._fitted else "not fitted"
        return f"DiscreteBayesianNetwork({n_nodes} nodes, {n_edges} edges, {status})"
