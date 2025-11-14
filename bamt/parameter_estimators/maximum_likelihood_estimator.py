"""
Maximum Likelihood Estimator for Bayesian Network parameters.

This module implements parameter estimation for Bayesian Networks using
maximum likelihood estimation (MLE).
"""

from typing import Dict, Any

import networkx as nx
import pandas as pd

from .parameters_estimator import ParametersEstimator


class MaximumLikelihoodEstimator(ParametersEstimator):
    """
    Maximum Likelihood Estimator for Bayesian Network parameters.

    This estimator fits node parameters by maximizing the likelihood
    of the observed data. For discrete nodes, this means computing
    conditional probability tables (CPTs). For continuous nodes,
    this means fitting distributions or regression models.

    Attributes:
        method: Estimation method ('MLE' or variants)
        **kwargs: Additional parameters for estimation
    """

    def __init__(self, method: str = "MLE", **kwargs):
        """
        Initialize Maximum Likelihood Estimator.

        Args:
            method: Estimation method (default: 'MLE')
            **kwargs: Additional parameters
        """
        super().__init__()
        self.method = method
        self.kwargs = kwargs

    def estimate(
        self,
        data: pd.DataFrame,
        structure: nx.DiGraph,
        nodes: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Estimate parameters for all nodes using MLE.

        For each node in the network:
        - If it's a root node (no parents): fit marginal distribution
        - If it has parents: fit conditional distribution/model

        Args:
            data: Training data
            structure: DAG structure (NetworkX DiGraph)
            nodes: Dictionary mapping node names to node objects
            **kwargs: Additional parameters

        Returns:
            Dictionary mapping node names to fitted parameters
        """
        parameters = {}

        for node_name in structure.nodes():
            node = nodes.get(node_name)
            if node is None:
                continue

            # Get parent names from structure
            parents = list(structure.predecessors(node_name))

            # Select relevant columns
            if parents:
                cols = [node_name] + parents
            else:
                cols = [node_name]

            node_data = data[cols]

            # Fit the node
            node.fit(node_data)

            # Store fitted parameters (in this case, the fitted node itself)
            parameters[node_name] = {
                "node": node,
                "parents": parents,
                "fitted": True,
            }

        return parameters

    def __str__(self):
        return f"MaximumLikelihoodEstimator(method={self.method})"
