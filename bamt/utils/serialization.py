"""
Serialization utilities for BAMT 2.0.0 Bayesian Networks.

Provides save/load functionality for BN models using a combination of
JSON (for structure and metadata) and pickle (for trained models).
"""

import json
import pickle
from pathlib import Path
from typing import Union, Dict, Any
import networkx as nx


class BayesianNetworkSerializer:
    """Serializer for Bayesian Network models."""

    @staticmethod
    def save(bn: "BayesianNetwork", filepath: Union[str, Path]) -> None:
        """
        Save a Bayesian Network to a file.

        Args:
            bn: Bayesian Network instance to save
            filepath: Path to save the model (without extension)

        The method saves two files:
            - {filepath}.json: Structure and metadata
            - {filepath}.pkl: Trained node models
        """
        filepath = Path(filepath)

        # Prepare structure data (JSON-serializable)
        structure_data = {
            "type": bn.__class__.__name__,
            "structure": {
                "nodes": list(bn.structure.nodes()),
                "edges": list(bn.structure.edges()),
            },
        }

        # Add type-specific data
        if hasattr(bn, "discrete_columns") and bn.discrete_columns is not None:
            structure_data["discrete_columns"] = bn.discrete_columns
        if hasattr(bn, "continuous_columns") and bn.continuous_columns is not None:
            structure_data["continuous_columns"] = bn.continuous_columns

        # Save structure as JSON
        json_path = filepath.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(structure_data, f, indent=2)

        # Save node models as pickle
        models_data = {
            "nodes_dict": bn.nodes_dict,
        }
        pkl_path = filepath.with_suffix(".pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(models_data, f)

    @staticmethod
    def load(filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a Bayesian Network from a file.

        Args:
            filepath: Path to the saved model (without extension)

        Returns:
            Dictionary with structure and models data

        Raises:
            FileNotFoundError: If the model files don't exist
        """
        filepath = Path(filepath)

        # Load structure from JSON
        json_path = filepath.with_suffix(".json")
        if not json_path.exists():
            raise FileNotFoundError(f"Structure file not found: {json_path}")

        with open(json_path, "r") as f:
            structure_data = json.load(f)

        # Load models from pickle
        pkl_path = filepath.with_suffix(".pkl")
        if not pkl_path.exists():
            raise FileNotFoundError(f"Models file not found: {pkl_path}")

        with open(pkl_path, "rb") as f:
            models_data = pickle.load(f)

        # Reconstruct the structure as networkx graph
        structure = nx.DiGraph()
        structure.add_nodes_from(structure_data["structure"]["nodes"])
        structure.add_edges_from(structure_data["structure"]["edges"])

        return {
            "type": structure_data["type"],
            "structure": structure,
            "nodes_dict": models_data["nodes_dict"],
            "discrete_columns": structure_data.get("discrete_columns"),
            "continuous_columns": structure_data.get("continuous_columns"),
        }


def save_bn(bn: "BayesianNetwork", filepath: Union[str, Path]) -> None:
    """
    Convenience function to save a Bayesian Network.

    Args:
        bn: Bayesian Network instance
        filepath: Path to save (without extension)
    """
    BayesianNetworkSerializer.save(bn, filepath)


def load_bn(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Convenience function to load a Bayesian Network.

    Args:
        filepath: Path to the saved model (without extension)

    Returns:
        Dictionary with loaded data
    """
    return BayesianNetworkSerializer.load(filepath)
