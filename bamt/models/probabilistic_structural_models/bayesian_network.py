from abc import abstractmethod
from pathlib import Path
from typing import Union, Any

from .probabilistic_structural_model import ProbabilisticStructuralModel


class BayesianNetwork(ProbabilisticStructuralModel):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def sample(self):
        pass

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save the Bayesian Network to a file.

        Args:
            filepath: Path to save the model (without extension).
                     Two files will be created: {filepath}.json and {filepath}.pkl

        Example:
            >>> bn.save("my_network")  # Creates my_network.json and my_network.pkl
        """
        from bamt.utils.serialization import save_bn

        save_bn(self, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "BayesianNetwork":
        """
        Load a Bayesian Network from a file.

        Args:
            filepath: Path to the saved model (without extension)

        Returns:
            Loaded Bayesian Network instance

        Example:
            >>> bn = DiscreteBayesianNetwork.load("my_network")

        Note:
            The class type is inferred from the saved data. You can call this
            method on any BayesianNetwork subclass.
        """
        from bamt.utils.serialization import load_bn
        from bamt.models.probabilistic_structural_models.discrete_bayesian_network import (
            DiscreteBayesianNetwork,
        )
        from bamt.models.probabilistic_structural_models.continuous_bayesian_network import (
            ContinuousBayesianNetwork,
        )
        from bamt.models.probabilistic_structural_models.hybrid_bayesian_network import (
            HybridBayesianNetwork,
        )

        data = load_bn(filepath)

        # Map type names to classes
        type_map = {
            "DiscreteBayesianNetwork": DiscreteBayesianNetwork,
            "ContinuousBayesianNetwork": ContinuousBayesianNetwork,
            "HybridBayesianNetwork": HybridBayesianNetwork,
        }

        bn_class = type_map.get(data["type"], cls)

        # Create instance
        if data["type"] == "HybridBayesianNetwork":
            bn = bn_class(
                structure=data["structure"],
                discrete_columns=data["discrete_columns"],
                continuous_columns=data["continuous_columns"],
            )
        else:
            bn = bn_class(structure=data["structure"])

        # Restore nodes
        bn.nodes_dict = data["nodes_dict"]

        return bn

    def plot(
        self, output_path: Union[str, Path], mode: str = "interactive", **kwargs: Any
    ) -> None:
        """
        Visualize the Bayesian Network structure.

        Args:
            output_path: Path to save the visualization
            mode: Either "interactive" (HTML with pyvis) or "static" (image with matplotlib)
            **kwargs: Additional arguments passed to the visualization function

        Example:
            >>> bn.plot("network.html")  # Interactive HTML
            >>> bn.plot("network.png", mode="static")  # Static image
            >>> bn.plot("network.html", height="600px", width="80%")
        """
        from bamt.utils.visualization import plot_bn

        plot_bn(self, output_path, mode=mode, **kwargs)
