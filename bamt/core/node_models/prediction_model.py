from abc import ABC, abstractmethod

import numpy as np


class PredictionModel(ABC):
    """Represents general prediction model implementations. Each prediction model should provide a fit and a predict
    method."""

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
