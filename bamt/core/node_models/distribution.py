from abc import ABC, abstractmethod

import numpy as np


class Distribution(ABC):

    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def sample(self, num_samples: int) -> np.ndarray:
        pass
