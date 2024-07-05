from .prediction_model import PredictionModel

import numpy as np


class Regressor(PredictionModel):
    def __init__(self, regressor=None, **parameters):
        self._regressor = regressor
        self._parameters = parameters

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self._regressor is None:
            # TODO: implement an algorithm that finds a regressor and fits it with chosen parameters
            pass
        else:
            self._regressor.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._regressor.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._regressor.predict_proba(X)

    def __str__(self):
        return str(self._regressor)
