from .prediction_model import PredictionModel

import numpy as np


class Classifier(PredictionModel):
    def __init__(self, classifier=None, **parameters):
        self._classifier = classifier
        self._parameters = parameters

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self._classifier is None:
            # TODO: implement an algorithm that finds a classifier and fits it with chosen parameters
            pass
        else:
            self._classifier.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._classifier.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._classifier.predict_proba(X)

    def __str__(self):
        return str(self._classifier)

    def __getattr__(self, name: str):
        if self._classifier:
            return getattr(self._classifier, name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )
