from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders = {}
    
    def fit(self, X, y=None):
        for column in X.columns:
            le = LabelEncoder()
            le.fit(X[column])
            self.encoders[column] = le
        return self
    
    def transform(self, X):
        X = X.copy()
        for column in X.columns:
            le = self.encoders[column]
            X[column] = le.transform(X[column])
        return X
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

