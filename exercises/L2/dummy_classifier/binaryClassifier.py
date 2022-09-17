from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class DummyClassifier(BaseEstimator, ClassifierMixin):

    def fit(self):
        print("Fitting...")
        pass

    def predict(self, X: np.array):
        print("Predicting...")
        return np.zeros((len(X), 1), dtype=bool) 