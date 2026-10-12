"""Baseline: predecir siempre la frecuencia histórica de H/D/A del entrenamiento."""
import numpy as np
import pandas as pd

from src.config import CLASSES


class FrequencyBaseline:
    def fit(self, X, y):
        freq = pd.Series(y).value_counts(normalize=True)
        self.probs_ = np.array([freq.get(c, 0.0) for c in CLASSES])
        return self

    def predict_proba(self, X):
        return np.tile(self.probs_, (len(X), 1))
