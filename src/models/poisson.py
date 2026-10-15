"""Poisson simple: goles esperados de cada equipo (desde elo_diff) → matriz de marcadores → H/D/A."""
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

MAX_GOLES = 10


def probs_hda(lam_local: float, lam_visit: float, max_goles: int = MAX_GOLES) -> np.ndarray:
    """Probabilidades [H, D, A] (en ese orden) a partir de los goles esperados de cada equipo,
    suponiendo goles independientes."""
    k = np.arange(max_goles + 1)
    matriz = np.outer(poisson.pmf(k, lam_local), poisson.pmf(k, lam_visit))  # filas: local; columnas: visitante
    p = np.array([np.tril(matriz, -1).sum(), np.trace(matriz), np.triu(matriz, 1).sum()])
    return p / p.sum()


class PoissonElo:
    def fit(self, df: pd.DataFrame):
        X = df[["elo_diff"]] / 100
        self.local_ = PoissonRegressor(alpha=0).fit(X, df["home_goals"])
        self.visit_ = PoissonRegressor(alpha=0).fit(X, df["away_goals"])
        return self

    def expected_goals(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = df[["elo_diff"]] / 100
        return self.local_.predict(X), self.visit_.predict(X)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Ya en orden H, D, A: no hace falta to_hda()."""
        lam_l, lam_v = self.expected_goals(df)
        return np.array([probs_hda(a, b) for a, b in zip(lam_l, lam_v)])