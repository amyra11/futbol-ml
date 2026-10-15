"""Poisson simple: goles esperados de cada equipo (desde elo_diff) → matriz de marcadores → H/D/A."""
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from src.config import ELO_HOME_ADV

MAX_GOLES = 10


def score_matrix(lam_local: float, lam_visit: float, max_goles: int = MAX_GOLES) -> np.ndarray:
    """Probabilidad de cada marcador. Filas: goles del local; columnas: goles del visitante."""
    k = np.arange(max_goles + 1)
    return np.outer(poisson.pmf(k, lam_local), poisson.pmf(k, lam_visit))


def probs_hda(lam_local: float, lam_visit: float, max_goles: int = MAX_GOLES) -> np.ndarray:
    """Probabilidades [H, D, A] (en ese orden) a partir de los goles esperados de cada equipo,
    suponiendo goles independientes."""
    matriz = score_matrix(lam_local, lam_visit, max_goles)
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

    def predict_teams(self, elo_home: float, elo_away: float) -> tuple[float, float]:
        """Goles esperados (local, visitante) para un partido dado por los Elo de ambos equipos."""
        diff = elo_home + ELO_HOME_ADV - elo_away
        lam_l, lam_v = self.expected_goals(pd.DataFrame({"elo_diff": [diff]}))
        return float(lam_l[0]), float(lam_v[0])