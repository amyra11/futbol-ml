"""Modelo provisional para la demo: regresión logística sobre elo_diff.
Se sustituye por el modelo final de la Persona B."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config import ELO_HOME_ADV, RANDOM_STATE
from src.evaluation.metrics import to_hda
from src.splits import split


class EloLogistic:
    def fit(self, features: pd.DataFrame):
        train, val = split(features)              # nunca usa la temporada de prueba
        d = pd.concat([train, val])
        self.clf_ = LogisticRegression(random_state=RANDOM_STATE).fit(d[["elo_diff"]] / 100, d["result"])
        return self

    def predict_proba_teams(self, elo_home: float, elo_away: float) -> np.ndarray:
        """Devuelve [P(local), P(empate), P(visitante)] en orden H, D, A."""
        diff = elo_home + ELO_HOME_ADV - elo_away
        X = pd.DataFrame({"elo_diff": [diff / 100]})
        return to_hda(self.clf_, self.clf_.predict_proba(X))[0]