"""Regresión logística multinomial: imputar → escalar → logística, todo en un Pipeline.

Al ir en un Pipeline, la mediana del imputador y la media/desviación del escalador se
calculan SOLO con los datos de `fit` (train, o el pliegue de entrenamiento dentro de
GridSearchCV), nunca con validación o test.
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import CLASSES

C_GRID = np.logspace(-4, 2, 13)

# Nombres legibles para tablas y gráficos
NOMBRES_ES = {
    "home_form_pts": "Puntos por partido del local (últimos 5)",
    "home_form_gf": "Goles a favor del local (últimos 5)",
    "home_form_ga": "Goles en contra del local (últimos 5)",
    "away_form_pts": "Puntos por partido del visitante (últimos 5)",
    "away_form_gf": "Goles a favor del visitante (últimos 5)",
    "away_form_ga": "Goles en contra del visitante (últimos 5)",
    "home_elo": "Elo del local",
    "away_elo": "Elo del visitante",
    "elo_diff": "Diferencia de Elo (local + ventaja - visitante)",
    "home_rest": "Días de descanso del local",
    "away_rest": "Días de descanso del visitante",
    "rest_diff": "Diferencia de descanso (local - visitante)",
}


def make_logistic(C: float = 1.0) -> Pipeline:
    """Pipeline sin ajustar: mediana para NaN → estandarizar → logística multinomial."""
    return Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
        ("logistica", LogisticRegression(C=C, max_iter=2000)),
    ])


def tune_logistic(train: pd.DataFrame, cols: list[str], target: str = "result",
                  Cs=C_GRID, n_splits: int = 5) -> GridSearchCV:
    """Elige C con TimeSeriesSplit SOLO dentro de train (ordenado por fecha) y
    reajusta el mejor pipeline con todo train (`.best_estimator_`)."""
    train = train.sort_values(["date", "match_id"], kind="mergesort")
    search = GridSearchCV(make_logistic(), {"logistica__C": list(Cs)},
                          cv=TimeSeriesSplit(n_splits=n_splits),
                          scoring="neg_log_loss")
    return search.fit(train[cols], train[target])


def coef_table(model: Pipeline, cols: list[str]) -> pd.DataFrame:
    """Coeficientes por clase en orden H, D, A (escala: por desviación estándar),
    con los nombres de variables en español."""
    logit = model.named_steps["logistica"]
    idx = [list(logit.classes_).index(c) for c in CLASSES]
    return pd.DataFrame(logit.coef_[idx].T, columns=CLASSES,
                        index=[NOMBRES_ES.get(c, c) for c in cols])
