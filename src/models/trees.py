"""Modelos de árboles: Random Forest y XGBoost, con búsqueda pequeña de hiperparámetros.

Igual que en la logística, el ajuste usa TimeSeriesSplit SOLO dentro de train (ordenado por
fecha) y `.best_estimator_` queda reajustado con todo train. Las probabilidades se sacan con
`proba_hda()` (XGBoost usa etiquetas 0, 1, 2).
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import RANDOM_STATE
from src.models.common import LABEL_A_INT

RF_GRID = {
    "bosque__max_depth": [3, 5, 8],
    "bosque__min_samples_leaf": [20, 50, 100],
}

XGB_GRID = {
    "max_depth": [2, 3, 4],
    "learning_rate": [0.03, 0.1],
    "n_estimators": [200, 400],
}


def make_random_forest(**params) -> Pipeline:
    """Pipeline sin ajustar: mediana para NaN → Random Forest de 400 árboles."""
    return Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("bosque", RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE,
                                          n_jobs=-1, **params)),
    ])


def make_xgboost(**params) -> XGBClassifier:
    """XGBoost multiclase sin ajustar. Maneja los NaN de inicio de temporada por sí mismo
    (aprende hacia qué rama mandarlos), así que no necesita imputador."""
    return XGBClassifier(objective="multi:softprob", subsample=0.8, eval_metric="mlogloss",
                         tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1, **params)


def _ordenar(train: pd.DataFrame) -> pd.DataFrame:
    return train.sort_values(["date", "match_id"], kind="mergesort")


def tune_random_forest(train: pd.DataFrame, cols: list[str], target: str = "result",
                       grid: dict = RF_GRID, n_splits: int = 5) -> GridSearchCV:
    train = _ordenar(train)
    search = GridSearchCV(make_random_forest(), grid, cv=TimeSeriesSplit(n_splits=n_splits),
                          scoring="neg_log_loss")
    return search.fit(train[cols], train[target])


def tune_xgboost(train: pd.DataFrame, cols: list[str], target: str = "result",
                 grid: dict = XGB_GRID, n_splits: int = 5) -> GridSearchCV:
    """Las etiquetas se pasan como LABEL_A_INT (H=0, D=1, A=2): XGBoost no acepta texto."""
    train = _ordenar(train)
    search = GridSearchCV(make_xgboost(), grid, cv=TimeSeriesSplit(n_splits=n_splits),
                          scoring="neg_log_loss")
    return search.fit(train[cols], train[target].map(LABEL_A_INT))
