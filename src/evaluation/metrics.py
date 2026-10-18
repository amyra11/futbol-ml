"""Métricas comunes para TODOS los modelos. Probabilidades siempre en orden H, D, A."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from src.config import CLASSES


def _onehot(y) -> np.ndarray:
    y = np.asarray(y)
    return np.stack([(y == c).astype(float) for c in CLASSES], axis=1)


def log_loss_por_partido(y, probs, eps: float = 1e-15) -> np.ndarray:
    """-log(probabilidad asignada al resultado real), una por partido (columnas H, D, A)."""
    p = np.clip(np.asarray(probs, dtype=float), eps, 1)
    return -np.log(np.sum(p * _onehot(y), axis=1))


def log_loss_hda(y, probs, eps: float = 1e-15) -> float:
    """Log loss con columnas en orden H, D, A.
    (sklearn.log_loss asume orden alfabético A, D, H: no usarlo directamente.)"""
    return float(np.mean(log_loss_por_partido(y, probs, eps)))


def to_hda(model, probs) -> np.ndarray:
    """Reordena predict_proba de sklearn/xgboost (orden de model.classes_) a H, D, A."""
    idx = [list(model.classes_).index(c) for c in CLASSES]
    return np.asarray(probs)[:, idx]


def brier_multiclass(y, probs) -> float:
    """Brier multiclase: media de la suma de errores cuadrados (0 = perfecto, 2 = peor)."""
    return float(np.mean(np.sum((np.asarray(probs) - _onehot(y)) ** 2, axis=1)))


def implied_probs(odds: pd.DataFrame) -> np.ndarray:
    """Probabilidades implícitas de cuotas decimales (odds_H, odds_D, odds_A),
    normalizadas para quitar el margen de la casa."""
    inv = 1.0 / odds[["odds_H", "odds_D", "odds_A"]].to_numpy(dtype=float)
    return inv / inv.sum(axis=1, keepdims=True)


def evaluate(y, probs, name: str = "modelo") -> dict:
    probs = np.asarray(probs, dtype=float)
    assert probs.shape[1] == 3, "probs debe tener 3 columnas H, D, A"
    assert np.allclose(probs.sum(axis=1), 1, atol=1e-6), "cada fila debe sumar 1"
    pred = np.array(CLASSES)[probs.argmax(axis=1)]
    return {
        "modelo": name,
        "n": len(probs),
        "log_loss": log_loss_hda(y, probs),
        "brier": brier_multiclass(y, probs),
        "accuracy": accuracy_score(y, pred),
    }


def confusion(y, probs) -> pd.DataFrame:
    pred = np.array(CLASSES)[np.asarray(probs).argmax(axis=1)]
    cm = confusion_matrix(y, pred, labels=CLASSES)
    return pd.DataFrame(cm, index=[f"real_{c}" for c in CLASSES],
                        columns=[f"pred_{c}" for c in CLASSES])


def compare(results: list[dict]) -> pd.DataFrame:
    """Tabla resumen ordenada por log loss (menor es mejor)."""
    return pd.DataFrame(results).sort_values("log_loss").reset_index(drop=True)
