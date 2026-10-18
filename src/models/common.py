"""Utilidades comunes a todos los modelos de clasificación H/D/A.

XGBClassifier exige etiquetas numéricas (0, 1, 2), así que su `classes_` es [0, 1, 2] y
`to_hda()` no encuentra "H", "D", "A". `proba_hda()` funciona con ambos tipos de etiqueta.
"""
import numpy as np

from src.config import CLASSES

LABEL_A_INT = {"H": 0, "D": 1, "A": 2}
INT_A_LABEL = {v: k for k, v in LABEL_A_INT.items()}


def proba_hda(model, X) -> np.ndarray:
    """predict_proba de `model` con las columnas en orden H, D, A.
    Sirve para modelos (o Pipelines) entrenados con etiquetas de texto ("A", "D", "H") o con
    las numéricas de LABEL_A_INT (0, 1, 2)."""
    probs = np.asarray(model.predict_proba(X))
    clases = [c if isinstance(c, str) else INT_A_LABEL[int(c)] for c in model.classes_]
    idx = [clases.index(c) for c in CLASSES]
    return probs[:, idx]
