"""Valores SHAP del modelo final (logística con variables imputadas y estandarizadas).

En un modelo lineal, el valor SHAP de la variable i hacia la clase k es

    coef[k, i] × (z_i − media de z_i en el fondo)

donde z es la variable ya imputada y escalada. Si el fondo son los mismos partidos con los que se
ajustó el escalador, esa media es 0 y el SHAP coincide con `ModeloFinal.factores()`.
Los valores están en la escala del puntaje lineal de cada clase (logit), no en probabilidad.
"""
import numpy as np
import pandas as pd

from src.config import CLASSES


def explicador_shap(modelo, fondo: pd.DataFrame):
    """shap.LinearExplainer de la logística con el fondo COMPLETO, imputado y escalado.

    Se pasa un `maskers.Independent` con max_samples = len(fondo) a propósito: si se le da la
    matriz sin más, shap la submuestrea a 100 filas, la media del fondo deja de ser 0 y los
    valores se desplazan respecto de `factores()`."""
    import shap   # import local: shap no está en el entorno del CI

    Z = modelo.pipeline[:-1].transform(fondo[modelo.cols])
    masker = shap.maskers.Independent(Z, max_samples=len(Z))
    return shap.LinearExplainer(modelo.pipeline.named_steps["logistica"], masker)


def _orden_hda(modelo) -> list[int]:
    clases = list(modelo.pipeline.named_steps["logistica"].classes_)
    return [clases.index(c) for c in CLASSES]


def valores_shap(modelo, explicador, X: pd.DataFrame) -> np.ndarray:
    """Matriz (partidos, variables, 3); la última dimensión en orden H, D, A."""
    Z = modelo.pipeline[:-1].transform(X[modelo.cols])
    return np.asarray(explicador.shap_values(Z))[..., _orden_hda(modelo)]


def valor_base(modelo, explicador) -> np.ndarray:
    """Puntaje medio del fondo por clase (H, D, A): punto de partida al que se suman los SHAP."""
    return np.asarray(explicador.expected_value)[_orden_hda(modelo)]


def shap_clase_predicha(modelo, sv: np.ndarray, X: pd.DataFrame) -> pd.DataFrame:
    """SHAP hacia la clase más probable de cada partido (positivo = la favorece).
    Filas = partidos de X, columnas = variables del modelo."""
    k = modelo.predict_proba(X).argmax(axis=1)
    return pd.DataFrame(sv[np.arange(len(k)), :, k], columns=modelo.cols, index=X.index)


def importancia_global(shap_pred: pd.DataFrame) -> pd.Series:
    """Media de |SHAP| por variable, de mayor a menor."""
    return shap_pred.abs().mean().sort_values(ascending=False)
