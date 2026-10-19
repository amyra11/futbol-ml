"""Calibración: ¿cuando el modelo dice 40 %, el resultado ocurre el 40 % de las veces?

Los bins son por cuantiles de la probabilidad predicha (misma cantidad de partidos en cada uno),
así ninguno queda vacío aunque el modelo casi nunca dé probabilidades extremas.
"""
import numpy as np
import pandas as pd

from src.config import CLASSES, RANDOM_STATE


def datos_calibracion(y, probs, clase: str, n_bins: int = 10) -> pd.DataFrame:
    """Para la clase `clase` (H, D o A): por bin, probabilidad media predicha, frecuencia
    observada del resultado y n. Si hay probabilidades repetidas puede haber menos bins."""
    p = np.asarray(probs, dtype=float)[:, CLASSES.index(clase)]
    ocurrio = (np.asarray(y) == clase).astype(float)
    bordes = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    if len(bordes) < 2:                                   # todas iguales (p. ej. el baseline)
        bins = np.zeros(len(p), dtype=int)
    else:
        bins = pd.cut(p, bordes, labels=False, include_lowest=True)
    return (pd.DataFrame({"bin": bins, "p": p, "ocurrio": ocurrio})
              .groupby("bin")
              .agg(prob_media=("p", "mean"), frec_observada=("ocurrio", "mean"), n=("p", "size"))
              .reset_index())


def ece_por_clase(y, probs, n_bins: int = 10) -> dict:
    """Error de calibración esperado de cada clase: media de |prob_media - frec_observada|
    ponderada por los partidos de cada bin (0 = perfecto)."""
    out = {}
    for c in CLASSES:
        d = datos_calibracion(y, probs, c, n_bins)
        out[c] = float(np.sum(d["n"] * (d["prob_media"] - d["frec_observada"]).abs()) / d["n"].sum())
    return out


def ece(y, probs, n_bins: int = 10) -> float:
    """ECE promediado sobre H, D y A."""
    return float(np.mean(list(ece_por_clase(y, probs, n_bins).values())))


def ece_por_azar(probs, n_bins: int = 10, n_sim: int = 500, seed: int = RANDOM_STATE) -> np.ndarray:
    """ECE de un modelo PERFECTAMENTE calibrado con estas mismas probabilidades: se sortean los
    resultados con `probs` y se mide el ECE, `n_sim` veces. Con pocos partidos por bin el ECE
    nunca es 0; esto dice cuánto sale solo por azar."""
    probs = np.asarray(probs, dtype=float)
    rng = np.random.default_rng(seed)
    acum = probs.cumsum(axis=1)
    sims = []
    for _ in range(n_sim):
        idx = (rng.random(len(probs))[:, None] > acum).sum(axis=1)
        sims.append(ece(np.array(CLASSES)[np.minimum(idx, 2)], probs, n_bins))
    return np.array(sims)
