"""Bootstrap pareado de la log loss: ¿la diferencia entre dos modelos es real o ruido?

Pareado = en cada remuestreo se eligen LOS MISMOS partidos para ambos modelos, así la
dificultad de cada partido se cancela y solo queda la diferencia entre modelos.
"""
import numpy as np
import pandas as pd

from src.config import RANDOM_STATE
from src.evaluation.metrics import log_loss_por_partido


def bootstrap_pareado(y, probs_modelo, probs_ref, n_boot: int = 1000,
                      seed: int = RANDOM_STATE, nivel: float = 0.95) -> dict:
    """Diferencia de log loss (modelo - referencia) e intervalo percentil.
    Diferencia negativa = el modelo es mejor que la referencia. Si el intervalo contiene el 0,
    la diferencia no es significativa."""
    d = log_loss_por_partido(y, probs_modelo) - log_loss_por_partido(y, probs_ref)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    medias = d[idx].mean(axis=1)
    alfa = (1 - nivel) / 2
    bajo, alto = np.quantile(medias, [alfa, 1 - alfa])
    return {
        "diferencia": float(d.mean()),
        "ic_bajo": float(bajo),
        "ic_alto": float(alto),
        "significativa": bool(alto < 0 or bajo > 0),
        "prop_mejor": float((medias < 0).mean()),   # % de remuestreos en que el modelo gana
    }


def tabla_bootstrap(y, probs: dict, ref: str, **kw) -> pd.DataFrame:
    """Una fila por modelo de `probs` (nombre → matriz H, D, A) contra el modelo `ref`."""
    filas = [{"modelo": nombre, **bootstrap_pareado(y, p, probs[ref], **kw)}
             for nombre, p in probs.items() if nombre != ref]
    return pd.DataFrame(filas).sort_values("diferencia").reset_index(drop=True)
