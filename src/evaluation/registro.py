"""Registro de resultados en validación: una fila por modelo en un CSV compartido."""
from pathlib import Path

import pandas as pd

from src.config import ROOT

RUTA_DEFECTO = ROOT / "docs" / "resultados_validacion.csv"
COLUMNAS = ["modelo", "n", "log_loss", "brier", "accuracy"]


def guardar_resultado(res: dict, ruta=RUTA_DEFECTO) -> pd.DataFrame:
    """Agrega o reemplaza (por nombre de modelo) la fila `res`, tal como la devuelve
    `evaluate()`. Devuelve la tabla completa ya guardada."""
    ruta = Path(ruta)
    fila = pd.DataFrame([{c: res[c] for c in COLUMNAS}])
    if ruta.exists():
        tabla = pd.read_csv(ruta)
        tabla = tabla[tabla["modelo"] != res["modelo"]]
        tabla = pd.concat([tabla, fila], ignore_index=True)
    else:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        tabla = fila
    tabla.to_csv(ruta, index=False)
    return tabla
