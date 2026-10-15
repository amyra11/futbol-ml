import pandas as pd
import pytest

from src.evaluation.registro import COLUMNAS, guardar_resultado


def _res(modelo, ll):
    return {"modelo": modelo, "n": 380, "log_loss": ll, "brier": 0.6, "accuracy": 0.5}


def test_crea_archivo_y_agrega_filas(tmp_path):
    ruta = tmp_path / "sub" / "res.csv"
    guardar_resultado(_res("a", 1.0), ruta)
    guardar_resultado(_res("b", 0.9), ruta)
    tabla = pd.read_csv(ruta)
    assert list(tabla.columns) == COLUMNAS
    assert list(tabla["modelo"]) == ["a", "b"]


def test_reemplaza_por_nombre_sin_duplicar(tmp_path):
    ruta = tmp_path / "res.csv"
    guardar_resultado(_res("a", 1.0), ruta)
    guardar_resultado(_res("b", 0.9), ruta)
    guardar_resultado(_res("a", 0.95), ruta)
    tabla = pd.read_csv(ruta)
    assert len(tabla) == 2
    assert tabla.loc[tabla["modelo"] == "a", "log_loss"].item() == pytest.approx(0.95)


def test_acepta_salida_de_evaluate(tmp_path):
    import numpy as np
    from src.evaluation.metrics import evaluate
    res = evaluate(["H", "D", "A"], np.full((3, 3), 1 / 3), "uniforme")
    guardar_resultado(res, tmp_path / "res.csv")
    assert pd.read_csv(tmp_path / "res.csv").loc[0, "modelo"] == "uniforme"
