import pandas as pd
import pytest

from src.evaluation import corrida_test as ct
from src.features.build import build_features

GRID_RF = {"bosque__max_depth": [2], "bosque__min_samples_leaf": [10]}
GRID_XGB = {"max_depth": [2], "learning_rate": [0.1], "n_estimators": [10]}


def test_ensayo_runs_all_models_and_never_uses_test(matches, tmp_path):
    f = build_features(matches)
    tabla = ct.correr(f, "ensayo", salida=tmp_path, grid_rf=GRID_RF, grid_xgb=GRID_XGB, n_boot=20)
    assert set(tabla["modelo"]) == {"baseline_frecuencias", "cuotas_mercado", "poisson_elo", "logistica_elo",
                                    "logistica_todas", ct.FINAL, "random_forest", "xgboost"}
    assert tabla["log_loss"].notna().all()
    assert (tmp_path / "resultados_ensayo.csv").exists()
    assert not (tmp_path / ct.ARCHIVO_CONTROL).exists()          # el ensayo no deja el archivo de control


def test_test_mode_refuses_if_results_already_exist(matches, tmp_path):
    (tmp_path / ct.ARCHIVO_CONTROL).write_text("modelo\n")
    with pytest.raises(RuntimeError, match="UNA sola vez"):
        ct.correr(build_features(matches), "test", salida=tmp_path)


def test_cli_test_without_confirmation_does_not_touch_test(monkeypatch, tmp_path):
    monkeypatch.setattr(ct, "SALIDA_TEST", tmp_path)
    monkeypatch.setattr(ct.pd, "read_csv", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr("builtins.input", lambda _: "no")
    assert ct.main(["--test"]) == 1
    assert not (tmp_path / ct.ARCHIVO_CONTROL).exists()


def test_cli_test_refuses_when_already_used(monkeypatch, tmp_path):
    (tmp_path / ct.ARCHIVO_CONTROL).write_text("modelo\n")
    monkeypatch.setattr(ct, "SALIDA_TEST", tmp_path)
    monkeypatch.setattr(ct.pd, "read_csv", lambda *a, **k: pd.DataFrame())
    assert ct.main(["--test"]) == 1