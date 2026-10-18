import numpy as np
import pandas as pd
import pytest

from src.data.clean import clean_season, finalize
from src.features.build import FEATURE_COLS, build_features
from src.features.current import matchup_features
from src.models.final import FEATURES_FINAL, ModeloFinal, cargar_modelo
from src.models.logistic import NOMBRES_ES
from tests.conftest import make_raw_season


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    """7 temporadas sintéticas (2017-18 a 2023-24): suficiente para TimeSeriesSplit(5)."""
    seasons = [(2017 + i, f"{17 + i:02d}{18 + i:02d}") for i in range(7)]
    frames = [clean_season(make_raw_season(y, i), s) for i, (y, s) in enumerate(seasons)]
    return build_features(finalize(pd.concat(frames, ignore_index=True)))


@pytest.fixture(scope="module")
def modelo(features, tmp_path_factory) -> ModeloFinal:
    return cargar_modelo(tmp_path_factory.mktemp("m") / "modelo_final.joblib", features)


def test_sin_columnas_redundantes_de_elo():
    assert "elo_diff" in FEATURES_FINAL
    assert "home_elo" not in FEATURES_FINAL and "away_elo" not in FEATURES_FINAL
    assert set(FEATURES_FINAL) <= set(FEATURE_COLS)


def test_contrato_con_salida_de_matchup_features(features, modelo):
    X = pd.concat([matchup_features(features, "Arsenal", "Chelsea"),
                   matchup_features(features, "Liverpool", "Tottenham", rest_home=3, rest_away=10)],
                  ignore_index=True)
    assert list(X.columns) == FEATURE_COLS          # la app pasa TODAS las FEATURE_COLS
    p = modelo.predict_proba(X)
    assert p.shape == (2, 3)
    assert np.allclose(p.sum(axis=1), 1)
    assert ((p > 0) & (p < 1)).all()


def test_contrato_orden_hda(modelo, features):
    """Columna j de la salida = probabilidad de CLASSES[j] (H, D, A), no el orden alfabético
    de sklearn (A, D, H)."""
    X = matchup_features(features, "Arsenal", "Chelsea")
    crudo = modelo.pipeline.predict_proba(X[FEATURES_FINAL])
    clases = list(modelo.pipeline.classes_)
    assert clases == ["A", "D", "H"]
    p = modelo.predict_proba(X)
    for j, c in enumerate(["H", "D", "A"]):
        assert p[0, j] == pytest.approx(crudo[0, clases.index(c)])


def test_factores_en_espanol_y_ordenados_por_impacto(features, modelo):
    X = matchup_features(features, "Arsenal", "Chelsea")
    fac = modelo.factores(X, top=4)
    assert len(fac) == 4
    assert all(set(f) == {"variable", "valor", "impacto"} for f in fac)
    assert all(f["variable"] in NOMBRES_ES.values() for f in fac)
    impactos = [abs(f["impacto"]) for f in fac]
    assert impactos == sorted(impactos, reverse=True)


def test_factores_suman_el_puntaje_de_la_clase_mas_probable(features, modelo):
    """Todos los impactos + intercepto = puntaje lineal de la clase predicha."""
    X = matchup_features(features, "Arsenal", "Chelsea")
    fac = modelo.factores(X, top=len(FEATURES_FINAL))
    logit = modelo.pipeline.named_steps["logistica"]
    clase = ["H", "D", "A"][modelo.predict_proba(X)[0].argmax()]
    k = list(logit.classes_).index(clase)
    puntaje = logit.decision_function(modelo.pipeline[:-1].transform(X[FEATURES_FINAL]))[0, k]
    assert sum(f["impacto"] for f in fac) + logit.intercept_[k] == pytest.approx(puntaje)


def test_cargar_modelo_entrena_si_no_existe_y_luego_carga(features, tmp_path):
    ruta = tmp_path / "sub" / "modelo_final.joblib"
    assert not ruta.exists()
    m1 = cargar_modelo(ruta, features)
    assert ruta.exists()
    assert m1.meta["features"] == FEATURES_FINAL
    assert "2425" not in m1.meta["temporadas"]       # el test nunca se usa
    m2 = cargar_modelo(ruta)                          # segunda vez: carga, no entrena
    X = matchup_features(features, "Arsenal", "Chelsea")
    assert np.allclose(m1.predict_proba(X), m2.predict_proba(X))
    assert m2.meta == m1.meta
