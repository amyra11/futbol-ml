"""Valores SHAP del modelo final. shap y matplotlib no están en el CI: estos tests se saltan allí."""
import numpy as np
import pandas as pd
import pytest

shap = pytest.importorskip("shap")

from src.config import CLASSES
from src.data.clean import clean_season, finalize
from src.evaluation.interpretabilidad import (explicador_shap, importancia_global,
                                              shap_clase_predicha, valor_base, valores_shap)
from src.features.build import build_features
from src.models.final import FEATURES_FINAL, ModeloFinal
from src.models.logistic import NOMBRES_ES, tune_logistic
from src.splits import split
from tests.conftest import make_raw_season


@pytest.fixture(scope="module")
def datos():
    """7 temporadas sintéticas; el modelo se entrena SOLO con train y se explica validación."""
    seasons = [(2017 + i, f"{17 + i:02d}{18 + i:02d}") for i in range(7)]
    frames = [clean_season(make_raw_season(y, i), s) for i, (y, s) in enumerate(seasons)]
    features = build_features(finalize(pd.concat(frames, ignore_index=True)))
    train, val = split(features)
    modelo = ModeloFinal(tune_logistic(train, FEATURES_FINAL).best_estimator_, FEATURES_FINAL)
    explicador = explicador_shap(modelo, train)
    return modelo, explicador, train, val


def test_shap_coincide_con_factores_en_la_clase_mas_probable(datos):
    """Fondo = train (los datos del escalador) → media del fondo 0 → SHAP = coef × z = factores()."""
    modelo, explicador, _, val = datos
    sv = valores_shap(modelo, explicador, val)
    pred = shap_clase_predicha(modelo, sv, val)
    a_col = {v: k for k, v in NOMBRES_ES.items()}
    for i in range(len(val)):
        fac = modelo.factores(val.iloc[[i]], top=len(FEATURES_FINAL))
        de_factores = pd.Series({a_col[f["variable"]]: f["impacto"] for f in fac})
        assert np.allclose(pred.iloc[i][de_factores.index], de_factores, atol=1e-8)


def test_fondo_submuestreado_ya_no_coincide(datos):
    """Con el masker por defecto de shap (100 filas al azar) la media del fondo no es 0."""
    modelo, explicador, train, val = datos
    assert len(train) > 100
    Z = modelo.pipeline[:-1].transform(train[modelo.cols])
    defecto = shap.LinearExplainer(modelo.pipeline.named_steps["logistica"], Z)
    diferencia = np.abs(valores_shap(modelo, defecto, val) - valores_shap(modelo, explicador, val))
    assert diferencia.max() > 1e-6


def test_orden_hda_y_suma_igual_al_puntaje(datos):
    """Columna j = clase CLASSES[j]; valor base + suma de SHAP = puntaje lineal de cada clase."""
    modelo, explicador, _, val = datos
    sv = valores_shap(modelo, explicador, val)
    assert sv.shape == (len(val), len(FEATURES_FINAL), 3)
    logit = modelo.pipeline.named_steps["logistica"]
    Z = modelo.pipeline[:-1].transform(val[FEATURES_FINAL])
    puntaje = logit.decision_function(Z)
    for j, c in enumerate(CLASSES):
        k = list(logit.classes_).index(c)
        assert np.allclose(valor_base(modelo, explicador)[j] + sv[:, :, j].sum(axis=1), puntaje[:, k])
        assert np.allclose(sv[:, :, j], logit.coef_[k] * Z, atol=1e-8)


def test_importancia_global_ordenada(datos):
    modelo, explicador, _, val = datos
    pred = shap_clase_predicha(modelo, valores_shap(modelo, explicador, val), val)
    imp = importancia_global(pred)
    assert set(imp.index) == set(FEATURES_FINAL)
    assert (imp >= 0).all() and imp.is_monotonic_decreasing


def test_figuras_shap_se_guardan(datos, tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from src.evaluation.figuras import fig_shap_ejemplo, fig_shap_global

    modelo, explicador, _, val = datos
    pred = shap_clase_predicha(modelo, valores_shap(modelo, explicador, val), val)
    fig_shap_global(importancia_global(pred), tmp_path / "global.png", len(val))
    fila = val.iloc[[0]]
    fig_shap_ejemplo(pred.iloc[0], fila[FEATURES_FINAL].iloc[0], modelo.predict_proba(fila)[0],
                     "Partido de prueba", "gana el local", tmp_path / "ejemplo.png", top=4)
    assert (tmp_path / "global.png").stat().st_size > 0
    assert (tmp_path / "ejemplo.png").stat().st_size > 0
