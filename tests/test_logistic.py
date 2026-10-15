import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import evaluate, to_hda
from src.features.build import FEATURE_COLS, build_features
from src.models.logistic import coef_table, make_logistic, tune_logistic
from src.splits import split


def _toy(n_por_clase=50, seed=0):
    """x bajo → H, x medio → D, x alto → A (orden distinto al alfabético a propósito)."""
    rng = np.random.default_rng(seed)
    x = np.concatenate([rng.normal(m, 0.5, n_por_clase) for m in (-3, 0, 3)])
    ruido = rng.normal(size=3 * n_por_clase)
    X = pd.DataFrame({"x": x, "ruido": ruido})
    y = np.array(["H"] * n_por_clase + ["D"] * n_por_clase + ["A"] * n_por_clase)
    return X, y


def test_probs_en_orden_hda_y_suman_1():
    X, y = _toy()
    model = make_logistic().fit(X, y)
    p = to_hda(model, model.predict_proba(pd.DataFrame({"x": [-3, 0, 3], "ruido": 0})))
    assert np.allclose(p.sum(axis=1), 1)
    assert list(p.argmax(axis=1)) == [0, 1, 2]  # H, D, A


def test_predice_con_nan_en_train_y_en_prediccion():
    X, y = _toy()
    X.loc[::7, "x"] = np.nan
    X.loc[::5, "ruido"] = np.nan
    model = make_logistic().fit(X, y)
    X_new = pd.DataFrame({"x": [np.nan, -3, np.nan], "ruido": [0, np.nan, np.nan]})
    p = to_hda(model, model.predict_proba(X_new))
    assert np.isfinite(p).all()
    assert np.allclose(p.sum(axis=1), 1)


def test_pipeline_con_features_reales_y_evaluate(matches):
    """Con features construidas (tienen NaN al inicio de temporada) y split temporal."""
    feats = build_features(matches)
    train, val = split(feats)
    assert train[FEATURE_COLS].isna().any().any(), "el caso de prueba debe tener NaN"
    model = make_logistic().fit(train[FEATURE_COLS], train["result"])
    p = to_hda(model, model.predict_proba(val[FEATURE_COLS]))
    res = evaluate(val["result"], p, "logistica")
    assert res["n"] == len(val)


def test_imputacion_y_escalado_se_ajustan_solo_con_fit():
    X, y = _toy()
    model = make_logistic().fit(X, y)
    escalar = model.named_steps["escalar"]
    media_antes = escalar.mean_.copy()
    model.predict_proba(X + 100)  # datos "nuevos" muy distintos
    assert np.allclose(escalar.mean_, media_antes)
    assert np.allclose(model.named_steps["imputar"].statistics_, X.median())


def test_tune_logistic_elige_c_de_la_grilla_y_ordena_por_fecha(monkeypatch):
    import src.models.logistic as logistic
    vistos = {}

    class GridSearchEspia(logistic.GridSearchCV):
        def fit(self, X, y=None, **kw):
            vistos["X"] = X
            return super().fit(X, y, **kw)

    monkeypatch.setattr(logistic, "GridSearchCV", GridSearchEspia)
    rng = np.random.default_rng(1)
    X, y = _toy(n_por_clase=100)
    orden = rng.permutation(len(y))  # x y resultado mezclados en el tiempo
    df = X.iloc[orden].reset_index(drop=True).assign(
        result=y[orden], match_id=np.arange(len(y)),
        date=pd.date_range("2020-01-01", periods=len(y)))
    df = df.sample(frac=1, random_state=0)  # filas desordenadas: la función debe reordenar
    search = tune_logistic(df, ["x", "ruido"], Cs=[0.01, 1.0])
    assert search.best_params_["logistica__C"] in (0.01, 1.0)
    assert isinstance(search.cv, logistic.TimeSeriesSplit) and search.cv.n_splits == 5
    fechas = df.loc[vistos["X"].index, "date"]
    assert fechas.is_monotonic_increasing


def test_coef_table_en_orden_hda_con_nombres_en_espanol():
    X, y = _toy()
    X = X.rename(columns={"x": "elo_diff"})
    model = make_logistic().fit(X, y)
    tabla = coef_table(model, ["elo_diff", "ruido"])
    assert list(tabla.columns) == ["H", "D", "A"]
    assert tabla.index[0].startswith("Diferencia de Elo")
    assert tabla.index[1] == "ruido"  # sin traducción, se deja el nombre original
    # x bajo → H: coeficiente de H negativo, de A positivo
    assert tabla.iloc[0]["H"] < 0 < tabla.iloc[0]["A"]
    assert tabla.sum(axis=1).to_numpy() == pytest.approx([0, 0], abs=1e-3)
