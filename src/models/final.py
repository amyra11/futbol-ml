"""Modelo final para la app: logística sin columnas redundantes de Elo.

Contrato con la app: `predict_proba(X)` recibe un DataFrame con las FEATURE_COLS de uno o
varios partidos y devuelve una matriz (n, 3) en orden H, D, A.

Uso:  python -m src.models.final      # entrena con train+val y guarda models/modelo_final.joblib
"""
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import CLASSES, DATA_PROCESSED, ROOT
from src.evaluation.metrics import to_hda
from src.features.build import FEATURE_COLS
from src.models.logistic import NOMBRES_ES, tune_logistic
from src.splits import split

RUTA_MODELO = ROOT / "models" / "modelo_final.joblib"

# elo_diff = home_elo + 60 - away_elo: con las tres columnas el efecto del Elo se reparte y no se
# puede explicar. Sin home_elo/away_elo la log loss en validación quedó igual o mejor
# (notebooks/08_modelo_final_B.ipynb).
FEATURES_FINAL = [c for c in FEATURE_COLS if c not in ("home_elo", "away_elo")]

# Grupos para explicar la predicción al público. Puntos, goles y Elo están correlacionados, así
# que una variable de forma suelta puede salir con un signo raro; la suma del grupo no. Como los
# SHAP de un modelo lineal son aditivos, sumar por grupo no pierde exactitud.
GRUPOS_FACTORES = {
    "Fuerza de los equipos (Elo)": ["elo_diff"],
    "Forma reciente (últimos 5 partidos)": ["home_form_pts", "home_form_gf", "home_form_ga",
                                            "away_form_pts", "away_form_gf", "away_form_ga"],
    "Descanso": ["home_rest", "away_rest", "rest_diff"],
}


def _num(v, fmt: str) -> str:
    return "sin dato" if pd.isna(v) else format(float(v), fmt)


def _detalle_grupo(grupo: str, fila: pd.Series) -> str:
    """Texto legible con los datos del partido que entran en el grupo."""
    if grupo == "Fuerza de los equipos (Elo)":
        dif = f"Diferencia {_num(fila['elo_diff'], '+.0f')} (con la ventaja de local)"
        if {"home_elo", "away_elo"} <= set(fila.index):     # la app y features.csv los traen
            return (f"Local {_num(fila['home_elo'], '.0f')} · "
                    f"Visitante {_num(fila['away_elo'], '.0f')} · {dif}")
        return dif
    if grupo == "Forma reciente (últimos 5 partidos)":
        return " · ".join(
            f"{lado} {_num(fila[f'{pre}_form_pts'], '.1f')} pts/partido, "
            f"{_num(fila[f'{pre}_form_gf'], '.1f')} goles a favor y "
            f"{_num(fila[f'{pre}_form_ga'], '.1f')} en contra"
            for lado, pre in (("Local", "home"), ("Visitante", "away")))
    if grupo == "Descanso":
        return (f"Local {_num(fila['home_rest'], '.0f')} días · "
                f"Visitante {_num(fila['away_rest'], '.0f')} días")
    raise ValueError(f"Grupo desconocido: {grupo}")


class ModeloFinal:
    def __init__(self, pipeline: Pipeline, cols: list[str], meta: dict | None = None):
        self.pipeline = pipeline
        self.cols = list(cols)
        self.meta = meta or {}

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """(n, 3) en orden H, D, A. X puede traer más columnas: se usan solo las del modelo."""
        return to_hda(self.pipeline, self.pipeline.predict_proba(X[self.cols]))

    def factores(self, X: pd.DataFrame, top: int = 5) -> list[dict]:
        """Para la PRIMERA fila de X: variables que más empujan hacia la clase más probable.
        impacto = coeficiente de esa clase × valor imputado y escalado (contribución lineal al
        puntaje de la clase; positivo = la favorece). Ordenado por |impacto|."""
        x = X[self.cols].iloc[[0]]
        z = self.pipeline[:-1].transform(x)[0]          # imputar + escalar, como en el ajuste
        logit = self.pipeline.named_steps["logistica"]
        clase = CLASSES[int(self.predict_proba(x)[0].argmax())]
        impacto = logit.coef_[list(logit.classes_).index(clase)] * z
        orden = np.argsort(-np.abs(impacto))[:top]
        return [{"variable": NOMBRES_ES.get(self.cols[i], self.cols[i]),
                 "valor": float(x.iloc[0, i]),
                 "impacto": float(impacto[i])} for i in orden]

    def factores_agrupados(self, X: pd.DataFrame) -> list[dict]:
        """Como `factores()`, pero sumando los impactos (SHAP) por grupo de GRUPOS_FACTORES:
        Elo, forma reciente y descanso. Para la PRIMERA fila de X, hacia la clase más probable.
        Cada grupo trae su nombre, un `detalle` legible con los datos del partido y su `impacto`
        (positivo = favorece esa clase). Ordenado por |impacto|."""
        a_col = {NOMBRES_ES.get(c, c): c for c in self.cols}
        por_col = {a_col[f["variable"]]: f["impacto"]
                   for f in self.factores(X, top=len(self.cols))}
        fila = X.iloc[0]
        grupos = [{"grupo": g,
                   "detalle": _detalle_grupo(g, fila),
                   "impacto": float(sum(por_col[c] for c in cols)),
                   "variables": list(cols)}
                  for g, cols in GRUPOS_FACTORES.items()]
        return sorted(grupos, key=lambda d: -abs(d["impacto"]))


def entrenar_y_guardar(features: pd.DataFrame | None = None, ruta=RUTA_MODELO) -> ModeloFinal:
    """Ajusta C con TimeSeriesSplit en train+val (el test 2024-25 no se usa) y guarda el modelo."""
    if features is None:
        features = pd.read_csv(DATA_PROCESSED / "features.csv", parse_dates=["date"])
    train, val = split(features)                         # sin use_test: nunca toca el test
    datos = pd.concat([train, val])
    busqueda = tune_logistic(datos, FEATURES_FINAL)
    meta = {
        "features": FEATURES_FINAL,
        "C": float(busqueda.best_params_["logistica__C"]),
        "log_loss_cv": float(-busqueda.best_score_),
        "temporadas": sorted(datos["season"].astype(str).str.zfill(4).unique()),
        "n_partidos": len(datos),
        "ultimo_partido": pd.Timestamp(datos["date"].max()).strftime("%Y-%m-%d"),
        "entrenado_el": datetime.now().isoformat(timespec="seconds"),
    }
    modelo = ModeloFinal(busqueda.best_estimator_, FEATURES_FINAL, meta)
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    # Se guarda un dict (no la clase) para que el joblib no dependa de cómo se ejecutó el script
    joblib.dump({"pipeline": modelo.pipeline, "cols": modelo.cols, "meta": meta}, ruta)
    return modelo


def cargar_modelo(ruta=RUTA_MODELO, features: pd.DataFrame | None = None) -> ModeloFinal:
    """Carga el modelo guardado; si no existe, lo entrena y lo guarda (la demo funciona aunque
    nadie haya corrido `python -m src.models.final`)."""
    ruta = Path(ruta)
    if not ruta.exists():
        return entrenar_y_guardar(features, ruta)
    return ModeloFinal(**joblib.load(ruta))


if __name__ == "__main__":
    m = entrenar_y_guardar()
    print(f"Modelo guardado en {RUTA_MODELO}")
    for k, v in m.meta.items():
        print(f"  {k}: {v}")
