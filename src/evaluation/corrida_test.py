"""Corrida de evaluación final: todos los modelos con el MISMO procedimiento.

Dos modos:
  --ensayo  Ajusta con train (2017-18 a 2022-23) y evalúa en VALIDACIÓN (2023-24). NO toca el test.
            Sirve para comprobar que el script reproduce la tabla de validación (docs/resultados_validacion.csv).
            Escribe en data/processed/ensayo/ (ignorado por git).
  --test    Ajusta con train+val (2017-18 a 2023-24) y evalúa en TEST (2024-25). SE USA UNA SOLA VEZ:
            pide confirmación escrita y se niega a correr si ya existe docs/resultados_test.csv.

Uso:  python -m src.evaluation.corrida_test --ensayo
      python -m src.evaluation.corrida_test --test
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, ROOT
from src.evaluation.bootstrap import tabla_bootstrap
from src.evaluation.metrics import compare, confusion, evaluate, implied_probs
from src.features.build import FEATURE_COLS
from src.models.baseline import FrequencyBaseline
from src.models.common import proba_hda
from src.models.final import FEATURES_FINAL
from src.models.logistic import tune_logistic
from src.models.poisson import PoissonElo
from src.models.trees import tune_random_forest, tune_xgboost
from src.splits import split

FINAL = "logistica_sin_redundancia"
CONFIRMACION = "USAR TEST"
SALIDA_TEST = ROOT / "docs"
SALIDA_ENSAYO = DATA_PROCESSED / "ensayo"
ARCHIVO_CONTROL = "resultados_test.csv"


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "desconocido"


def _versiones() -> dict:
    import platform
    import sklearn
    import xgboost
    return {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
            "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__}


def probabilidades(fit: pd.DataFrame, ev: pd.DataFrame, grid_rf=None, grid_xgb=None) -> dict:
    """Ajusta cada modelo SOLO con `fit` (con TimeSeriesSplit dentro de fit) y devuelve sus
    probabilidades H, D, A sobre `ev`. Mismo procedimiento que notebooks/09_arboles_comparacion_B."""
    fit = fit.sort_values(["date", "match_id"])
    assert fit["date"].max() < ev["date"].min(), "los datos de ajuste deben ser anteriores a los evaluados"
    kw_rf = {"grid": grid_rf} if grid_rf else {}
    kw_xgb = {"grid": grid_xgb} if grid_xgb else {}
    probs = {
        "baseline_frecuencias": FrequencyBaseline().fit(None, fit["result"]).predict_proba(ev),
        "cuotas_mercado": implied_probs(ev),
        "poisson_elo": PoissonElo().fit(fit).predict_proba(ev),
    }
    for nombre, cols in {"logistica_elo": ["elo_diff"], "logistica_todas": FEATURE_COLS,
                         FINAL: FEATURES_FINAL}.items():
        probs[nombre] = proba_hda(tune_logistic(fit, cols).best_estimator_, ev[cols])
    probs["random_forest"] = proba_hda(tune_random_forest(fit, FEATURES_FINAL, **kw_rf).best_estimator_,
                                       ev[FEATURES_FINAL])
    probs["xgboost"] = proba_hda(tune_xgboost(fit, FEATURES_FINAL, **kw_xgb).best_estimator_,
                                 ev[FEATURES_FINAL])
    return probs


def correr(features: pd.DataFrame, modo: str, salida: Path | None = None, grid_rf=None, grid_xgb=None,
           n_boot: int = 1000, verificar_modelo_app: bool = True) -> pd.DataFrame:
    """Devuelve la tabla comparativa y escribe los archivos en `salida`.
    modo='ensayo' usa validación; modo='test' usa el conjunto de prueba (solo con confirmación previa)."""
    if modo not in ("ensayo", "test"):
        raise ValueError("modo debe ser 'ensayo' o 'test'")
    salida = Path(salida) if salida else (SALIDA_TEST if modo == "test" else SALIDA_ENSAYO)
    if modo == "test" and (salida / ARCHIVO_CONTROL).exists():
        raise RuntimeError(f"{salida / ARCHIVO_CONTROL} ya existe: el test se usa UNA sola vez.")

    fit, ev = split(features, use_test=True) if modo == "test" else split(features)
    ev = ev[ev[["odds_H", "odds_D", "odds_A"]].notna().all(axis=1)]     # mismos partidos para todos
    y = ev["result"]
    print(f"[{modo}] ajuste: {len(fit)} partidos | evaluación: {len(ev)} partidos con cuotas")

    probs = probabilidades(fit, ev, grid_rf, grid_xgb)
    tabla = compare([evaluate(y, p, nombre) for nombre, p in probs.items()])
    boot = tabla_bootstrap(y, probs, ref=FINAL, n_boot=n_boot)

    if modo == "test" and verificar_modelo_app:
        # el modelo que se evalúa debe ser el mismo que usa la app (models/modelo_final.joblib)
        from src.models.final import cargar_modelo
        guardado = cargar_modelo(features=features)
        assert np.allclose(guardado.predict_proba(ev), probs[FINAL], atol=1e-9), \
            "el modelo guardado de la app no coincide con el evaluado: reentrénalo con `python -m src.models.final`"

    salida.mkdir(parents=True, exist_ok=True)
    nombre_res = ARCHIVO_CONTROL if modo == "test" else "resultados_ensayo.csv"
    tabla.to_csv(salida / nombre_res, index=False)
    boot.to_csv(salida / ("bootstrap_test.csv" if modo == "test" else "bootstrap_ensayo.csv"), index=False)
    confusion(y, probs[FINAL]).to_csv(salida / ("confusion_test.csv" if modo == "test" else "confusion_ensayo.csv"))
    meta = {"versiones": _versiones(), "modo": modo, "fecha": datetime.now().isoformat(timespec="seconds"),
            "commit": _commit(),
            "temporadas_ajuste": sorted(fit["season"].astype(str).str.zfill(4).unique()),
            "temporadas_evaluacion": sorted(ev["season"].astype(str).str.zfill(4).unique()),
            "n_ajuste": int(len(fit)), "n_evaluacion": int(len(ev)), "modelo_final": FINAL}
    (salida / ("resultados_test_meta.json" if modo == "test" else "meta_ensayo.json")).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False))
    return tabla


def comparar_con_validacion(tabla: pd.DataFrame, tol: float = 1e-6) -> None:
    """Ensayo: compara con docs/resultados_validacion.csv. Cualquier diferencia > tol se avisa;
    en XGBoost es esperable un pequeño desfase si cambia la versión de la librería."""
    ref = pd.read_csv(ROOT / "docs" / "resultados_validacion.csv").set_index("modelo")["log_loss"]
    dif = (tabla.set_index("modelo")["log_loss"] - ref.reindex(tabla["modelo"]).to_numpy()).abs()
    print("\nComparación con docs/resultados_validacion.csv (log loss):")
    for modelo, d in dif.items():
        print(f"  {'OK ' if d <= tol else 'DIF'} {modelo:<28} diferencia {d:.6f}")


def _confirmar() -> bool:
    print("\n⚠️  Vas a evaluar en el CONJUNTO DE PRUEBA (2024-25). Esto se hace UNA sola vez.")
    print("   Después no se cambia el modelo final ni se repite la corrida.")
    return input(f"   Escribe exactamente «{CONFIRMACION}» para continuar: ").strip() == CONFIRMACION


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--ensayo", action="store_true", help="ajusta con train y evalúa en validación")
    g.add_argument("--test", action="store_true", help="CORRIDA ÚNICA sobre el conjunto de prueba")
    args = p.parse_args(argv)

    features = pd.read_csv(DATA_PROCESSED / "features.csv", parse_dates=["date"])
    if args.test:
        if (SALIDA_TEST / ARCHIVO_CONTROL).exists():
            print(f"✋ {SALIDA_TEST / ARCHIVO_CONTROL} ya existe: el test ya se usó. No se vuelve a correr.")
            return 1
        if not _confirmar():
            print("Cancelado. No se tocó el conjunto de prueba.")
            return 1
    tabla = correr(features, "test" if args.test else "ensayo")
    print(tabla.round(4).to_string(index=False))
    if not args.test:
        comparar_con_validacion(tabla)
    return 0


if __name__ == "__main__":
    sys.exit(main())