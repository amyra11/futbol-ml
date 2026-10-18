"""Figuras para la presentación (letra grande: se proyectan en la feria)."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

NOMBRES_MODELOS = {
    "baseline_frecuencias": "Baseline (frecuencias)",
    "cuotas_mercado": "Cuotas del mercado",
    "logistica_elo": "Logística · solo Elo",
    "logistica_todas": "Logística · todas las variables",
    "logistica_sin_redundancia": "Logística · sin redundancia",
    "poisson_elo": "Poisson · Elo",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

_FINAL, _OTRO, _TEXTO, _SUAVE = "#2a78d6", "#b9b8b2", "#0b0b0b", "#52514e"


def fig_comparacion_modelos(tabla: pd.DataFrame, ruta, final: str = "logistica_sin_redundancia",
                            peores_signif: set[str] = frozenset()):
    """Barras horizontales de log loss en validación (mejor arriba). Baseline y cuotas van como
    líneas de referencia; `final` se resalta; `peores_signif` lleva un asterisco."""
    ref = tabla.set_index("modelo")["log_loss"]
    t = (tabla[~tabla["modelo"].isin(["baseline_frecuencias", "cuotas_mercado"])]
         .sort_values("log_loss", ascending=False))   # barh dibuja de abajo hacia arriba

    plt.rcParams.update({"font.size": 18, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(14, 7.5), dpi=150)
    colores = [_FINAL if m == final else _OTRO for m in t["modelo"]]
    ax.barh(range(len(t)), t["log_loss"], height=0.6, color=colores)
    etiquetas = [NOMBRES_MODELOS.get(m, m) + (" *" if m in peores_signif else "") for m in t["modelo"]]
    ax.set_yticks(range(len(t)), etiquetas)
    for lbl, m in zip(ax.get_yticklabels(), t["modelo"]):
        lbl.set_fontweight("bold" if m == final else "normal")
    for i, (m, v) in enumerate(zip(t["modelo"], t["log_loss"])):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", color=_TEXTO,
                fontweight="bold" if m == final else "normal")

    # Etiqueta de las cuotas a la derecha de su línea; la del baseline, a la izquierda
    for clave, texto, lado in [("cuotas_mercado", "Cuotas del mercado", "left"),
                               ("baseline_frecuencias", "Baseline (frecuencias)", "right")]:
        v = ref[clave]
        ax.axvline(v, color=_SUAVE, ls="--", lw=2)
        ax.text(v + (0.002 if lado == "left" else -0.002), len(t) - 0.35, f"{texto}\n{v:.3f}",
                ha=lado, va="bottom", color=_SUAVE, fontsize=15)

    x0 = 0.88
    ax.set_xlim(x0, ref["baseline_frecuencias"] + 0.012)
    ax.set_ylim(-0.6, len(t) + 0.55)
    ax.set_xlabel("Log loss en validación 2023-24 (menor es mejor)")
    ax.set_title("¿Qué modelo da mejores probabilidades?", fontsize=24, loc="left", pad=46,
                 fontweight="bold")
    ax.text(0, 1.015, f"380 partidos de la Premier 2023-24 · el eje empieza en {x0}",
            transform=ax.transAxes, color=_SUAVE, fontsize=15)
    if peores_signif:
        fig.text(0.01, 0.01, "* Peor que el modelo final con diferencia significativa "
                 "(bootstrap pareado, IC 95 %)", color=_SUAVE, fontsize=14)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e6e5e0", lw=1)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, facecolor="white")
    return fig
