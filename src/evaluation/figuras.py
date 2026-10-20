"""Figuras para la presentación (letra grande: se proyectan en la feria)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
_SEGUNDO = "#eb6834"      # segunda serie (cuotas) en gráficos con dos modelos

NOMBRES_CLASES = {"H": "Gana el local (H)", "D": "Empate (D)", "A": "Gana el visitante (A)"}


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


def fig_calibracion(y, probs: dict, ruta, n_bins: int = 10):
    """Curvas de calibración en 3 paneles (H, D, A). `probs` = {etiqueta: matriz H, D, A} con
    dos modelos como máximo: el primero en azul (modelo final), el segundo en naranja (cuotas).
    Cada punto es un bin por cuantiles; su área es proporcional a n."""
    from src.evaluation.calibracion import datos_calibracion, ece

    plt.rcParams.update({"font.size": 18, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 3, figsize=(19, 8.6), dpi=150)
    colores = [_FINAL, _SEGUNDO]
    for ax, clase in zip(axes, ["H", "D", "A"]):
        datos = {nombre: datos_calibracion(y, p, clase, n_bins) for nombre, p in probs.items()}
        tope = max(max(d["prob_media"].max(), d["frec_observada"].max()) for d in datos.values())
        lim = min(1.0, np.ceil((tope + 0.05) * 10) / 10)
        ax.plot([0, lim], [0, lim], color=_SUAVE, ls="--", lw=2, zorder=1)
        for (nombre, d), color in zip(datos.items(), colores):
            ax.plot(d["prob_media"], d["frec_observada"], color=color, lw=2, zorder=2)
            ax.scatter(d["prob_media"], d["frec_observada"], s=d["n"] * 4, color=color,
                       edgecolor="white", linewidth=2, zorder=3)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.set_title(NOMBRES_CLASES[clase], fontsize=20, pad=12)
        ax.set_xlabel("Probabilidad predicha")
        ax.grid(color="#e6e5e0", lw=1)
        ax.set_axisbelow(True)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
    axes[0].set_ylabel("Frecuencia real")

    manejadores = [plt.Line2D([], [], color=c, lw=2, marker="o", markersize=12,
                              markeredgecolor="white", markeredgewidth=2,
                              label=f"{nombre}  (ECE {ece(y, p, n_bins):.3f})")
                   for (nombre, p), c in zip(probs.items(), colores)]
    manejadores.append(plt.Line2D([], [], color=_SUAVE, ls="--", lw=2, label="Calibración perfecta"))
    fig.legend(handles=manejadores, loc="lower center", ncol=3, frameon=False, fontsize=16,
               bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("¿Las probabilidades dicen la verdad?", x=0.01, ha="left", fontsize=24,
                 fontweight="bold")
    n_partidos = len(np.asarray(y))
    fig.text(0.01, 0.885, f"Validación 2023-24 · {n_partidos} partidos · cada punto agrupa "
             f"~{n_partidos // n_bins} partidos · sobre la diagonal = lo predicho ocurre",
             color=_SUAVE, fontsize=15)
    # Márgenes a mano: con ejes de aspecto igual, tight_layout deja la leyenda encima del eje X
    fig.subplots_adjust(left=0.05, right=0.99, top=0.80, bottom=0.2, wspace=0.2)
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, facecolor="white")
    return fig


# Contribuciones SHAP: verde = a favor, rojo = en contra (mismos colores que la app).
# Par validado para daltonismo; además cada barra lleva su valor con signo.
_A_FAVOR, _EN_CONTRA = "#00703f", "#e8664f"


def _nombre_variable(col: str) -> str:
    from src.models.logistic import NOMBRES_ES
    return NOMBRES_ES.get(col, col)


def _formato_valor(v: float) -> str:
    if abs(v) >= 20 or float(v).is_integer():
        return f"{v:.0f}"
    return f"{v:.1f}"


def _estilo_barras_horizontales(ax):
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e6e5e0", lw=1)
    ax.set_axisbelow(True)


def fig_shap_global(importancia: pd.Series, ruta, n_partidos: int, referencia: str = "elo_diff"):
    """Barras horizontales de la media de |SHAP| por variable (la mayor arriba). La variable
    `referencia` (Elo) va resaltada; las demás llevan su peso relativo a ella."""
    imp = importancia.sort_values()                      # barh dibuja de abajo hacia arriba
    ref = importancia[referencia]

    plt.rcParams.update({"font.size": 18, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=150)
    colores = [_FINAL if c == referencia else _OTRO for c in imp.index]
    ax.barh(range(len(imp)), imp.to_numpy(), height=0.62, color=colores)
    ax.set_yticks(range(len(imp)), [_nombre_variable(c) for c in imp.index])
    for lbl, c in zip(ax.get_yticklabels(), imp.index):
        lbl.set_fontweight("bold" if c == referencia else "normal")
    for i, (c, v) in enumerate(imp.items()):
        texto = f"{v:.2f}" if c == referencia else f"{v:.2f}  ({v / ref:.0%} del Elo)"
        ax.text(v + ref * 0.012, i, texto, va="center", color=_TEXTO, fontsize=16,
                fontweight="bold" if c == referencia else "normal")

    ax.set_xlim(0, ref * 1.32)
    ax.set_ylim(-0.6, len(imp) - 0.4)
    ax.set_xlabel("Media de |SHAP| hacia el resultado predicho (escala logit)")
    fig.suptitle("¿Qué variables mueven más la predicción?", x=0.01, y=0.975, ha="left",
                 fontsize=24, fontweight="bold")
    fig.text(0.01, 0.9, f"{n_partidos} partidos de validación 2023-24 · modelo final entrenado "
             "solo con train · azul = Elo", color=_SUAVE, fontsize=15)
    _estilo_barras_horizontales(ax)
    fig.tight_layout()
    fig.subplots_adjust(top=0.86)
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, facecolor="white")
    return fig


def fig_shap_ejemplo(shap_fila: pd.Series, valores: pd.Series, probs, titulo: str,
                     resultado_real: str, ruta, top: int = 6):
    """Explicación de un partido: contribución de cada variable hacia la clase predicha
    (verde = la favorece, rojo = en contra). Las variables fuera del `top` se suman en una
    barra "Otras". `shap_fila` y `valores` van indexados por nombre de columna;
    `probs` en orden H, D, A."""
    from src.config import CLASSES

    probs = np.asarray(probs, dtype=float)
    clase = CLASSES[int(probs.argmax())]
    orden = shap_fila.abs().sort_values(ascending=False).index
    principales, resto = list(orden[:top]), list(orden[top:])
    etiquetas = [f"{_nombre_variable(c)} = {_formato_valor(valores[c])}" for c in principales]
    contrib = [float(shap_fila[c]) for c in principales]
    if resto:
        etiquetas.append(f"Otras {len(resto)} variables")
        contrib.append(float(shap_fila[resto].sum()))
    etiquetas, contrib = etiquetas[::-1], np.array(contrib[::-1])   # la mayor arriba

    plt.rcParams.update({"font.size": 18, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=150)
    colores = [_A_FAVOR if v >= 0 else _EN_CONTRA for v in contrib]
    ax.barh(range(len(contrib)), contrib, height=0.62, color=colores)
    ax.axvline(0, color=_SUAVE, lw=1.5)
    ax.set_yticks(range(len(contrib)), etiquetas)
    lim = np.abs(contrib).max()
    for i, v in enumerate(contrib):
        ax.text(v + (1 if v >= 0 else -1) * lim * 0.015, i, f"{v:+.2f}".replace("-", "−"),
                va="center", ha="left" if v >= 0 else "right", color=_TEXTO, fontsize=16)
    ax.set_xlim(min(contrib.min(), 0) - lim * 0.18, max(contrib.max(), 0) + lim * 0.15)
    ax.set_ylim(-0.6, len(contrib) - 0.4)
    ax.set_xlabel(f"Empuje hacia «{NOMBRES_CLASES[clase]}» (valor SHAP, escala logit)")

    pH, pD, pA = probs
    fig.suptitle(titulo, x=0.01, y=0.975, ha="left", fontsize=24, fontweight="bold")
    fig.text(0.01, 0.885, f"Modelo: local {pH:.0%} · empate {pD:.0%} · visitante {pA:.0%}   │   "
             f"Resultado real: {resultado_real}", color=_TEXTO, fontsize=18)
    fig.text(0.01, 0.835, "Verde = hace más probable el resultado predicho · Rojo = lo hace "
             "menos probable", color=_SUAVE, fontsize=15)
    _estilo_barras_horizontales(ax)
    fig.tight_layout()
    fig.subplots_adjust(top=0.8)
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, facecolor="white")
    return fig


def fig_shap_ejemplo_agrupado(grupos: list[dict], probs, titulo: str, resultado_real: str, ruta):
    """Explicación de un partido para el público: SHAP sumado por grupo (Elo, forma, descanso)
    hacia la clase predicha. `grupos` = salida de `ModeloFinal.factores_agrupados()`;
    `probs` en orden H, D, A. Debajo del nombre de cada grupo van sus datos del partido."""
    from matplotlib.transforms import blended_transform_factory

    from src.config import CLASSES

    probs = np.asarray(probs, dtype=float)
    clase = CLASSES[int(probs.argmax())]
    grupos = sorted(grupos, key=lambda g: abs(g["impacto"]))      # la mayor arriba
    contrib = np.array([g["impacto"] for g in grupos])

    plt.rcParams.update({"font.size": 18, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=150)
    colores = [_A_FAVOR if v >= 0 else _EN_CONTRA for v in contrib]
    ax.barh(range(len(contrib)), contrib, height=0.62, color=colores)
    ax.axvline(0, color=_SUAVE, lw=1.5)
    ax.set_yticks(range(len(contrib)), [""] * len(contrib))
    # Nombre del grupo en negrita y, debajo, sus datos (una línea por equipo)
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    for i, g in enumerate(grupos):
        ax.text(-0.02, i + 0.12, g["grupo"], transform=trans, ha="right", va="bottom",
                color=_TEXTO, fontsize=18, fontweight="bold")
        ax.text(-0.02, i + 0.06, g["detalle"].replace(" · ", "\n"), transform=trans,
                ha="right", va="top", color=_SUAVE, fontsize=14, linespacing=1.3)
    lim = np.abs(contrib).max()
    for i, v in enumerate(contrib):
        ax.text(v + (1 if v >= 0 else -1) * lim * 0.015, i, f"{v:+.2f}".replace("-", "−"),
                va="center", ha="left" if v >= 0 else "right", color=_TEXTO, fontsize=16)
    ax.set_xlim(min(contrib.min(), 0) - lim * 0.18, max(contrib.max(), 0) + lim * 0.15)
    ax.set_ylim(-0.6, len(contrib) - 0.4)
    ax.set_xlabel(f"Empuje hacia «{NOMBRES_CLASES[clase]}» (suma de SHAP, escala logit)")

    pH, pD, pA = probs
    fig.suptitle(titulo, x=0.01, y=0.975, ha="left", fontsize=24, fontweight="bold")
    fig.text(0.01, 0.885, f"Modelo: local {pH:.0%} · empate {pD:.0%} · visitante {pA:.0%}   │   "
             f"Resultado real: {resultado_real}", color=_TEXTO, fontsize=18)
    fig.text(0.01, 0.835, "Verde = hace más probable el resultado predicho · Rojo = lo hace "
             "menos probable · cada barra suma las variables de su grupo", color=_SUAVE, fontsize=15)
    _estilo_barras_horizontales(ax)
    fig.subplots_adjust(left=0.43, right=0.97, top=0.8, bottom=0.12)
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, facecolor="white")
    return fig
