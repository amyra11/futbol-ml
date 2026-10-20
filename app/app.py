"""Demo: elige dos equipos y mira las probabilidades.  Uso: streamlit run app/app.py"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))   # para importar src/

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import DATA_PROCESSED
from src.features.current import matchup_features, recent_form
from src.models.elo_logistic import EloLogistic
from src.models.poisson import PoissonElo, score_matrix
from src.splits import split

st.set_page_config(page_title="Predicción de fútbol", page_icon="⚽")


@st.cache_data
def load_features() -> pd.DataFrame:
    return pd.read_csv(DATA_PROCESSED / "features.csv", parse_dates=["date"])


@st.cache_resource
def load_model():
    """Modelo final de probabilidades H/D/A (src/models/final.py). Contrato: predict_proba(X) con
    las FEATURE_COLS de uno o varios partidos, devolviendo (n, 3) en orden H, D, A.
    Si models/modelo_final.joblib no existe, se entrena y se guarda la primera vez."""
    from src.models.final import cargar_modelo
    return cargar_modelo()


@st.cache_resource
def load_poisson() -> PoissonElo:
    train, val = split(load_features())          # nunca usa la temporada de prueba
    return PoissonElo().fit(pd.concat([train, val]))


f = load_features()
season_actual = f["season"].max()
teams = sorted(set(f.loc[f["season"] == season_actual, "home"]))

st.title("⚽ Predicción de partidos de la Premier League")
st.caption("Herramienta educativa y analítica. No es un consejo para apostar.")

col1, col2 = st.columns(2)
home = col1.selectbox("Equipo local", teams, index=0)
away = col2.selectbox("Equipo visitante", teams, index=1)
rest_h = col1.slider("Días de descanso del local", 3, 14, 7,
                     help="14 significa dos semanas o más (inicio de temporada o parón). La mediana histórica es 7.")
rest_a = col2.slider("Días de descanso del visitante", 3, 14, 7,
                     help="14 significa dos semanas o más (inicio de temporada o parón). La mediana histórica es 7.")

if home == away:
    st.warning("Elige dos equipos distintos.")
    st.stop()

X = matchup_features(f, home, away, rest_home=rest_h, rest_away=rest_a)   # las 12 features del partido
elo_h, elo_a = float(X.loc[0, "home_elo"]), float(X.loc[0, "away_elo"])
col1.metric("Elo local", f"{elo_h:.0f}")
col2.metric("Elo visitante", f"{elo_a:.0f}")

probs = load_model().predict_proba(X)[0]
etiquetas = [f"1 · Gana {home}", "2 · Empate", f"3 · Gana {away}"]

st.subheader("Probabilidades")
c1, c2, c3 = st.columns(3)
for col, etiqueta, p in zip((c1, c2, c3), etiquetas, probs):
    col.metric(etiqueta, f"{p:.0%}")
st.bar_chart(pd.DataFrame({"Probabilidad": probs}, index=etiquetas))
st.info("Modelo final: regresión logística con la diferencia de Elo, la forma de los últimos 5 "
        "partidos y los días de descanso, entrenada con las temporadas 2017-18 a 2023-24.")

st.subheader("Goles esperados (modelo Poisson)")
lam_l, lam_v = (a[0] for a in load_poisson().expected_goals(X))
g1, g2 = st.columns(2)
g1.metric(f"Goles esperados · {home}", f"{lam_l:.2f}")
g2.metric(f"Goles esperados · {away}", f"{lam_v:.2f}")

M = score_matrix(lam_l, lam_v, max_goles=5)
i, j = np.unravel_index(M.argmax(), M.shape)
st.write(f"Marcador más probable: **{home} {i} – {j} {away}** ({M[i, j]:.0%})")
matriz = pd.DataFrame(M * 100, index=range(6), columns=range(6))
matriz.index.name, matriz.columns.name = f"Goles {home}", f"Goles {away}"
st.dataframe(matriz.style.format("{:.1f}%").background_gradient(cmap="Blues"))
st.caption(f"Filas: goles de {home}. Columnas: goles de {away}. Se muestran hasta 5 goles por equipo. "
           "Cada celda es la probabilidad de ese marcador exacto; aun el más probable ronda el 10–15 %, "
           "porque el fútbol tiene mucho azar.")

st.subheader("Factores clave")
fh, fa = recent_form(f, home), recent_form(f, away)
tabla = pd.DataFrame(
    {
        home: [f"{elo_h:.0f}", f"{fh['pts']:.2f}", f"{fh['gf']:.1f}", f"{fh['ga']:.1f}", fh["racha"]],
        away: [f"{elo_a:.0f}", f"{fa['pts']:.2f}", f"{fa['gf']:.1f}", f"{fa['ga']:.1f}", fa["racha"]],
    },
    index=["Elo", "Puntos por partido (últimos 5)", "Goles a favor (prom.)",
           "Goles en contra (prom.)", "Racha (antiguo → reciente)"],
)
st.table(tabla)
st.caption("Datos hasta el final de la temporada 2024-25. El Elo es el actual (tras el último partido). "
           "V = victoria, E = empate, D = derrota.")

st.markdown("#### ¿Por qué predice esto el modelo?")
mas_probable = [f"gana {home}", "empate", f"gana {away}"][int(probs.argmax())]
st.write(f"Cada barra muestra cuánto empuja ese grupo de datos hacia el resultado más probable "
         f"(**{mas_probable}**, {probs.max():.0%}): **verde** lo hace más probable y **rojo** menos "
         "probable. Debajo de cada nombre están los datos del partido.")
grupos = pd.DataFrame(load_model().factores_agrupados(X))
# Etiqueta en varias líneas: nombre del grupo y luego sus datos; "|" marca cada salto de línea
grupos["etiqueta"] = grupos["grupo"] + "|" + grupos["detalle"].str.replace(" · ", "|")
grupos["sentido"] = np.where(grupos["impacto"] >= 0, "A favor", "En contra")
grafico_grupos = alt.Chart(grupos).mark_bar().encode(
    x=alt.X("impacto:Q", title="Empuje hacia el resultado más probable (suma de valores SHAP)"),
    y=alt.Y("etiqueta:N", sort=None, title=None,
            axis=alt.Axis(labelExpr="split(datum.label, '|')", labelLimit=420, labelPadding=8)),
    color=alt.Color("sentido:N", legend=None,
                    scale=alt.Scale(domain=["A favor", "En contra"], range=["#00703f", "#e8664f"])),
    tooltip=[alt.Tooltip("grupo:N", title="Grupo"), alt.Tooltip("detalle:N", title="Datos"),
             alt.Tooltip("impacto:Q", title="Empuje", format="+.2f"), alt.Tooltip("sentido:N", title="Sentido")],
).properties(height=330)
st.altair_chart(grafico_grupos, width="stretch")
st.caption("Cada grupo suma los valores SHAP de sus variables (en un modelo lineal se pueden sumar sin perder "
           "exactitud). Explican al modelo, no causas del juego.")

with st.expander("Ver detalle por variable"):
    st.write("Las 4 variables que más pesan en este partido, una por una. Como puntos, goles y Elo están "
             "relacionados entre sí, alguna variable de forma suelta puede salir con un signo que parece "
             "raro; por eso arriba se muestran sumadas por grupo.")
    factores = pd.DataFrame(load_model().factores(X, top=4))
    factores["dato"] = factores["variable"] + " = " + factores["valor"].map(
        lambda v: f"{v:.0f}" if abs(v) >= 20 or float(v).is_integer() else f"{v:.1f}")
    factores["sentido"] = np.where(factores["impacto"] >= 0, "A favor", "En contra")
    grafico_factores = alt.Chart(factores).mark_bar().encode(
        x=alt.X("impacto:Q", title="Empuje hacia el resultado más probable (valor SHAP)"),
        y=alt.Y("dato:N", sort=None, title=None, axis=alt.Axis(labelLimit=420)),
        color=alt.Color("sentido:N", legend=None,
                        scale=alt.Scale(domain=["A favor", "En contra"], range=["#00703f", "#e8664f"])),
        tooltip=[alt.Tooltip("variable:N", title="Variable"), alt.Tooltip("valor:Q", title="Valor", format=".2f"),
                 alt.Tooltip("impacto:Q", title="Empuje", format="+.2f"), alt.Tooltip("sentido:N", title="Sentido")],
    )
    st.altair_chart(grafico_factores, width="stretch")
    st.caption("Valores SHAP del modelo lineal: coeficiente × dato estandarizado. Explican al modelo, no causas del juego.")
