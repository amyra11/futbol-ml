"""Demo: elige dos equipos y mira las probabilidades.  Uso: streamlit run app/app.py"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))   # para importar src/

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
rest_h = col1.slider("Días de descanso del local", 2, 14, 7)
rest_a = col2.slider("Días de descanso del visitante", 2, 14, 7)

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