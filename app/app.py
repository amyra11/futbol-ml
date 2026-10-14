"""Demo: elige dos equipos y mira las probabilidades.  Uso: streamlit run app/app.py"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))   # para importar src/

import pandas as pd
import streamlit as st

from src.config import DATA_PROCESSED
from src.models.elo_logistic import EloLogistic
from src.features.current import recent_form

st.set_page_config(page_title="Predicción de fútbol", page_icon="⚽")


@st.cache_data
def load_features() -> pd.DataFrame:
    return pd.read_csv(DATA_PROCESSED / "features.csv", parse_dates=["date"])


def latest_elo(f: pd.DataFrame, team: str) -> float:
    """Último Elo previo al partido conocido del equipo."""
    rows = f[(f["home"] == team) | (f["away"] == team)].sort_values("match_id")
    last = rows.iloc[-1]
    return float(last["home_elo"] if last["home"] == team else last["away_elo"])


f = load_features()
@st.cache_resource
def load_model() -> EloLogistic:
    return EloLogistic().fit(load_features())
season_actual = f["season"].max()
teams = sorted(set(f.loc[f["season"] == season_actual, "home"]))

st.title("⚽ Predicción de partidos de la Premier League")
st.caption("Herramienta educativa y analítica. No es un consejo para apostar.")

col1, col2 = st.columns(2)
home = col1.selectbox("Equipo local", teams, index=0)
away = col2.selectbox("Equipo visitante", teams, index=1)

if home == away:
    st.warning("Elige dos equipos distintos.")
    st.stop()

elo_h, elo_a = latest_elo(f, home), latest_elo(f, away)
col1.metric("Elo local", f"{elo_h:.0f}")
col2.metric("Elo visitante", f"{elo_a:.0f}")

probs = load_model().predict_proba_teams(elo_h, elo_a)
etiquetas = [f"1 · Gana {home}", "2 · Empate", f"3 · Gana {away}"]

st.subheader("Probabilidades")
c1, c2, c3 = st.columns(3)
for col, etiqueta, p in zip((c1, c2, c3), etiquetas, probs):
    col.metric(etiqueta, f"{p:.0%}")
st.bar_chart(pd.DataFrame({"Probabilidad": probs}, index=etiquetas))
st.info("Modelo provisional basado solo en el Elo. Se reemplazará por el modelo final.")

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
st.caption("Datos hasta el final de la temporada 2024-25. V = victoria, E = empate, D = derrota.")