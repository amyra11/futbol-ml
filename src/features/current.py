"""Estado actual de un equipo (últimos partidos conocidos), para la demo."""
import numpy as np
import pandas as pd

from src.config import ELO_HOME_ADV, FORM_WINDOW
from src.features.build import FEATURE_COLS, final_elo


def recent_form(f: pd.DataFrame, team: str, window: int = FORM_WINDOW) -> dict:
    """Puntos, goles a favor/en contra (promedios) y racha de los últimos `window` partidos.
    La racha va del más antiguo al más reciente: V victoria, E empate, D derrota."""
    rows = f[(f["home"] == team) | (f["away"] == team)].sort_values("match_id").tail(window)
    is_home = (rows["home"] == team).to_numpy()
    gf = np.where(is_home, rows["home_goals"], rows["away_goals"])
    ga = np.where(is_home, rows["away_goals"], rows["home_goals"])
    pts = np.select([gf > ga, gf == ga], [3, 1], 0)
    racha = "".join("V" if p == 3 else "E" if p == 1 else "D" for p in pts)
    return {"pts": float(pts.mean()), "gf": float(gf.mean()), "ga": float(ga.mean()), "racha": racha}

def matchup_features(f: pd.DataFrame, home: str, away: str,
                     rest_home: float = 7, rest_away: float = 7) -> pd.DataFrame:
    """Una fila con las FEATURE_COLS de un partido hipotético `home` vs `away`, calculadas con el
    estado actual: Elo tras el último partido y forma de los últimos 5 partidos de cada equipo.
    Los días de descanso no se conocen, por eso son un parámetro (por defecto 7, la mediana histórica)."""
    ratings = final_elo(f)
    fh, fa = recent_form(f, home), recent_form(f, away)
    elo_h, elo_a = ratings[home], ratings[away]
    fila = {
        "home_form_pts": fh["pts"], "home_form_gf": fh["gf"], "home_form_ga": fh["ga"],
        "away_form_pts": fa["pts"], "away_form_gf": fa["gf"], "away_form_ga": fa["ga"],
        "home_elo": elo_h, "away_elo": elo_a, "elo_diff": elo_h + ELO_HOME_ADV - elo_a,
        "home_rest": float(rest_home), "away_rest": float(rest_away),
        "rest_diff": float(rest_home) - float(rest_away),
    }
    return pd.DataFrame([fila])[FEATURE_COLS]