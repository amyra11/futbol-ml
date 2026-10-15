"""Estado actual de un equipo (últimos partidos conocidos), para la demo."""
import numpy as np
import pandas as pd

from src.config import FORM_WINDOW


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