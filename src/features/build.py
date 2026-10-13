"""Features previas al partido. REGLA: cada valor solo usa partidos de fechas anteriores.

Uso:  python -m src.features.build
Salida: data/processed/features.csv (una fila por partido)
"""
import pandas as pd

from src.config import (DATA_PROCESSED, ELO_HOME_ADV, ELO_INIT, ELO_K,
                        FORM_WINDOW, ELO_PROMOTED_INIT)

FEATURE_COLS = [
    "home_form_pts", "home_form_gf", "home_form_ga",
    "away_form_pts", "away_form_gf", "away_form_ga",
    "home_elo", "away_elo", "elo_diff",
    "home_rest", "away_rest", "rest_diff",
]


def _long_format(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por (partido, equipo) desde la perspectiva de ese equipo."""
    pts_home = df["result"].map({"H": 3, "D": 1, "A": 0})
    pts_away = df["result"].map({"H": 0, "D": 1, "A": 3})
    home = pd.DataFrame({"match_id": df["match_id"], "date": df["date"], "team": df["home"],
                         "side": "home", "gf": df["home_goals"], "ga": df["away_goals"],
                         "pts": pts_home})
    away = pd.DataFrame({"match_id": df["match_id"], "date": df["date"], "team": df["away"],
                         "side": "away", "gf": df["away_goals"], "ga": df["home_goals"],
                         "pts": pts_away})
    return (pd.concat([home, away], ignore_index=True)
              .sort_values(["team", "date", "match_id"], kind="mergesort"))


def add_form_and_rest(df: pd.DataFrame, window: int = FORM_WINDOW) -> pd.DataFrame:
    """Forma (media de últimos `window` partidos) y días de descanso."""
    long = _long_format(df)
    g = long.groupby("team", sort=False)
    for col in ["pts", "gf", "ga"]:
        # shift(1): el partido actual NUNCA entra en su propia ventana
        long[f"form_{col}"] = g[col].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    long["rest"] = g["date"].diff().dt.days

    wide = long.pivot(index="match_id", columns="side",
                      values=["form_pts", "form_gf", "form_ga", "rest"])
    wide.columns = [f"{side}_{name}" for name, side in wide.columns]
    out = df.merge(wide, left_on="match_id", right_index=True, how="left")
    # Descanso largo (inicio de temporada) se acota para no dominar el modelo
    for side in ["home", "away"]:
        out[f"{side}_rest"] = out[f"{side}_rest"].clip(upper=14)
    out["rest_diff"] = out["home_rest"] - out["away_rest"]
    return out


def add_elo(df: pd.DataFrame, k: float = ELO_K, home_adv: float = ELO_HOME_ADV,
            init: float = ELO_INIT, promoted_init: float = ELO_PROMOTED_INIT) -> pd.DataFrame:
    """Elo previo al partido: se guarda el rating ANTES de actualizarlo.
    Los equipos que aparecen después de la primera temporada (ascendidos) empiezan en promoted_init."""
    ratings: dict[str, float] = {}
    first_season = df["season"].min()
    home_pre, away_pre = [], []
    for row in df.sort_values(["date", "match_id"]).itertuples():
        start = init if row.season == first_season else promoted_init
        rh = ratings.get(row.home, start)
        ra = ratings.get(row.away, start)
        home_pre.append((row.match_id, rh))
        away_pre.append((row.match_id, ra))
        expected_h = 1 / (1 + 10 ** ((ra - (rh + home_adv)) / 400))
        score_h = {"H": 1.0, "D": 0.5, "A": 0.0}[row.result]
        delta = k * (score_h - expected_h)
        ratings[row.home] = rh + delta
        ratings[row.away] = ra - delta
    out = df.copy()
    out["home_elo"] = out["match_id"].map(dict(home_pre))
    out["away_elo"] = out["match_id"].map(dict(away_pre))
    out["elo_diff"] = out["home_elo"] + home_adv - out["away_elo"]
    return out


def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    df = matches.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = add_form_and_rest(df)
    df = add_elo(df)
    return df.sort_values("match_id").reset_index(drop=True)


if __name__ == "__main__":
    m = pd.read_csv(DATA_PROCESSED / "matches.csv", parse_dates=["date"])
    f = build_features(m)
    f.to_csv(DATA_PROCESSED / "features.csv", index=False)
    print(f[FEATURE_COLS].describe().T.round(2))
