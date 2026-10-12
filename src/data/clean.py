"""Une las temporadas en una sola tabla limpia: data/processed/matches.csv

Uso:  python -m src.data.clean
"""
import pandas as pd

from src.config import DATA_PROCESSED, DATA_RAW, LEAGUE, SEASONS

# Columnas de cuotas por orden de preferencia (los nombres cambian entre temporadas)
ODDS_CANDIDATES = [
    ("AvgH", "AvgD", "AvgA"),      # promedio de mercado (temporadas recientes)
    ("BbAvH", "BbAvD", "BbAvA"),   # promedio de mercado (temporadas antiguas)
    ("B365H", "B365D", "B365A"),   # Bet365 como respaldo
]

# Unificación de nombres de equipos. Ampliar si aparecen variantes.
TEAM_ALIASES = {
    "Man United": "Manchester United",
    "Man City": "Manchester City",
    "Nott'm Forest": "Nottingham Forest",
    "Sheffield United": "Sheffield Utd",
    "Spurs": "Tottenham",
}


def _pick_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Toma las mejores cuotas disponibles fila a fila."""
    out = pd.DataFrame(index=df.index, columns=["odds_H", "odds_D", "odds_A"], dtype=float)
    for h, d, a in ODDS_CANDIDATES:
        if {h, d, a} <= set(df.columns):
            fill = df[[h, d, a]].astype(float).set_axis(out.columns, axis=1)
            out = out.fillna(fill)
    return out


def clean_season(df: pd.DataFrame, season: str) -> pd.DataFrame:
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTR"]).copy()
    out = pd.DataFrame({
        "date": pd.to_datetime(df["Date"], dayfirst=True, format="mixed"),
        "season": season,
        "home": df["HomeTeam"].str.strip().replace(TEAM_ALIASES),
        "away": df["AwayTeam"].str.strip().replace(TEAM_ALIASES),
        "home_goals": df["FTHG"].astype(int),
        "away_goals": df["FTAG"].astype(int),
        "result": df["FTR"].str.strip(),
    })
    return pd.concat([out, _pick_odds(df)], axis=1)


def load_raw(seasons=SEASONS, league=LEAGUE) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = DATA_RAW / f"{league}_{season}.csv"
        raw = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
        frames.append(clean_season(raw, season))
    df = pd.concat(frames, ignore_index=True)
    return finalize(df)


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Orden cronológico estable y un id por partido."""
    df = df.sort_values(["date", "home"], kind="mergesort").reset_index(drop=True)
    df.insert(0, "match_id", range(len(df)))
    assert df["result"].isin(["H", "D", "A"]).all(), "resultado inválido"
    assert not df.duplicated(["date", "home", "away"]).any(), "partidos duplicados"
    return df


if __name__ == "__main__":
    matches = load_raw()
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    matches.to_csv(DATA_PROCESSED / "matches.csv", index=False)
    print(f"{len(matches)} partidos, {matches['home'].nunique()} equipos")
    print(matches["result"].value_counts(normalize=True).round(3))
