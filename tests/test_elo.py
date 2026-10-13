import pandas as pd

from src.features.build import add_elo


def test_new_team_after_first_season_starts_at_promoted_init():
    df = pd.DataFrame({
        "match_id": [0, 1, 2],
        "date": pd.to_datetime(["2022-08-10", "2022-08-17", "2023-08-10"]),
        "season": ["2223", "2223", "2324"],
        "home": ["A", "A", "A"], "away": ["B", "B", "C"],
        "result": ["H", "D", "H"],
    })
    out = add_elo(df)
    assert out.loc[0, "away_elo"] == 1500.0     # B: primera temporada
    assert out.loc[2, "away_elo"] == 1450.0     # C: ascendido