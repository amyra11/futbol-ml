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

def test_final_elo_matches_manual_update_and_add_elo_is_unchanged():
    from src.features.build import final_elo

    df = pd.DataFrame({
        "match_id": [0], "date": pd.to_datetime(["2022-08-10"]), "season": ["2223"],
        "home": ["A"], "away": ["B"], "result": ["H"],
    })
    r = final_elo(df, k=30, home_adv=60, init=1500)
    expected_h = 1 / (1 + 10 ** (-60 / 400))
    delta = 30 * (1 - expected_h)
    assert r["A"] == 1500 + delta and r["B"] == 1500 - delta
    out = add_elo(df, k=30, home_adv=60)
    assert out.loc[0, "home_elo"] == 1500 and out.loc[0, "away_elo"] == 1500