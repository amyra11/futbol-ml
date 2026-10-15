import pandas as pd

from src.features.current import recent_form


def test_recent_form_counts_points_and_goals_from_team_perspective():
    f = pd.DataFrame({
        "match_id": [0, 1, 2],
        "home": ["A", "C", "A"], "away": ["B", "A", "D"],
        "home_goals": [2, 1, 0], "away_goals": [0, 1, 3],
    })
    r = recent_form(f, "A", window=3)
    assert r["racha"] == "VED"
    assert r["pts"] == (3 + 1 + 0) / 3
    assert r["gf"] == (2 + 1 + 0) / 3
    assert r["ga"] == (0 + 1 + 3) / 3