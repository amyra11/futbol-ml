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

def test_matchup_features_has_feature_cols_and_matches_training_logic(matches):
    from src.features.build import FEATURE_COLS, build_features, final_elo
    from src.features.current import matchup_features

    f = build_features(matches)
    x = matchup_features(f, "Arsenal", "Chelsea", rest_home=5, rest_away=9)
    assert list(x.columns) == FEATURE_COLS and len(x) == 1
    assert x.loc[0, "rest_diff"] == -4
    ratings = final_elo(f)
    assert x.loc[0, "home_elo"] == ratings["Arsenal"]
    assert x.loc[0, "elo_diff"] == x.loc[0, "home_elo"] + 60 - x.loc[0, "away_elo"]
    assert x.loc[0, "home_form_pts"] == recent_form(f, "Arsenal")["pts"]