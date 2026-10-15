import numpy as np

from src.features.build import build_features
from src.models.elo_logistic import EloLogistic


def test_probabilities_are_valid(matches):
    model = EloLogistic().fit(build_features(matches))
    p = model.predict_proba_teams(1600, 1450)
    assert p.shape == (3,)
    assert np.isclose(p.sum(), 1.0)
    assert ((p > 0) & (p < 1)).all()

def test_predict_proba_follows_common_contract(matches):
    from src.features.current import matchup_features

    f = build_features(matches)
    model = EloLogistic().fit(f)
    x = matchup_features(f, "Arsenal", "Chelsea")
    p = model.predict_proba(x)
    assert p.shape == (1, 3)
    assert np.allclose(p[0], model.predict_proba_teams(x.loc[0, "home_elo"], x.loc[0, "away_elo"]))