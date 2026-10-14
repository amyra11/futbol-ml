import numpy as np

from src.features.build import build_features
from src.models.elo_logistic import EloLogistic


def test_probabilities_are_valid(matches):
    model = EloLogistic().fit(build_features(matches))
    p = model.predict_proba_teams(1600, 1450)
    assert p.shape == (3,)
    assert np.isclose(p.sum(), 1.0)
    assert ((p > 0) & (p < 1)).all()