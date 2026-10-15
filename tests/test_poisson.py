import numpy as np
import pytest

from src.features.build import build_features
from src.models.poisson import PoissonElo, probs_hda


def test_probs_sum_to_one_in_order_h_d_a():
    p = probs_hda(2.4, 0.7)
    assert p.sum() == pytest.approx(1.0)
    assert p[0] > p[1] > p[2]            # local muy favorito: H > D > A


def test_even_match_is_symmetric():
    p = probs_hda(1.3, 1.3)
    assert p[0] == pytest.approx(p[2])


def test_model_predicts_valid_probabilities(matches):
    f = build_features(matches)
    p = PoissonElo().fit(f).predict_proba(f)
    assert p.shape == (len(f), 3)
    assert np.allclose(p.sum(axis=1), 1.0)

import pandas as pd

from src.config import ELO_HOME_ADV
from src.models.poisson import score_matrix


def test_score_matrix_sums_to_about_one():
    assert score_matrix(1.3, 1.1).sum() == pytest.approx(1.0, abs=1e-3)


def test_predict_teams_matches_expected_goals(matches):
    f = build_features(matches)
    modelo = PoissonElo().fit(f)
    lam_l, lam_v = modelo.predict_teams(1600, 1450)
    esperado = modelo.expected_goals(pd.DataFrame({"elo_diff": [1600 + ELO_HOME_ADV - 1450]}))
    assert lam_l == pytest.approx(esperado[0][0]) and lam_v == pytest.approx(esperado[1][0])
    assert lam_l > 0 and lam_v > 0