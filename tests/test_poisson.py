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