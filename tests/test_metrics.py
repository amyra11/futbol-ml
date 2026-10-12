import numpy as np
import pytest

from src.evaluation.metrics import brier_multiclass, evaluate, implied_probs
from src.models.baseline import FrequencyBaseline
from src.splits import split


def test_perfect_prediction():
    y = ["H", "D", "A"]
    probs = np.eye(3)
    assert brier_multiclass(y, probs) == pytest.approx(0.0)
    assert evaluate(y, np.clip(probs, 1e-9, 1) / np.clip(probs, 1e-9, 1).sum(1, keepdims=True))["accuracy"] == 1.0


def test_uniform_log_loss_is_log3():
    y = ["H", "D", "A", "H"]
    probs = np.full((4, 3), 1 / 3)
    assert evaluate(y, probs)["log_loss"] == pytest.approx(np.log(3))


def test_log_loss_respects_hda_order():
    """Asimétrico a propósito: detecta si alguien reordena las columnas."""
    y = ["H", "H"]
    probs = np.array([[0.7, 0.2, 0.1], [0.7, 0.2, 0.1]])
    assert evaluate(y, probs)["log_loss"] == pytest.approx(-np.log(0.7))


def test_to_hda_reorders_sklearn_columns():
    from sklearn.linear_model import LogisticRegression
    from src.evaluation.metrics import to_hda
    X = np.arange(30).reshape(-1, 1)
    y = np.array(["H"] * 10 + ["D"] * 10 + ["A"] * 10)
    model = LogisticRegression().fit(X, y)
    assert list(model.classes_) == ["A", "D", "H"]
    p = to_hda(model, model.predict_proba([[0]]))
    assert p[0].argmax() == 0  # X=0 está en la zona de "H"


def test_implied_probs_remove_margin(matches):
    p = implied_probs(matches)
    assert np.allclose(p.sum(axis=1), 1)
    assert (p > 0).all()


def test_baseline_matches_train_frequencies(matches):
    train, val = split(matches)
    assert len(train) and len(val)
    assert train["date"].max() < val["date"].min(), "train debe ser anterior a val"
    model = FrequencyBaseline().fit(None, train["result"])
    probs = model.predict_proba(val)
    freq = train["result"].value_counts(normalize=True)
    assert probs[0] == pytest.approx([freq["H"], freq.get("D", 0), freq["A"]])


def test_team_aliases_applied(matches):
    assert "Man City" not in set(matches["home"])
    assert "Manchester City" in set(matches["home"])
