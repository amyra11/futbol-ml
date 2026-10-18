import numpy as np
import pandas as pd
import pytest

from src.data.clean import clean_season, finalize
from src.features.build import build_features
from src.models.common import proba_hda
from src.models.final import FEATURES_FINAL
from src.models.trees import tune_random_forest, tune_xgboost
from src.splits import split
from tests.conftest import make_raw_season


@pytest.fixture(scope="module")
def train_val():
    seasons = [(2017 + i, f"{17 + i:02d}{18 + i:02d}") for i in range(7)]
    frames = [clean_season(make_raw_season(y, i), s) for i, (y, s) in enumerate(seasons)]
    return split(build_features(finalize(pd.concat(frames, ignore_index=True))))


# Rejillas de un solo punto: el test comprueba el contrato, no el rendimiento
@pytest.mark.parametrize("tune, grid", [
    (tune_random_forest, {"bosque__max_depth": [3], "bosque__min_samples_leaf": [20]}),
    (tune_xgboost, {"max_depth": [2], "learning_rate": [0.1], "n_estimators": [20]}),
])
def test_arboles_devuelven_probabilidades_hda(train_val, tune, grid):
    train, val = train_val
    assert train[FEATURES_FINAL].isna().any().any()   # hay NaN de inicio de temporada
    s = tune(train, FEATURES_FINAL, grid=grid)
    p = proba_hda(s.best_estimator_, val[FEATURES_FINAL])
    assert p.shape == (len(val), 3)
    assert np.allclose(p.sum(axis=1), 1)
    assert ((p > 0) & (p < 1)).all()
