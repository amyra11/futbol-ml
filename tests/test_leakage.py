"""El test más importante del proyecto: ninguna feature puede ver el futuro.

Idea: si cambiamos los resultados de todos los partidos posteriores a una fecha de corte,
las features de los partidos hasta esa fecha (inclusive) deben quedar idénticas.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.build import FEATURE_COLS, build_features


def _scramble_after(df: pd.DataFrame, cutoff: pd.Timestamp, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.copy()
    future = df["date"] > cutoff
    n = int(future.sum())
    df.loc[future, "home_goals"] = rng.integers(0, 6, n)
    df.loc[future, "away_goals"] = rng.integers(0, 6, n)
    hg, ag = df.loc[future, "home_goals"], df.loc[future, "away_goals"]
    df.loc[future, "result"] = np.where(hg > ag, "H", np.where(ag > hg, "A", "D"))
    return df


@pytest.mark.parametrize("quantile", [0.1, 0.5, 0.9])
def test_features_do_not_see_future_results(matches, quantile):
    cutoff = matches["date"].quantile(quantile)
    original = build_features(matches)
    scrambled = build_features(_scramble_after(matches, cutoff))
    past = original["date"] <= cutoff
    pd.testing.assert_frame_equal(
        original.loc[past, FEATURE_COLS].reset_index(drop=True),
        scrambled.loc[past, FEATURE_COLS].reset_index(drop=True),
    )


def test_current_match_not_in_own_features(matches):
    """Cambiar SOLO el resultado de un partido no cambia sus propias features."""
    i = len(matches) // 2
    original = build_features(matches)
    m2 = matches.copy()
    m2.loc[i, ["home_goals", "away_goals", "result"]] = [9, 0, "H"]
    changed = build_features(m2)
    pd.testing.assert_series_equal(original.loc[i, FEATURE_COLS], changed.loc[i, FEATURE_COLS])


def test_first_match_has_no_form(matches):
    f = build_features(matches)
    first = f.iloc[0]
    assert np.isnan(first["home_form_pts"]) and np.isnan(first["away_form_pts"])
    assert first["home_elo"] == first["away_elo"] == 1500.0
