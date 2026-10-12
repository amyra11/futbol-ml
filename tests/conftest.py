"""Datos sintéticos con el mismo formato que football-data.co.uk."""
import numpy as np
import pandas as pd
import pytest

from src.data.clean import clean_season, finalize

TEAMS = ["Arsenal", "Chelsea", "Liverpool", "Man City", "Man United", "Spurs"]


def make_raw_season(season_start: int, seed: int) -> pd.DataFrame:
    """Doble round-robin con fechas semanales y cuotas aleatorias."""
    rng = np.random.default_rng(seed)
    rows, day = [], pd.Timestamp(f"{season_start}-08-10")
    for h in TEAMS:
        for a in TEAMS:
            if h == a:
                continue
            hg, ag = rng.poisson(1.5), rng.poisson(1.1)
            ftr = "H" if hg > ag else ("A" if ag > hg else "D")
            odds = 1 / rng.dirichlet([4, 3, 3]) * 0.95
            rows.append({"Date": day.strftime("%d/%m/%Y"), "HomeTeam": h, "AwayTeam": a,
                         "FTHG": hg, "FTAG": ag, "FTR": ftr,
                         "B365H": odds[0], "B365D": odds[1], "B365A": odds[2]})
            day += pd.Timedelta(days=int(rng.integers(3, 8)))
    return pd.DataFrame(rows)


@pytest.fixture
def matches() -> pd.DataFrame:
    seasons = [(2022, "2223"), (2023, "2324")]
    frames = [clean_season(make_raw_season(y, i), s) for i, (y, s) in enumerate(seasons)]
    return finalize(pd.concat(frames, ignore_index=True))
