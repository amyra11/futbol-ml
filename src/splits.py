"""Cortes temporales. Todos los notebooks y scripts usan ESTA función."""
import pandas as pd

from src.config import TEST_SEASONS, TRAIN_SEASONS, VAL_SEASONS


def split(df: pd.DataFrame, use_test: bool = False):
    """Devuelve (train, val) o, con use_test=True, (train+val, test).

    use_test=True solo se usa una vez, el miércoles 21, para los resultados finales.
    """
    season = df["season"].astype(str).str.zfill(4)
    train = df[season.isin(TRAIN_SEASONS)]
    val = df[season.isin(VAL_SEASONS)]
    if not use_test:
        return train, val
    test = df[season.isin(TEST_SEASONS)]
    return pd.concat([train, val]), test
