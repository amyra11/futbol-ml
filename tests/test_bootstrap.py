import numpy as np
import pytest

from src.evaluation.bootstrap import bootstrap_pareado, tabla_bootstrap
from src.evaluation.metrics import log_loss_hda, log_loss_por_partido


@pytest.fixture
def partidos():
    rng = np.random.default_rng(1)
    y = rng.choice(["H", "D", "A"], size=400, p=[0.45, 0.25, 0.30])
    uniforme = np.full((400, 3), 1 / 3)
    # Modelo "bueno": 0.6 al resultado real y 0.2 a los otros dos
    bueno = np.full((400, 3), 0.2)
    bueno[np.arange(400), [["H", "D", "A"].index(c) for c in y]] = 0.6
    return y, uniforme, bueno


def test_log_loss_por_partido_promedia_a_log_loss_hda(partidos):
    y, uniforme, bueno = partidos
    assert log_loss_por_partido(y, bueno).mean() == pytest.approx(log_loss_hda(y, bueno))
    assert np.allclose(log_loss_por_partido(y, uniforme), np.log(3))


def test_mismo_modelo_diferencia_cero(partidos):
    y, uniforme, _ = partidos
    r = bootstrap_pareado(y, uniforme, uniforme)
    assert r["diferencia"] == r["ic_bajo"] == r["ic_alto"] == 0
    assert not r["significativa"]


def test_modelo_claramente_mejor_es_significativo(partidos):
    y, uniforme, bueno = partidos
    r = bootstrap_pareado(y, bueno, uniforme)
    assert r["diferencia"] == pytest.approx(np.log(0.6 / (1 / 3)) * -1)
    assert r["ic_bajo"] <= r["diferencia"] <= r["ic_alto"] < 0
    assert r["significativa"] and r["prop_mejor"] == 1


def test_semilla_fija_reproducible(partidos):
    y, uniforme, _ = partidos
    rng = np.random.default_rng(2)
    ruido = rng.dirichlet([5, 3, 4], size=len(y))
    assert bootstrap_pareado(y, ruido, uniforme) == bootstrap_pareado(y, ruido, uniforme)
    assert bootstrap_pareado(y, ruido, uniforme, seed=1) != bootstrap_pareado(y, ruido, uniforme, seed=2)


def test_tabla_excluye_la_referencia(partidos):
    y, uniforme, bueno = partidos
    t = tabla_bootstrap(y, {"ref": uniforme, "bueno": bueno}, "ref", n_boot=200)
    assert list(t["modelo"]) == ["bueno"]
