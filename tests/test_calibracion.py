import numpy as np
import pytest

from src.evaluation.calibracion import datos_calibracion, ece, ece_por_azar, ece_por_clase


def _simular(n: int, seed: int = 0):
    """Probabilidades al azar y resultados sorteados CON esas probabilidades:
    un modelo perfectamente calibrado por construcción."""
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet([4, 2.5, 3], size=n)
    u = rng.random(n)[:, None]
    idx = (u > probs.cumsum(axis=1)).sum(axis=1)
    return np.array(["H", "D", "A"])[idx], probs


def test_modelo_perfectamente_calibrado_ece_cercano_a_cero():
    y, probs = _simular(200_000)
    assert ece(y, probs) < 0.01
    assert all(v < 0.01 for v in ece_por_clase(y, probs).values())


def test_modelo_sobreconfiado_tiene_ece_alto():
    """Llevar las probabilidades a los extremos (temperatura < 1) rompe la calibración."""
    y, probs = _simular(200_000)
    extremo = probs ** 3 / (probs ** 3).sum(axis=1, keepdims=True)
    assert ece(y, extremo) > 0.05 > ece(y, probs)


def test_bins_por_cuantiles_sin_vacios():
    y, probs = _simular(1000)
    for clase in ["H", "D", "A"]:
        d = datos_calibracion(y, probs, clase, n_bins=10)
        assert len(d) == 10
        assert d["n"].sum() == 1000 and (d["n"] >= 99).all()
        assert d["prob_media"].is_monotonic_increasing
        assert d["frec_observada"].between(0, 1).all()


def test_ece_por_azar_crece_con_pocos_partidos():
    """Con 380 partidos un modelo perfecto ya tiene ECE claramente > 0; con muchos, casi 0."""
    _, pocos = _simular(380)
    _, muchos = _simular(20_000)
    azar_pocos = ece_por_azar(pocos, n_sim=50)
    assert azar_pocos.shape == (50,)
    assert np.median(azar_pocos) > 0.03
    assert np.median(ece_por_azar(muchos, n_sim=5)) < 0.015
    assert np.array_equal(azar_pocos, ece_por_azar(pocos, n_sim=50))      # semilla fija


def test_columna_correcta_por_clase():
    """La frecuencia observada de la clase D se calcula con la columna D (índice 1)."""
    y = np.array(["D"] * 50 + ["H"] * 50)
    probs = np.tile([0.5, 0.3, 0.2], (100, 1))
    d = datos_calibracion(y, probs, "D")
    assert len(d) == 1                                    # probabilidades iguales → un solo bin
    assert d.loc[0, "prob_media"] == pytest.approx(0.3)
    assert d.loc[0, "frec_observada"] == pytest.approx(0.5)
    assert ece_por_clase(y, probs)["D"] == pytest.approx(0.2)
