import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.models.common import LABEL_A_INT, proba_hda


@pytest.fixture
def datos():
    """y depende de x para que cada clase tenga probabilidades distintas y reconocibles."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 2))
    y = np.where(X[:, 0] > 0.5, "H", np.where(X[:, 0] < -0.5, "A", "D"))
    return X, y


def test_label_a_int():
    assert LABEL_A_INT == {"H": 0, "D": 1, "A": 2}


def test_logistica_con_etiquetas_de_texto(datos):
    X, y = datos
    m = LogisticRegression().fit(X, y)
    assert list(m.classes_) == ["A", "D", "H"]                  # orden alfabético de sklearn
    p, crudo = proba_hda(m, X), m.predict_proba(X)
    for j, c in enumerate(["H", "D", "A"]):
        assert np.allclose(p[:, j], crudo[:, list(m.classes_).index(c)])
    assert np.allclose(p.sum(axis=1), 1)


def test_xgboost_con_etiquetas_numericas(datos):
    X, y = datos
    y_int = np.array([LABEL_A_INT[c] for c in y])
    m = XGBClassifier(n_estimators=20, max_depth=2).fit(X, y_int)
    assert list(m.classes_) == [0, 1, 2]
    p, crudo = proba_hda(m, X), m.predict_proba(X)
    for j, c in enumerate(["H", "D", "A"]):
        assert np.allclose(p[:, j], crudo[:, LABEL_A_INT[c]])
    # Coherencia: la columna H es la más alta justo cuando el resultado real es H
    assert (np.array(["H", "D", "A"])[p.argmax(axis=1)] == y).mean() > 0.9


def test_mismo_orden_con_ambos_tipos_de_etiqueta(datos):
    """El mismo árbol entrenado con texto y con números debe dar la misma matriz H, D, A."""
    X, y = datos
    from sklearn.tree import DecisionTreeClassifier
    texto = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
    numeros = DecisionTreeClassifier(max_depth=3, random_state=0).fit(
        X, np.array([LABEL_A_INT[c] for c in y]))
    assert np.allclose(proba_hda(texto, X), proba_hda(numeros, X))
