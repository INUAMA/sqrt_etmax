from types import SimpleNamespace
import numpy as np
import pytest
from sqrt_etmax import sqrt_etmax

@pytest.mark.parametrize("alpha", [0.1, 1.0, 10.0])
def test_un_cero_aporta_menos_k(alpha):
    resultado = sqrt_etmax.log_likelihood(
        [0.0], k=2.0, alpha=alpha
    )

    assert resultado == pytest.approx(-2.0)

@pytest.mark.parametrize(
    "datos,k,alpha,esperado",
    [
        ([1.0, 4.0, 8.0], 2.0, 0.7, -9.172349163544846),
        ([0.0, 1.0, 4.0, 8.0], 2.0, 0.7, -11.172349163544846),
        ([0.0, 0.0, 0.0], 2.0, 0.7, -6.0),
        ([1.0], 2.0, 1.0, -1.0 - 4.0 / np.e),
        ([1.0, 1.0], 2.0, 1.0, -2.0 - 8.0 / np.e),
    ],
    ids=[
        "positivos",
        "mixta",
        "todos_ceros",
        "un_positivo",
        "constante_positiva",
    ],
)
def test_log_likelihood_coincide_con_referencia(
    datos, k, alpha, esperado
):
    resultado = sqrt_etmax.log_likelihood(datos, k, alpha)

    assert isinstance(resultado, float)
    assert resultado == pytest.approx(
        esperado, rel=1e-12, abs=1e-12
    )

@pytest.mark.parametrize(
    "datos,mensaje",
    [
        ([], "al menos 1"),
        (1.0, "unidimensional"),
        ([[1.0, 2.0]], "unidimensional"),
        ([1.0, np.nan], "finitos"),
        ([1.0, np.inf], "finitos"),
        ([-1.0, 2.0], "no negativos"),
        ([1.0 + 1.0j, 2.0], "complejos"),
        (
            np.ma.array([1.0, 2.0], mask=[False, True]),
            "enmascaradas",
        ),
    ],
    ids=[
        "vacia",
        "escalar",
        "bidimensional",
        "nan",
        "infinito",
        "negativo",
        "complejo",
        "enmascarada",
    ],
)
def test_log_likelihood_rechaza_muestras_invalidas(datos, mensaje):
    with pytest.raises(ValueError, match=mensaje):
        sqrt_etmax.log_likelihood(datos, k=2.0, alpha=0.7)

def test_log_likelihood_no_modifica_la_muestra():
    datos = np.array([8.0, 0.0, 1.0, 4.0])
    original = datos.copy()

    sqrt_etmax.log_likelihood(datos, k=2.0, alpha=0.7)

    np.testing.assert_array_equal(datos, original)

@pytest.mark.parametrize("nombre", ["k", "alpha"])
@pytest.mark.parametrize(
    "valor",
    [0.0, -1.0, np.nan, np.inf, True, "2"],
    ids=["cero", "negativo", "nan", "infinito", "booleano", "texto"],
)
def test_log_likelihood_rechaza_parametros_invalidos(nombre, valor):
    parametros = {"k": 2.0, "alpha": 0.7}
    parametros[nombre] = valor

    with pytest.raises(ValueError, match=nombre):
        sqrt_etmax.log_likelihood([0.0, 1.0, 4.0], **parametros)

@pytest.mark.parametrize(
    "datos",
    [
        [1.0, 4.0, 8.0],
        [0.0, 1.0, 4.0, 8.0],
    ],
    ids=["positivos", "mixta"],
)
def test_objetivo_mle_coincide_con_log_likelihood(monkeypatch, datos):
    parametros = np.array([2.0, 0.7])
    esperado = -sqrt_etmax.log_likelihood(
        datos, k=parametros[0], alpha=parametros[1]
    )
    evaluado = {}

    def minimizar(objetivo, x0, args=(), **kwargs):
        valor = objetivo(parametros, *args)
        evaluado["valor"] = valor
        return SimpleNamespace(
            success=True,
            x=parametros.copy(),
            fun=valor,
        )

    monkeypatch.setattr(
        "sqrt_etmax.distribution.optimize.minimize",
        minimizar,
    )

    sqrt_etmax.fit_custom(datos)

    assert evaluado["valor"] == pytest.approx(
        esperado, rel=1e-12, abs=1e-12
    )