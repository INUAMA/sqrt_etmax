import numpy as np
import pytest

from sqrt_etmax import sqrt_etmax


@pytest.mark.parametrize(
    "metodo, esperado",
    [
        ("sf", -np.expm1(-2.0)),
        ("logcdf", -2.0),
        ("logsf", np.log(-np.expm1(-2.0))),
    ],
)

def test_probabilidades_en_cero(metodo, esperado):
    funcion = getattr(sqrt_etmax, metodo)

    assert funcion(0.0, 2.0) == pytest.approx(esperado)

def test_cdf_en_cero_incluye_masa():
    k = 2.0
    resultado = sqrt_etmax.cdf(0.0, k)
    esperado = np.exp(-k)
    
    assert resultado == pytest.approx(esperado)

@pytest.mark.parametrize("loc, scale", [(0.0, 1.0), (5.0, 3.0)])
@pytest.mark.parametrize("metodo", ["cdf", "sf", "logcdf", "logsf"])
def test_probabilidades_congeladas_en_el_atomo(loc, scale, metodo):
    k = 2.0
    distribucion = sqrt_etmax(k, loc=loc, scale=scale)

    puntos = np.array([loc - scale, loc])
    esperados = {
        "cdf": [0.0, np.exp(-k)],
        "sf": [1.0, -np.expm1(-k)],
        "logcdf": [-np.inf, -k],
        "logsf": [0.0, np.log(-np.expm1(-k))],
    }

    np.testing.assert_allclose(
        getattr(distribucion, metodo)(puntos),
        esperados[metodo],
        rtol=1e-12,
        atol=0.0,
    )

def test_logcdf_en_atomo_con_k_grande():
    assert sqrt_etmax.logcdf(0.0, 1000.0) == pytest.approx(-1000.0)


def test_sf_en_atomo_con_k_pequeno():
    k = 1e-20
    esperado = -np.expm1(-k)

    assert sqrt_etmax.sf(0.0, k) == pytest.approx(
        esperado, rel=1e-12, abs=0.0
    )


def test_logsf_en_atomo_con_k_pequeno():
    k = 1e-20
    esperado = np.log(-np.expm1(-k))

    assert sqrt_etmax.logsf(0.0, k) == pytest.approx(esperado) 
    
