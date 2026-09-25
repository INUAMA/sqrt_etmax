import numpy as np
import pytest

from sqrt_etmax import sqrt_etmax


def test_ppf_en_limite_del_atomo_devuelve_cero():
    k = 2.0
    p0 = np.exp(-k)

    resultado = sqrt_etmax.ppf(p0, k)

    assert resultado == 0.0

def test_ppf_cerca_del_atomo_coincide_con_referencia():
    k = 2.0
    p = 0.1353352833366127

    # Referencia independiente: bisección con Decimal a 70 y 100 dígitos.
    # Resolver (1 + u) * exp(-u) = -log(p) / k; el cuantil es u**2.
    # Se usa el valor binario exacto de p mediante Decimal.from_float.
    esperado = 7.389191383687398e-10

    resultado = sqrt_etmax.ppf(p, k)

    assert resultado == pytest.approx(
        esperado, rel=1e-6, abs=0.0
    )
@pytest.mark.parametrize("k", [0.1, 2.0, 10.0, 100.0])
def test_ppf_vector_respeta_atomo_y_orden(k):
    p0 = np.exp(-k)
    probabilidades = np.array([
        0.0,
        p0 / 2.0,
        np.nextafter(p0, 0.0),
        p0,
        np.nextafter(p0, 1.0),
        p0 + (1.0 - p0) * 1e-8,
        (1.0 + p0) / 2.0,
        np.nextafter(1.0, 0.0),
        1.0,
    ])

    resultado = sqrt_etmax.ppf(probabilidades, k)

    np.testing.assert_array_equal(resultado[:4], np.zeros(4))
    assert np.all(np.isfinite(resultado[4:-1]))
    assert np.all(resultado[4:-1] > 0.0)
    assert resultado[-1] == np.inf
    assert np.all(np.diff(resultado) >= 0.0)


@pytest.mark.parametrize("loc, scale", [(0.0, 1.0), (5.0, 3.0)])
def test_ppf_congelada_respeta_atomo_y_transformacion(loc, scale):
    k = 2.0
    p0 = np.exp(-k)
    probabilidades = np.array([0.0, p0 / 2.0, p0, 0.5, 0.9])
    dist = sqrt_etmax(k, loc=loc, scale=scale)

    resultado = dist.ppf(probabilidades)

    np.testing.assert_array_equal(resultado[:3], np.full(3, loc))
    esperado = loc + scale * sqrt_etmax.ppf(probabilidades[3:], k)
    np.testing.assert_allclose(
        resultado[3:], esperado, rtol=1e-13, atol=0.0
    )


def test_ppf_admite_broadcasting_de_probabilidades_y_k():
    k = np.array([[0.1, 2.0, 10.0]])
    x = np.array([[0.0], [0.01], [1.0], [10.0]])
    raiz = np.sqrt(x)
    probabilidades = np.exp(-k * (1.0 + raiz) * np.exp(-raiz))

    resultado = sqrt_etmax.ppf(probabilidades, k)

    esperado = np.broadcast_to(x, (4, 3))
    assert resultado.shape == (4, 3)
    np.testing.assert_allclose(
        resultado, esperado, rtol=1e-10, atol=0.0
    )