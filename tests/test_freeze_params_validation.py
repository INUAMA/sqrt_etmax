import numpy as np
import pytest

from sqrt_etmax import sqrt_etmax


@pytest.mark.parametrize("parametro", ["k", "alpha"])
@pytest.mark.parametrize(
    "valor",
    [0.0, -1.0, np.nan, np.inf, -np.inf],
    ids=["cero", "negativo", "nan", "infinito", "menos_infinito"],
)
def test_freeze_params_rechaza_valores_invalidos(parametro, valor):
    """Rechaza parámetros no positivos o no finitos antes de congelar."""
    parametros = {"k": 2.0, "alpha": 0.7}
    parametros[parametro] = valor

    with pytest.raises(ValueError, match=parametro):
        sqrt_etmax.freeze_params(**parametros)

@pytest.mark.parametrize("parametro", ["k", "alpha"])
@pytest.mark.parametrize(
    "valor",
    [
        pytest.param(True, id="booleano_python"),
        pytest.param(np.bool_(True), id="booleano_numpy"),
        pytest.param("2.0", id="texto_numerico"),
        pytest.param(None, id="none"),
        pytest.param(2.0 + 0.0j, id="complejo_python"),
        pytest.param(
            np.complex128(2.0 + 1.0j),
            id="complejo_numpy",
        ),
        pytest.param(np.ma.masked, id="valor_enmascarado"),
        pytest.param(
            np.ma.array(2.0, mask=True),
            id="array_enmascarado",
        ),
        pytest.param([2.0], id="lista"),
        pytest.param(np.array(2.0), id="array_cero_dimensiones"),
        pytest.param(np.array([2.0]), id="array_un_elemento"),
        pytest.param(np.array([2.0, 3.0]), id="array_varios_elementos"),
    ],
)
def test_freeze_params_rechaza_tipos_invalidos(parametro, valor):
    """Exige parámetros escalares reales del tipo admitido."""
    parametros = {"k": 2.0, "alpha": 0.7}
    parametros[parametro] = valor

    with pytest.raises(ValueError, match=parametro):
        sqrt_etmax.freeze_params(**parametros)

def test_freeze_params_rechaza_escala_no_representable():
    """Rechaza un alpha cuya inversa no produce una escala finita."""
    with pytest.raises(ValueError, match="alpha.*escala"):
        sqrt_etmax.freeze_params(k=2.0, alpha=1e-320)
        
@pytest.mark.parametrize(
    "k, alpha",
    [
        pytest.param(2, 3, id="enteros_python"),
        pytest.param(2.0, 0.7, id="flotantes_python"),
        pytest.param(np.int64(2), np.int64(3), id="enteros_numpy"),
        pytest.param(
            np.float32(2.0), np.float32(0.7), id="float32"
        ),
        pytest.param(
            np.float64(2.0), np.float64(0.7), id="float64"
        ),
    ],
)
def test_freeze_params_conserva_distribucion_valida(k, alpha):
    """Conserva la CDF, el átomo y los cuantiles con parámetros válidos."""
    dist = sqrt_etmax.freeze_params(k, alpha)
    k_real = float(k)
    alpha_real = float(alpha)

    x = np.array([0.0, 0.5, 2.0, 10.0])
    raiz = np.sqrt(alpha_real * x)
    cdf_esperada = np.exp(
        -k_real * (1.0 + raiz) * np.exp(-raiz)
    )

    np.testing.assert_allclose(
        dist.cdf(x), cdf_esperada, rtol=1e-12, atol=0.0
    )
    assert dist.cdf(-1.0) == 0.0

    probabilidades = np.array([0.5, 0.9, 0.99])
    cuantiles_esperados = (
        sqrt_etmax.ppf(probabilidades, k_real) / alpha_real
    )

    np.testing.assert_allclose(
        dist.ppf(probabilidades),
        cuantiles_esperados,
        rtol=1e-12,
        atol=0.0,
    )


@pytest.mark.parametrize("parametro", ["k", "alpha"])
def test_freeze_params_rechaza_entero_fuera_del_rango_float(parametro):
    """Da un error explícito si el entero no es representable como float."""
    parametros = {"k": 2.0, "alpha": 0.7}
    parametros[parametro] = 10**400

    with pytest.raises(ValueError, match=parametro):
        sqrt_etmax.freeze_params(**parametros)