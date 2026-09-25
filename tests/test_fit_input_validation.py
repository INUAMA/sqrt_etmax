import numpy as np
import pytest

from sqrt_etmax import sqrt_etmax


@pytest.mark.parametrize("metodo", ["fit_custom", "fit_lmoments"])
def test_ajuste_equivalente_con_lista_y_array(metodo):
    """La misma muestra debe producir el mismo ajuste con lista y array."""
    datos = [0.0, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0]
    ajustar = getattr(sqrt_etmax, metodo)

    resultado_array = ajustar(np.array(datos, dtype=float))
    resultado_lista = ajustar(datos)

    np.testing.assert_allclose(
        resultado_lista,
        resultado_array,
        rtol=1e-12,
        atol=1e-12,
        equal_nan=False
    )

@pytest.mark.parametrize("metodo", ["fit_custom", "fit_lmoments"])
@pytest.mark.parametrize(
    "datos, mensaje",
    [
        pytest.param([], "al menos 2", id="vacia"),
        pytest.param([3.0], "al menos 2", id="un_dato"),
        pytest.param(3.0, "unidimensional", id="escalar"),
        pytest.param(
            [[1.0, 4.0], [8.0, 12.0]],
            "unidimensional",
            id="bidimensional",
        ),
    ],
)
def test_ajuste_rechaza_dimension_o_tamano_invalidos(
    metodo, datos, mensaje
):
    """El ajuste exige una muestra unidimensional con al menos dos datos."""
    ajustar = getattr(sqrt_etmax, metodo)

    with pytest.raises(ValueError, match=mensaje):
        ajustar(datos)

@pytest.mark.parametrize("metodo", ["fit_custom", "fit_lmoments"])
@pytest.mark.parametrize(
    "datos, mensaje",
    [
        pytest.param(
            [1.0, np.nan, 3.0], "finitos", id="nan"
        ),
        pytest.param(
            [1.0, np.inf, 3.0], "finitos", id="infinito_positivo"
        ),
        pytest.param(
            [1.0, -np.inf, 3.0], "finitos", id="infinito_negativo"
        ),
        pytest.param(
            [-1.0, 1.0, 4.0], "no negativos", id="negativo"
        ),
        pytest.param(
            [3.0, 3.0, 3.0], "valores distintos", id="constante"
        ),
        pytest.param(
            [0.0, 0.0, 0.0], "valores distintos", id="todos_ceros"
        ),
    ],
)
def test_ajuste_rechaza_valores_invalidos(metodo, datos, mensaje):
    """Rechaza muestras con valores incompatibles con el ajuste."""
    ajustar = getattr(sqrt_etmax, metodo)

    with pytest.raises(ValueError, match=mensaje):
        ajustar(datos)

@pytest.mark.parametrize("metodo", ["fit_custom", "fit_lmoments"])
@pytest.mark.parametrize(
    "datos, mensaje",
    [
        pytest.param(
            ["lluvia", 2.0], "numéricos reales", id="texto"
        ),
        pytest.param(
            [object(), 2.0], "numéricos reales", id="objeto"
        ),
        pytest.param(
            [1.0 + 2.0j, 3.0], "complejos", id="lista_compleja"
        ),
        pytest.param(
            np.array([1.0 + 2.0j, 3.0]),
            "complejos",
            id="array_complejo",
        ),
        pytest.param(
            np.array([1.0, 3.0], dtype=complex),
            "complejos",
            id="complejos_sin_parte_imaginaria",
        ),
        pytest.param(
            np.ma.array([1.0, 2.0, 4.0], mask=[False, True, False]),
            "enmascaradas",
            id="mascara_parcial",
        ),
        pytest.param(
            np.ma.array([1.0, 2.0, 4.0], mask=True),
            "enmascaradas",
            id="mascara_total",
        ),
        pytest.param(
                    np.array(
                        [np.complex128(1.0 + 2.0j), 3.0, 8.0],
                        dtype=object,
                    ),
                    "complejos",
                    id="complejo_dentro_de_array_object",
                ),
    ],
)
def test_ajuste_rechaza_tipos_o_mascaras_invalidos(
    metodo, datos, mensaje
):
    """Rechaza entradas que no representan una muestra real completa."""
    ajustar = getattr(sqrt_etmax, metodo)

    with pytest.raises(ValueError, match=mensaje):
        ajustar(datos)

@pytest.mark.parametrize("metodo", ["fit_custom", "fit_lmoments"])
@pytest.mark.parametrize(
    "datos",
    [
        [4, 0, 12, 1, 0, 8, 2, 6],
        np.array([4, 0, 12, 1, 0, 8, 2, 6], dtype=int),
        np.array([4, 0, 12, 1, 0, 8, 2, 6], dtype=float),
        np.ma.array([4, 0, 12, 1, 0, 8, 2, 6], mask=False),
    ],
    ids=["lista", "enteros", "flotantes", "sin_datos_enmascarados"],
)
def test_ajuste_conserva_muestras_validas(metodo, datos):
    """Acepta muestras válidas sin cambiar sus valores, orden o máscara."""
    ajustar = getattr(sqrt_etmax, metodo)
    original = datos.copy()
    referencia = np.asarray(original, dtype=float).copy()

    esperado = ajustar(referencia)
    resultado = ajustar(datos)

    np.testing.assert_allclose(
        resultado,
        esperado,
        rtol=1e-12,
        atol=1e-12,
        equal_nan=False,
    )
    np.testing.assert_array_equal(datos, original)

    if np.ma.isMaskedArray(datos):
        np.testing.assert_array_equal(
            np.ma.getmaskarray(datos),
            np.ma.getmaskarray(original),
        )