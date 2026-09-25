import numpy as np
import pytest

from sqrt_etmax import sqrt_etmax


@pytest.mark.parametrize(
    "datos, tau2_esperado",
    [
        (
            sqrt_etmax.ppf((np.arange(52) + 0.5) / 52, 1e8),
            "0.066661",
        ),
        (np.array([0.0, 0.0, 1.0]), "1.000000"),
    ],
    ids=["tau2_inferior", "tau2_superior"],
)
def test_lmoments_identifica_limite_del_intervalo_de_busqueda(
    datos, tau2_esperado
):
    with pytest.raises(
        RuntimeError, match="intervalo de búsqueda"
    ) as error:
        sqrt_etmax.fit_lmoments(datos)

    mensaje = str(error.value)

    assert f"τ₂ muestral ({tau2_esperado})" in mensaje
    assert "rango numérico (0.106800, 0.999625)" in mensaje
    assert "k [0.001, 50000]" in mensaje
    assert "La familia no puede representar estos datos" not in mensaje
