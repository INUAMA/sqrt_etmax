from types import SimpleNamespace

import numpy as np
import pytest

from sqrt_etmax import sqrt_etmax


def test_fit_custom_rechaza_optimizacion_fallida(monkeypatch):
    fallo = SimpleNamespace(
        success=False,
        x=np.array([2.0, 0.7]),
        fun=12.0,
        message="Límite de iteraciones alcanzado",
    )

    def simular_fallo(*args, **kwargs):
        return fallo

    monkeypatch.setattr(
        "sqrt_etmax.distribution.optimize.minimize",
        simular_fallo,
    )

    with pytest.raises(RuntimeError, match="Límite de iteraciones"):
        sqrt_etmax.fit_custom(np.array([1.0, 4.0, 8.0]))
        
@pytest.mark.parametrize(
    "parametros",
    [
        [0.0, 0.7],
        [-2.0, 0.7],
        [2.0, 0.0],
        [2.0, -0.7],
        [np.nan, 0.7],
        [2.0, np.nan],
        [np.inf, 0.7],
        [2.0, np.inf],
    ],
)
def test_fit_custom_rechaza_parametros_invalidos(monkeypatch, parametros):
    resultado = SimpleNamespace(
        success=True,
        x=np.array(parametros),
        fun=12.0,
        message="Optimización terminada",
    )

    def simular_resultado(*args, **kwargs):
        return resultado

    monkeypatch.setattr(
        "sqrt_etmax.distribution.optimize.minimize",
        simular_resultado,
    )

    with pytest.raises(RuntimeError, match="parámetros"):
        sqrt_etmax.fit_custom(np.array([1.0, 4.0, 8.0]))

@pytest.mark.parametrize("objetivo", [np.nan, np.inf, -np.inf])
def test_fit_custom_rechaza_objetivo_no_finito(monkeypatch, objetivo):
    resultado = SimpleNamespace(
        success=True,
        x=np.array([2.0, 0.7]),
        fun=objetivo,
        message="Optimización terminada",
    )

    def simular_resultado(*args, **kwargs):
        return resultado

    monkeypatch.setattr(
        "sqrt_etmax.distribution.optimize.minimize",
        simular_resultado,
    )

    with pytest.raises(RuntimeError, match="objetivo"):
        sqrt_etmax.fit_custom(np.array([1.0, 4.0, 8.0]))
