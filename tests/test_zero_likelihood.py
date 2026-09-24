from types import SimpleNamespace
import numpy as np
import pytest
from sqrt_etmax.distribution import sqrt_etmax
from scipy.optimize import minimize_scalar

@pytest.mark.parametrize(
    "k, alpha",
    [
        (2.0, 0.7),
        (2.0, 0.2),
        (0.5, 3.0),
    ],
)


def test_un_cero_aporta_k_a_la_verosimilitud_negativa(monkeypatch, k, alpha):
    evaluaciones = []
    
    def captura_objetivo(funcion, inicio, args, method):
        parametros = np.array([k, alpha])
        evaluaciones.append(float(funcion(parametros, *args)))
        return SimpleNamespace(x=parametros)
    
    monkeypatch.setattr(
        "sqrt_etmax.distribution.optimize.minimize",
        captura_objetivo
    )
    
    sqrt_etmax.fit_custom(np.array([1.0, 4.0]))
    sqrt_etmax.fit_custom(np.array([0.0, 1.0, 4.0]))
    
    incremento = evaluaciones[1] - evaluaciones[0]
    assert incremento == pytest.approx(k)
    
def test_verosimilitud_positiva_coincide_con_densidad(monkeypatch):
    datos = np.array([0.5, 1.0, 4.0, 12.0])
    k = 2.0
    alpha = 0.7
    evaluaciones = []

    def capturar_objetivo(funcion, inicio, args, method):
        parametros = np.array([k, alpha])
        evaluaciones.append(float(funcion(parametros, *args)))
        return SimpleNamespace(x=parametros)

    monkeypatch.setattr(
        "sqrt_etmax.distribution.optimize.minimize",
        capturar_objetivo,
    )

    sqrt_etmax.fit_custom(datos)

    densidades = sqrt_etmax.pdf(datos, k, loc=0, scale=1 / alpha)
    esperado = -np.log(densidades).sum()

    assert evaluaciones[0] == pytest.approx(esperado)

def test_ajuste_mixto_coincide_con_referencia_perfilada():
    datos = np.array([0.0, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
    positivos = datos[datos > 0]
    n_ceros = np.count_nonzero(datos == 0)
    n_positivos = len(positivos)

    def k_optimo(alpha):
        u = np.sqrt(alpha * positivos)
        return n_positivos / (
            n_ceros + np.sum((1 + u) * np.exp(-u))
        )

    def objetivo_perfilado(log_alpha):
        alpha = np.exp(log_alpha)
        k = k_optimo(alpha)
        u = np.sqrt(alpha * positivos)

        return (
            n_positivos
            - n_positivos * np.log(k * alpha / 2)
            + np.sum(u)
        )

    referencia = minimize_scalar(
        objetivo_perfilado,
        bounds=(-10, 10),
        method="bounded",
        options={"xatol": 1e-12},
    )
    assert referencia.success

    alpha_ref = np.exp(referencia.x)
    k_ref = k_optimo(alpha_ref)

    estimados = sqrt_etmax.fit_custom(datos)

    np.testing.assert_allclose(
        estimados,
        [k_ref, alpha_ref],
        rtol=1e-4,
        atol=1e-6,
    )


