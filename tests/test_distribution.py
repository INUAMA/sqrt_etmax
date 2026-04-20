import numpy as np
import pytest
from sqrt_etmax.distribution import sqrt_etmax

def test_cdf_at_zero():
    """La CDF tendiendo a 0 por la derecha debe ser exp(-k)."""
    k = 1.5
    # Recordatorio matemático: F(0) = exp(-k * (1 + 0) * exp(0)) = exp(-k)
    # Nota: Scipy fuerza cdf(a)=0 en rv_continuous si a=0.0. 
    # Por ello evaluamos el límite por la derecha.
    expected = np.exp(-k)
    assert np.isclose(sqrt_etmax.cdf(1e-10, k), expected)

def test_cdf_at_infinity():
    """La CDF debe tender a 1 cuando x tiende a infinito."""
    k = 1.5
    # Usamos un valor lo suficientemente alto para aproximar a infinito
    assert np.isclose(sqrt_etmax.cdf(1000, k), 1.0, atol=1e-4)

def test_pdf_non_negative():
    """La PDF nunca debe devolver probabilidades negativas."""
    k = 2.0
    x = np.linspace(0, 50, 200)
    pdf_vals = sqrt_etmax.pdf(x, k)
    assert np.all(pdf_vals >= 0)

def test_freeze_params():
    """La función freeze_params debe generar un objeto frozen con los parámetros esperados."""
    k = 2.0
    alpha = 0.5
    dist = sqrt_etmax.freeze_params(k, alpha)
    
    # Validar evaluando la CDF usando el objeto congelado en x->0+
    expected_cdf_0 = np.exp(-k)
    assert np.isclose(dist.cdf(1e-10), expected_cdf_0)

def test_fit_custom_returns_valid_params():
    """El método fit_custom debe devolver parámetros k y alpha mayores que 0."""
    np.random.seed(42)
    data = np.random.exponential(scale=1.5, size=100)
    k_est, alpha_est = sqrt_etmax.fit_custom(data)
    assert k_est > 0
    assert alpha_est > 0

def test_ppf_inverses_cdf():
    """La función cuantil (PPF) debe ser la inversa exacta de la CDF."""
    k = 1.5
    # La CDF en x=0 es exp(-1.5) ~ 0.223. Evaluamos probabilidades mayores.
    p = np.array([0.3, 0.5, 0.9, 0.95, 0.99])
    
    # Calculamos los cuantiles para estas probabilidades
    x = sqrt_etmax.ppf(p, k)
    # Verificamos que al aplicar la CDF a los cuantiles recuperamos 'p'
    p_calc = sqrt_etmax.cdf(x, k)
    
    np.testing.assert_allclose(p, p_calc, rtol=1e-5)

def test_ppf_below_minimum_probability():
    """Para probabilidades por debajo de exp(-k), la PPF debe devolver 0."""
    k = 1.5
    p_min = np.exp(-k)
    p = np.array([0.01, 0.1, p_min - 1e-5])
    
    x = sqrt_etmax.ppf(p, k)
    np.testing.assert_allclose(x, 0.0, atol=1e-7)

def test_fit_lmoments_returns_valid_params():
    """El método fit_lmoments debe devolver parámetros k y alpha mayores que 0."""
    np.random.seed(42)
    data = np.random.exponential(scale=1.5, size=100)
    k_est, alpha_est = sqrt_etmax.fit_lmoments(data)
    assert k_est > 0
    assert alpha_est > 0
