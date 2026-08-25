import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from sqrt_etmax.distribution import sqrt_etmax

# Nodos y pesos de Gauss-Legendre (256 nodos), idénticos a los de la implementación
_GL_NODES, _GL_WEIGHTS = leggauss(256)

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

def test_ppf_cdf_consistency_wide():
    """La PPF debe invertir la CDF en un barrido denso de probabilidades."""
    k = 1.5
    p_min = np.exp(-k)
    p = np.linspace(p_min + 1e-6, 1 - 1e-9, 500)
    x = sqrt_etmax.ppf(p, k)
    np.testing.assert_allclose(sqrt_etmax.cdf(x, k), p, rtol=1e-6, atol=1e-9)

def test_pdf_is_derivative_of_cdf():
    """La PDF analítica debe coincidir con la derivada numérica de la CDF."""
    k = 2.0
    dist = sqrt_etmax.freeze_params(k, 0.7)
    h = 1e-6
    for x in [0.1, 1.0, 5.0, 20.0]:
        num = (dist.cdf(x + h) - dist.cdf(x - h)) / (2 * h)
        assert np.isclose(dist.pdf(x), num, rtol=1e-4)

def test_ppf_matches_numerical_inversion():
    """La PPF analítica (Lambert W) debe coincidir con la inversión numérica."""
    from scipy.optimize import brentq
    k = 1.5
    dist = sqrt_etmax.freeze_params(k, 0.7)
    for p in [0.3, 0.5, 0.9, 0.99, 0.999]:
        x = dist.ppf(p)
        f = lambda x: dist.cdf(x) - p
        x_num = brentq(f, 0.0, 1e6)
        assert np.isclose(x, x_num, rtol=1e-8)

def test_rvs_atom_probability():
    """rvs debe reproducir la masa de probabilidad en el origen exp(-k)."""
    k = 2.0
    alpha = 0.7
    n = 200000
    sample = sqrt_etmax.rvs(k, scale=1.0 / alpha, size=n, random_state=7)
    frac = np.mean(sample == 0)
    assert abs(frac - np.exp(-k)) < 2e-3

def test_fit_lmoments_recovers_parameters():
    """fit_lmoments debe recuperar los parámetros de una muestra sintética conocida."""
    k_true, alpha_true = 2.0, 0.7
    n = 8000
    data = sqrt_etmax.rvs(k_true, scale=1.0 / alpha_true, size=n, random_state=42)
    k_est, alpha_est = sqrt_etmax.fit_lmoments(data)
    assert abs(k_est - k_true) / k_true < 0.05
    assert abs(alpha_est - alpha_true) / alpha_true < 0.05

def test_fit_lmoments_recovers_design_quantiles():
    """fit_lmoments debe recuperar los cuantiles de diseño (T=10, 100, 1000)."""
    k_true, alpha_true = 2.0, 0.7
    n = 8000
    data = sqrt_etmax.rvs(k_true, scale=1.0 / alpha_true, size=n, random_state=42)
    k_est, alpha_est = sqrt_etmax.fit_lmoments(data)
    for T in [10, 100, 1000]:
        p = 1.0 - 1.0 / T
        q_true = sqrt_etmax.ppf(p, k_true) / alpha_true
        q_est = sqrt_etmax.ppf(p, k_est) / alpha_est
        assert abs(q_est - q_true) / q_true < 0.05

def test_fit_custom_recovers_design_quantiles():
    """fit_custom (MLE) debe aproximar los cuantiles de diseño dentro de una tolerancia amplia."""
    k_true, alpha_true = 2.0, 0.7
    n = 8000
    data = sqrt_etmax.rvs(k_true, scale=1.0 / alpha_true, size=n, random_state=42)
    k_est, alpha_est = sqrt_etmax.fit_custom(data)
    for T in [10, 100, 1000]:
        p = 1.0 - 1.0 / T
        q_true = sqrt_etmax.ppf(p, k_true) / alpha_true
        q_est = sqrt_etmax.ppf(p, k_est) / alpha_est
        assert abs(q_est - q_true) / q_true < 0.30


# ---------------------------------------------------------------------------
# Tests nuevos: reimplementación fit_lmoments (Paso 3)
# ---------------------------------------------------------------------------

def test_fit_lmoments_exact_moment_matching():
    """Los L-momentos teóricos de (k̂, α̂) ajustados deben igualar los muestrales.

    Verifica la propiedad clave del estimador exacto (brentq + GL-256):
    con los parámetros ajustados, los L-momentos teóricos coinciden con los
    PWM muestrales insesgados de Hosking dentro de tolerancia numérica.

    Los L-momentos teóricos se calculan por cuadratura de Gauss-Legendre
    (256 nodos) sobre la PPF, el mismo método que usa fit_lmoments internamente,
    para verificar el matching exacto hasta precisión de máquina.
    """
    k_true, alpha_true = 2.0, 0.7
    data = sqrt_etmax.rvs(k_true, scale=1.0 / alpha_true, size=8000, random_state=42)
    k_hat, alpha_hat = sqrt_etmax.fit_lmoments(data)

    # --- PWM muestrales insesgados de Hosking (índice 0-based) ---
    n = len(data)
    x_sorted = np.sort(data)
    idx = np.arange(n, dtype=np.float64)
    b1 = np.sum(x_sorted * idx) / (n * (n - 1.0))
    l1 = np.mean(x_sorted)
    l2 = 2.0 * b1 - l1

    # --- L-momentos teóricos por Gauss-Legendre 256 sobre la PPF ---
    # Mismo método que la implementación interna de fit_lmoments
    p_min = np.exp(-k_hat)
    p_gl = p_min + (1.0 - p_min) * (_GL_NODES + 1.0) / 2.0
    w_scale = (1.0 - p_min) / 2.0
    q_vals = sqrt_etmax.ppf(p_gl, k_hat, scale=1.0 / alpha_hat)

    l1_theo = float(np.sum(q_vals * _GL_WEIGHTS) * w_scale)
    l2_theo = float(np.sum(q_vals * (2.0 * p_gl - 1.0) * _GL_WEIGHTS) * w_scale)

    np.testing.assert_allclose(l1_theo, l1, rtol=1e-8, atol=1e-8,
                               err_msg="L1 teórico no coincide con muestral")
    np.testing.assert_allclose(l2_theo, l2, rtol=1e-8, atol=1e-8,
                               err_msg="L2 teórico no coincide con muestral")


def test_fit_lmoments_raises_on_constant_data():
    """Una serie constante debe provocar ValueError (l₂ = 0, varianza nula)."""
    data = np.ones(50)
    with pytest.raises(ValueError):
        sqrt_etmax.fit_lmoments(data)


def test_fit_lmoments_raises_on_too_few_data():
    """Una muestra de un solo valor debe provocar ValueError (n < 2)."""
    data = np.array([5.0])
    with pytest.raises(ValueError):
        sqrt_etmax.fit_lmoments(data)
