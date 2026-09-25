import numpy as np
from scipy import optimize
from scipy.optimize import brentq
from scipy.special import lambertw
from scipy.stats import rv_continuous
from numpy.polynomial.legendre import leggauss

# Constantes de módulo: nodos y pesos de Gauss-Legendre (256 nodos)
_GL_NODES, _GL_WEIGHTS = leggauss(256)

# Límites del bracket para brentq
_K_MIN = 0.001
_K_MAX = 50000.0

def _validate_sample(data):
    """Valida y normaliza una muestra para los métodos de ajuste.

    Args:
        data (array_like): Observaciones de la muestra.

    Returns:
        numpy.ndarray: Muestra real validada, de tipo float.

    Raises:
        ValueError: Si la muestra incumple el contrato de entrada.
    """
    if np.ma.isMaskedArray(data) and np.any(np.ma.getmaskarray(data)):
        raise ValueError(
            "La muestra no puede contener observaciones enmascaradas."
        )

    # Conservamos el tipo para detectar complejos antes de convertir.
    try:
        data = np.asarray(data)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "La muestra debe contener valores numéricos reales."
        ) from exc

    if np.iscomplexobj(data) or (
        data.dtype.kind == "O"
        and any(np.iscomplexobj(valor) for valor in data.flat)
    ):
        raise ValueError(
            "La muestra no puede contener números complejos."
        )

    # Convertimos a float después de comprobar los tipos.
    try:
        data = np.asarray(data, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "La muestra debe contener valores numéricos reales."
        ) from exc

    if data.ndim != 1:
        raise ValueError("La muestra debe ser unidimensional.")

    if data.size < 2:
        raise ValueError(
            "La muestra debe contener al menos 2 observaciones."
        )

    if not np.all(np.isfinite(data)):
        raise ValueError(
            "La muestra debe contener solo valores finitos."
        )

    if np.any(data < 0):
        raise ValueError(
            "La muestra debe contener valores no negativos."
        )

    if np.all(data == data[0]):
        raise ValueError(
            "La muestra debe contener al menos dos valores distintos."
        )

    return data

def _validate_positive_scalar(value, name):
    """Valida y normaliza un parámetro escalar real y positivo."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(
            f"El parámetro {name} debe ser un escalar real "
            "de tipo entero o flotante, no booleano."
        )

    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"El parámetro {name} no se puede representar como float."
        ) from exc

    if not np.isfinite(value) or value <= 0:
        raise ValueError(
            f"El parámetro {name} debe ser finito y estrictamente positivo."
        )

    return value

class sqrt_etmax_gen(rv_continuous):
    """
    Implementación de la distribución SQRT-ETmax para hidrología española.
    Referencia: Etoh et al. (1987), Salas (2004), Norma 5.2-IC.
    
    Args:
        k (float): Parámetro de forma (shape).
        alpha (float, opcional): Parámetro de escala inverso. Para scipy, scale = 1/alpha.
    """
    def _open_support_mask(self, x, *args):
        """Incluye el extremo inferior donde está la masa puntual."""
        a, b = self._get_support(*args)

        with np.errstate(invalid="ignore"):
            return (a <= x) & (x < b)

    def _logcdf(self, x, k):
        """Calcula el logaritmo de la CDF sin exponenciar primero."""
        raiz = np.sqrt(x)
        return -k * (1 + raiz) * np.exp(-raiz)

    def _sf(self, x, k):
        """Calcula la supervivencia evitando restar números próximos."""
        return -np.expm1(self._logcdf(x, k))

    def _logsf(self, x, k):
        """Calcula el logaritmo de la supervivencia."""
        with np.errstate(divide="ignore"):
            return np.log(self._sf(x, k))
    
    def _cdf(self, x, k):
        """
        Función de distribución acumulada (CDF).
        F(x) = exp[ -k * (1 + sqrt(x)) * exp(-sqrt(x)) ]

        Args:
            x (array_like): Cuantiles normalizados.
            k (float): Parámetro de forma.
            
        Returns:
            ndarray: Valores de la probabilidad acumulada.
        """
        # Protección contra raíces de números negativos
        x_safe = np.where(x < 0, 0, x)
        sqrt_x = np.sqrt(x_safe)
        
        return np.exp(-k * (1 + sqrt_x) * np.exp(-sqrt_x))

    def _pdf(self, x, k):
        """
        Función de densidad de probabilidad (PDF).
        Derivada analítica de la CDF para el ajuste por máxima verosimilitud (MLE).
        f(x) = F(x) * (k / 2) * exp(-sqrt(x))

        Args:
            x (array_like): Cuantiles normalizados.
            k (float): Parámetro de forma.
            
        Returns:
            ndarray: Valores de densidad de probabilidad.
        """
        x_safe = np.where(x <= 0, 0, x)
        # Calculamos la CDF internamente para reutilizar
        cdf_val = self._cdf(x_safe, k)
        
        # Evitamos división por cero o valores inválidos
        with np.errstate(divide='ignore', invalid='ignore'):
            pdf_val = cdf_val * (k / 2.0) * np.exp(-np.sqrt(x_safe))
        
        return np.where(x <= 0, 0, pdf_val)

    def _ppf(self, p, k):
        """ Inversa exacta con Lambert W.
            Elimina la necesidad de integración numérica
        """
        # La probabilidad en x=0 es exp(-k). Valores menores no tienen inversa real positiva.
        p_min = np.exp(-k)
        p_safe = np.maximum(p, p_min)
        
        # y = -ln(p) / k
        y = -np.log(p_safe) / k
        # resuelve (1+u)*exp(-u) = y
        arg = -y / np.exp(1.0)
        w = np.real(lambertw(arg, k=-1))
        u = -1.0 - w
        
        # Si p < p_min, el cuantil es 0.0
        return np.where(p < p_min, 0.0, u**2)

    def _logpdf(self, x, k):
        """
        Logaritmo de la PDF para mejorar estabilidad numérica.
        ln(f(x)) = ln(k/2) - sqrt(x) - k(1+sqrt(x))exp(-sqrt(x))

        Args:
            x (array_like): Cuantiles normalizados.
            k (float): Parámetro de forma.
            
        Returns:
            ndarray: Logaritmo natural de la densidad de probabilidad.
        """
        x_safe = np.where(x < 0, 0, x)
        sqrt_x = np.sqrt(x_safe)
        
        # Término proveniente de ln(F(x))
        ln_cdf = -k * (1 + sqrt_x) * np.exp(-sqrt_x)
        # Término restante de la derivada
        ln_rest = np.log(k / 2.0) - sqrt_x
        
        return np.where(x <= 0, -np.inf, ln_cdf + ln_rest)

    def fit_custom(self, data):
        """
        Método de ajuste robusto usando optimización directa de Log-Likelihood.
        El método.fit() genérico de Scipy puede fallar con esta función no estándar.
        
        Args:
            data (array_like): Muestra unidimensional de al menos dos
                observaciones reales, finitas y no negativas, con al
                menos dos valores distintos. Puede incluir ceros.
            
        Returns:
            numpy.ndarray: Parámetros [k, alpha] estimados.
        Notas:
            Los ceros exactos aportan -k a la log-verosimilitud,
            correspondiente a P(X=0) = exp(-k). Los valores positivos
            aportan su log-densidad.

            Este ajuste no interpreta los ceros como datos ausentes,
            censurados o redondeados. No realiza reintentos automáticos.
            Un resultado aceptado no garantiza un máximo global
            de la verosimilitud..

        Raises:
            RuntimeError: Si el optimizador no converge, devuelve
                parámetros no finitos o no positivos, o el valor
                final del objetivo no es finito.
            ValueError: Si la muestra incumple el contrato de entrada.
        """
        data = _validate_sample(data)

        def neg_log_likelihood(params, data):
            k, alpha = params
            if k <= 0 or alpha <= 0:
                return np.inf # Penalización infinita para restringir el dominio (k>0, alpha>0)
            
            # Log-Likelihood manual para mayor precisión numérica
            # L = sum( ln(f(xi)) )
            sqrt_ax = np.sqrt(alpha * data) # alpha actúa como inverso de scale
            
            # Término 1 proveniente de ln(F(x))
            # Equivalente a la lógica en _cdf pero vectorizado
            term1 = -k * (1 + sqrt_ax) * np.exp(-sqrt_ax)
            # Término 2 proveniente de ln(derivada)
            # Equivalente a la parte derivativa de _logpdf
            term2 = np.log(k * alpha / 2.0) - sqrt_ax
            
            # Los ceros aportan Log(P(X=0)) = -k
            # Los valores positivos conservan su Log-densidad
            log_aportaciones = np.where(
                data == 0,
                -k,
                term1 + term2
            )
            return -np.sum(log_aportaciones)

        # Estimación inicial de parámetros (Semillas)
        # Basado en aproximaciones de momentos (Salas, 2004)
        mean = np.mean(data)
        # Semilla heurística: alpha ~ 1/mean, k ~ 1
        initial_guess = [1.0, 1.0/mean]

        # Optimización Nelder-Mead (robusta para funciones no diferenciables)
        result = optimize.minimize(neg_log_likelihood, initial_guess, args=(data,), method='Nelder-Mead')

        if not result.success:
            raise RuntimeError(
                f"El ajuste MLE no convergió: {result.message}"
            )

        if (
            not np.all(np.isfinite(result.x))
            or np.any(result.x <= 0)
        ):
            raise RuntimeError(
                "El ajuste MLE devolvió parámetros no válidos."
            )

        if not np.isfinite(result.fun):
            raise RuntimeError(
                "El ajuste MLE devolvió un objetivo no finito."
            )

        return result.x # Retorna [k, alpha]

    def _lmoments_theoretical(self, k):
        """L-momentos teóricos estandarizados (α=1) por cuadratura Gauss-Legendre.

        Calcula los dos primeros L-momentos teóricos de la distribución
        SQRT-ETmax con α=1 usando cuadratura de Gauss-Legendre con 256 nodos
        sobre la PPF analítica, integrando desde p_min = exp(−k) hasta 1.

        La integral se realiza mediante cambio de variable de [−1, 1] a
        [p_min, 1]:
            p = p_min + (1 − p_min)(t + 1)/2
            dp = (1 − p_min)/2 · dt

        Args:
            k (float): Parámetro de forma.

        Returns:
            tuple: (a1, a2) donde a1 = ∫ Q(p,k) dp y a2 = ∫ Q(p,k)(2p−1) dp.
        """
        p_min = np.exp(-k)
        p = p_min + (1.0 - p_min) * (_GL_NODES + 1.0) / 2.0
        q = self._ppf(p, k)
        w_scale = (1.0 - p_min) / 2.0
        a1 = float(np.sum(q * _GL_WEIGHTS) * w_scale)
        a2 = float(np.sum(q * (2.0 * p - 1.0) * _GL_WEIGHTS) * w_scale)
        return a1, a2

    def fit_lmoments(self, data):
        """Ajuste por L-momentos (Hosking, 1990; Hosking & Wallis, 1997).

        Estima los parámetros (k, α) de la distribución SQRT-ETmax igualando
        los dos primeros L-momentos muestrales y teóricos. El método aprovecha
        la separabilidad de la PPF en forma y escala para reducir el problema
        a una ecuación 1D:

        1. Calcula los PWM muestrales insesgados de Hosking (1990) con la
           muestra ordenada x_(j), j = 0..n−1:
               b1 = Σ x_(j)·j / (n(n−1))
               l1 = media muestral
               l2 = 2b1 − l1
        2. τ₂(k) = λ₂/λ₁ = a₂(k)/a₁(k) es función exclusiva de k (la PPF
           es separable en forma y escala: x(p) = Q(p,k)/α).
        3. Resuelve τ₂(k) = τ₂_muestral por el método de Brent (brentq)
           con tolerancia xtol=1e-12.
        4. Calcula α de forma analítica: α = a₁(k̂)/l1.

        Los L-momentos teóricos se obtienen por cuadratura de Gauss-Legendre
        con 256 nodos sobre la PPF analítica (función W de Lambert), integrando
        desde p_min = exp(−k) hasta 1.

        Args:
            data (array_like): Muestra unidimensional de al menos dos
                observaciones reales, finitas y no negativas, con al
                menos dos valores distintos. Puede incluir ceros.

        Returns:
            list: Lista con los parámetros [k, alpha] estimados, donde k > 0
                es el parámetro de forma y α > 0 es el parámetro de escala inverso.

        Raises:
            ValueError: Si la muestra incumple el contrato de entrada
                o sus L-momentos muestrales L1 o L2 no son positivos.
            RuntimeError: Si τ₂ muestral cae fuera del rango alcanzable por
                la familia SQRT-ETmax, o si brentq no converge.
        """
        data = _validate_sample(data)
        n = len(data)

        data_sorted = np.sort(data)
        i = np.arange(n, dtype=np.float64)

        # PWM muestrales insesgados de Hosking (1990)
        b1 = np.sum(data_sorted * i) / (n * (n - 1.0))
        l1 = np.mean(data_sorted)
        l2 = 2.0 * b1 - l1

        if l1 <= 0:
            raise ValueError(
                f"L1 (media) debe ser positiva, se obtuvo {l1}"
            )
        if l2 <= 0:
            raise ValueError(
                f"L2 (escala) debe ser positiva, se obtuvo {l2}. "
                "Posible serie constante."
            )

        tau2_sample = l2 / l1

        # Rango alcanzable de τ₂(k): τ₂ es decreciente en k
        a1_kmin, a2_kmin = self._lmoments_theoretical(_K_MIN)
        a1_kmax, a2_kmax = self._lmoments_theoretical(_K_MAX)
        tau2_min = a2_kmax / a1_kmax  # τ₂(K_MAX) — mínimo
        tau2_max = a2_kmin / a1_kmin  # τ₂(K_MIN) — máximo

        if tau2_sample <= tau2_min or tau2_sample >= tau2_max:
            raise RuntimeError(
                f"τ₂ muestral ({tau2_sample:.6f}) fuera del rango alcanzable "
                f"por SQRT-ETmax: ({tau2_min:.6f}, {tau2_max:.6f}). "
                "La familia no puede representar estos datos."
            )

        # Resolver τ₂(k) = τ₂_muestral por brentq
        def tau2_diff(k):
            a1_k, a2_k = self._lmoments_theoretical(k)
            return a2_k / a1_k - tau2_sample

        try:
            k_hat = brentq(tau2_diff, _K_MIN, _K_MAX, xtol=1e-12)
        except Exception as e:
            raise RuntimeError(
                f"brentq no convergió para τ₂ = {tau2_sample:.6f}: {e}"
            ) from e

        # α analítico: α = a₁(k̂) / l1
        a1_hat, _ = self._lmoments_theoretical(k_hat)
        alpha_hat = a1_hat / l1

        return [k_hat, alpha_hat]

    def freeze_params(self, k, alpha):
        """Crea una distribución congelada con parámetros validados.

        Admite enteros y flotantes escalares de Python y NumPy,
        que se normalizan a float.

        Args:
            k (float): Parámetro de forma, finito y estrictamente positivo.
            alpha (float): Escala inversa, finita y estrictamente positiva.

        Returns:
            scipy.stats.distributions.rv_frozen: Distribución con loc=0
                y scale=1/alpha.

        Raises:
            ValueError: Si un parámetro tiene un tipo no admitido, no puede
                representarse como float finito y positivo, o su escala
                derivada no es finita y estrictamente positiva.

        Examples:
            >>> dist = sqrt_etmax.freeze_params(2.0, 0.7)
            >>> cuantil_99 = dist.ppf(0.99)
        """
        k = _validate_positive_scalar(k, "k")
        alpha = _validate_positive_scalar(alpha, "alpha")

        scale = 1.0 / alpha

        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(
                "El parámetro alpha no permite representar una escala "
                "finita y estrictamente positiva."
            )

        return self(k, loc=0, scale=scale)

# Instancia global de la distribución
sqrt_etmax = sqrt_etmax_gen(name='sqrt_etmax', a=0.0)
