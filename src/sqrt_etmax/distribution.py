import numpy as np
from scipy import optimize, integrate
from scipy.stats import rv_continuous

class sqrt_etmax_gen(rv_continuous):
    """
    Implementación de la distribución SQRT-ETmax para hidrología española.
    Referencia: Etoh et al. (1987), Salas (2004), Norma 5.2-IC.
    
    Args:
        k (float): Parámetro de forma (shape).
        alpha (float, opcional): Parámetro de escala inverso. Para scipy, scale = 1/alpha.
    """
    
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
            data (array_like): Datos a ajustar.
            
        Returns:
            list: Lista con los parámetros [k, alpha] estimados.
        """

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
            
            return -np.sum(term1 + term2)

        # Estimación inicial de parámetros (Semillas)
        # Basado en aproximaciones de momentos (Salas, 2004)
        mean = np.mean(data)
        # Semilla heurística: alpha ~ 1/mean, k ~ 1
        initial_guess = [1.0, 1.0/mean]
        
        # Optimización Nelder-Mead (robusta para funciones no diferenciables)
        result = optimize.minimize(neg_log_likelihood, initial_guess, args=(data,), method='Nelder-Mead')
        return result.x # Retorna [k, alpha]

    def fit_lmoments(self, data):
        """
        Ajuste mediante L-Momentos (igualando momentos muestrales y teóricos).
        
        ADVERTENCIA: La distribución SQRT-ETmax no tiene una función cuantil (inversa de la CDF) 
        analítica. Por tanto, este método utiliza integración numérica iterativa para aproximar 
        los L-momentos teóricos. Esto puede ser computacionalmente costoso y numéricamente inestable.
        Se recomienda priorizar el uso de `fit_custom` (Máxima Verosimilitud).

        Args:
            data (array_like): Datos a ajustar.
            
        Returns:
            list: Lista con los parámetros [k, alpha] estimados.
        """
        # 1. Calcular L-Momentos Muestrales (b0 y b1 son PWMs)

        n = len(data)
        data_sorted = np.sort(data)
        
        # b0 es simplemente la media
        b0 = np.mean(data_sorted)
        
        # b1 usando el estimador de la Guía CEDEX/Hosking (i-0.35)/n
        i = np.arange(1, n + 1)
        weights = (i - 0.35) / n 
        b1 = np.sum(data_sorted * weights) / n
        
        # L-Momentos muestrales
        l1_sample = b0
        l2_sample = 2 * b1 - b0
        #t2_sample = l2_sample / l1_sample # L-CV muestral

        # 2. Definir función para calcular L-Momentos Teóricos (integración numérica)
        # Como no tiene inversa explícita, integramos x*pdf(x) para la media (l1)
        # y usamos aproximaciones numéricas para l2.
        
        def theoretical_statistics(params):
            k_est, alpha_est = params
            if k_est <= 0 or alpha_est <= 0:
                return 1e6, 1e6 # Penalización
            
            # Definimos la PDF escalada localmente para integrar
            scale = 1.0 / alpha_est
            
            # Media teórica (L1) = Integral(x * pdf(x))
            # Para SQRT-ETmax, la media teórica se puede aproximar o integrar.
            # Integramos numéricamente hasta un límite alto razonable
            upper_limit = scale * 100 # Límite práctico
            
            def integrand_l1(x):
                return x * self.pdf(x, k_est, loc=0, scale=scale)
            
            l1_theo, _ = integrate.quad(integrand_l1, 0, upper_limit)
            
            # L2 teórica = Integral( (2*F(x) - 1) * x * pdf(x) )
            def integrand_l2(x):
                F_x = self.cdf(x, k_est, loc=0, scale=scale)
                return x * (2 * F_x - 1) * self.pdf(x, k_est, loc=0, scale=scale)
            
            l2_theo, _ = integrate.quad(integrand_l2, 0, upper_limit)
            
            return l1_theo, l2_theo

        # 3. Optimizar para igualar momentos
        def objective(params):
            l1_theo, l2_theo = theoretical_statistics(params)
            #t2_theo = l1_theo / l1_theo
            # Minimizamos la diferencia cuadrática relativa
            err1 = (l1_theo - l1_sample) / l1_sample
            err2 = (l2_theo - l2_sample) / l2_sample
            #err2 = (t2_theo - t2_sample) / t2_sample
            return err1**2 + err2**2

        # Estimación inicial (puedes usar la media para alpha)
        initial_guess = [1.0, 1.0/l1_sample] 
        
        res = optimize.minimize(objective, initial_guess, method='Nelder-Mead', tol=1e-4)
        return res.x # Retorna [k, alpha]

    def freeze_params(self, k, alpha):
        """
        Convierte los parámetros k y alpha (de fit_custom) a un objeto Scipy listo para usar.
        
        Args:
            k (float): Parámetro de forma.
            alpha (float): Parámetro de escala inverso.
            
        Returns:
            scipy.stats.distributions.rv_frozen: Distribución con parámetros congelados.
            
        Ejemplo:
            >>> dist = sqrt_etmax.freeze_params(k, alpha)
            >>> caudal_T100 = dist.ppf(0.99)
        """
        return self(k, loc=0, scale=1.0/alpha)

# Instancia global de la distribución
sqrt_etmax = sqrt_etmax_gen(name='sqrt_etmax', a=0.0)
