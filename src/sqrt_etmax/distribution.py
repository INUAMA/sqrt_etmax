import numpy as np
from scipy import optimize, integrate
from scipy.special import lambertw
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
        
        Utiliza la función cuantil analítica exacta (_ppf) basada en la función W 
        de Lambert para calcular los L-momentos teóricos con precisión y estabilidad.

        Args:
            data (array_like): Datos a ajustar.
            
        Returns:
            list: Lista con los parámetros [k, alpha] estimados.
        """
        n = len(data)
        data_sorted = np.sort(data)
        
        # 1. Calcular L-Momentos Muestrales
        l1_sample = np.mean(data_sorted)
        
        # b1 usando el estimador de la Guía CEDEX/Hosking (i-0.35)/n
        i = np.arange(1, n + 1)
        weights = (i - 0.35) / n 
        b1 = np.sum(data_sorted * weights) / n
        
        l2_sample = 2 * b1 - l1_sample
        
        def get_theoretical_lmoments(params):
            k_est, alpha_est = params
            if k_est <= 0 or alpha_est <= 0:
                return 1e9, 1e9
            
            # El cuantil es 0 para p < exp(-k).
            # Integrar desde ese punto evita zonas planas y previene el IntegrationWarning.
            p_min = np.exp(-k_est)
            
            l1_theo, _ = integrate.quad(lambda p: self._ppf(p, k_est) / alpha_est, p_min, 1.0)
            l2_theo, _ = integrate.quad(lambda p: (self._ppf(p, k_est) / alpha_est ) * (2*p -1), p_min, 1.0)
            
            return l1_theo, l2_theo

        # 3. Optimizar para igualar momentos
        def objective(params):
            l1_theo, l2_theo = get_theoretical_lmoments(params)
            # Minimizamos la diferencia cuadrática relativa
            err1 = (l1_theo - l1_sample) / l1_sample
            err2 = (l2_theo - l2_sample) / l2_sample
            return err1**2 + err2**2

        # Estimación inicial
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
