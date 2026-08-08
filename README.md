# sqrt_etmax

Implementación en Python de la distribución estadística **SQRT-ETmax**, utilizada ampliamente en la hidrología española para el cálculo de caudales y precipitaciones máximas según la **Norma 5.2-IC** del Ministerio de Fomento.

Este paquete está construido sobre la interfaz `scipy.stats.rv_continuous`, lo que significa que hereda toda la funcionalidad estándar de Scipy (cálculo de cuantiles, intervalos de confianza, generación de números aleatorios, etc.).

---

## Instalación

Una vez publicado en el índice PyPI, el paquete se instalará directamente con:

```bash
pip install sqrt_etmax
```

Mientras tanto, se puede instalar desde el código fuente. Clona este repositorio y ejecuta:

```bash
pip install .
```

Si vas a modificar el código, se recomienda instalarlo en modo desarrollo:

```bash
pip install -e .
```

## Uso Básico

La distribución expone una instancia global llamada `sqrt_etmax` que puede usarse directamente como cualquier otra distribución de `scipy.stats`.

### 1. Ajuste de datos y cálculo de periodos de retorno

```python
import numpy as np
from sqrt_etmax.distribution import sqrt_etmax

# 1. Datos de ejemplo (ej. serie de precipitaciones máximas anuales)
datos = np.array([45.2, 56.1, 38.9, 78.4, 62.0, 41.5, 92.3, 55.6])

# 2. Ajuste de parámetros (k, alpha)
# Puedes utilizar fit_custom() (Máxima Verosimilitud) o fit_lmoments() (L-Momentos).
# NOTA: fit_lmoments() ahora utiliza la función cuantil analítica exacta (W de Lambert) mejorando su precisión y rendimiento.
k_est, alpha_est = sqrt_etmax.fit_custom(datos)
print(f"Parámetros ajustados -> k: {k_est:.4f}, alpha: {alpha_est:.4f}")

# 3. Crear un objeto congelado ("frozen") con los parámetros ajustados
dist = sqrt_etmax.freeze_params(k_est, alpha_est)

# 4. Calcular el cuantil para un Periodo de Retorno (T)
T = 100
probabilidad_no_excedencia = 1 - (1 / T)
valor_T100 = dist.ppf(probabilidad_no_excedencia)

print(f"Precipitación para T={T} años: {valor_T100:.2f} mm")
```

## Parametrización y masa en el origen

La API de `sqrt_etmax` utiliza `alpha` como parámetro de escala inversa.
Cuando se crea una distribución congelada, la relación con la convención de
SciPy es:

```text
scale = 1 / alpha
```

La función de distribución acumulada para `x >= 0` es:

```text
F(x) = exp[-k * (1 + sqrt(alpha * x)) * exp(-sqrt(alpha * x))]
```

Por tanto, el límite por la derecha en el origen es:

```text
F(0+) = exp(-k)
```

Este valor representa la masa de probabilidad asociada al origen. El objeto
de SciPy tiene soporte `x >= 0` y puede devolver `cdf(0) = 0` por la
convención de `rv_continuous`; para comprobar el límite matemático debe
evaluarse un valor positivo muy próximo a cero:

```python
import numpy as np
from sqrt_etmax.distribution import sqrt_etmax

k = 2.0
alpha = 0.7
dist = sqrt_etmax.freeze_params(k, alpha)

print(f"scale = {1 / alpha:.6f}")
print(f"F(0+) esperado = {np.exp(-k):.6f}")
print(f"F(0+) calculado = {dist.cdf(1e-10):.6f}")

# rvs reproduce la masa en el origen aproximadamente como exp(-k).
sample = dist.rvs(size=10000, random_state=42)
print(f"Fracción de ceros = {np.mean(sample == 0):.3f}")
```

## Cómo citar

Si utilizas `sqrt_etmax` en tu investigación, cita el artículo correspondiente:

> Molina-Pérez, G., et al. (pendiente de publicación). *sqrt_etmax: distribución SQRT-ETmax para Python*. Journal of Open Source Software. DOI: [pendiente]

## Ejecución de Pruebas (Tests)

El proyecto incluye una suite de pruebas unitarias basadas en `pytest` para garantizar el comportamiento correcto de las funciones matemáticas (CDF, PDF, ajustes, etc.).

Para correr las pruebas, primero instala las dependencias de desarrollo:

```bash
pip install -e .[test]
pytest tests/
```

## Contribuciones y Reporte de Errores

Si deseas contribuir al código o has encontrado algún comportamiento matemático anómalo, por favor revisa el archivo CONTRIBUTING.md para más detalles sobre cómo abrir un *Pull Request* o un *Issue*.
