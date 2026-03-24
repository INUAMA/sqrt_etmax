# sqrt_etmax

Implementación en Python de la distribución estadística **SQRT-ETmax**, utilizada ampliamente en la hidrología española para el cálculo de caudales y precipitaciones máximas según la **Norma 5.2-IC** del Ministerio de Fomento.

Este paquete está construido sobre la interfaz `scipy.stats.rv_continuous`, lo que significa que hereda toda la funcionalidad estándar de Scipy (cálculo de cuantiles, intervalos de confianza, generación de números aleatorios, etc.).

---

## Instalación

Actualmente el paquete se puede instalar desde el código fuente. Clona este repositorio y ejecuta:

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
# Se recomienda usar fit_custom() o fit_lmoments() para mayor estabilidad matemática
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

## Ejecución de Pruebas (Tests)

El proyecto incluye una suite de pruebas unitarias basadas en `pytest` para garantizar el comportamiento correcto de las funciones matemáticas (CDF, PDF, ajustes, etc.).

Para correr las pruebas, primero instala las dependencias de desarrollo:

```bash
pip install -e .[test]
pytest tests/
```

## Contribuciones y Reporte de Errores

Si deseas contribuir al código o has encontrado algún comportamiento matemático anómalo, por favor revisa el archivo CONTRIBUTING.md para más detalles sobre cómo abrir un *Pull Request* o un *Issue*.
