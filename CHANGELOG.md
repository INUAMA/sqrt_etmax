# Changelog

Todos los cambios notables de este proyecto se documentarán en este archivo.

El formato sigue [Keep a Changelog](https://keepachangelog.com/es/1.1.0/), y el proyecto se adhiere a [Versionado Semántico](https://semver.org/lang/es/).

## [Unreleased]

### Corregido
- `fit_custom()` utiliza la masa de probabilidad `P(X=0)=exp(-k)`
  para los ceros exactos: cada cero aporta `-k` a la
  log-verosimilitud. Se conserva la contribución de los valores
  positivos. Resuelve #11.
- La API pública incluye la masa puntual en el extremo inferior:
  CDF, supervivencia y sus logaritmos son coherentes, también para
  distribuciones congeladas y transformaciones loc/scale. Se mejora
  la estabilidad numérica en el átomo. Resuelve #13.
- `fit_custom()` lanza `RuntimeError` ante fallos del optimizador,
  parámetros no finitos o no positivos, u objetivos no finitos.
  Los fallos de convergencia incluyen el diagnóstico del optimizador.
  Se conserva el retorno de parámetros para resultados válidos.
  Resuelve #15.
- `fit_custom()` y `fit_lmoments()` comparten la normalización y
  validación de muestras. Se admiten listas y arrays válidos y se
  rechazan entradas incompatibles mediante `ValueError` antes del
  ajuste numérico. Se evita perder la parte imaginaria durante la
  conversión, también en arrays de tipo object. Resuelve #17.
- `freeze_params()` valida los parámetros escalares `k` y `alpha`
  antes de calcular la escala y construir la distribución. Rechaza
  tipos incompatibles, valores no finitos o no positivos y escalas
  derivadas no representables mediante `ValueError`. Normaliza los
  escalares admitidos de Python y NumPy a float. Resuelve #19.

### Añadido
- Pruebas de regresión para la contribución de ceros con distintos
  parámetros, la coherencia con la densidad positiva y un ajuste
  mixto contrastado con una referencia de verosimilitud perfilada.
- Pruebas de equivalencia entre formatos de entrada, rechazo de
  muestras inválidas y conservación de los valores, el orden y
  la máscara de los datos originales.
- Pruebas de validación de parámetros, límites de conversión a float
  y conservación de la CDF, el átomo y los cuantiles al utilizar
  parámetros válidos en `freeze_params()`.

## [0.3.0] - 2026-09-03

### Cambiado
- Refactorización completada de `fit_lmoments()`: sustitución de la optimización Nelder-Mead 2D por una solución exacta 1D mediante `brentq` sobre el L-ratio τ₂ (función exclusiva de `k`), cuadratura de Gauss-Legendre de 256 nodos para los L-momentos teóricos, PWM muestrales insesgados de Hosking (1990) y obtención analítica de `alpha`. Mejora la exactitud (igualación exacta de momentos), garantiza la convergencia y reduce el coste computacional.
- `.opencode/` excluido del control de versiones (configuración interna de agentes, no versionada).

### Añadido
- Documentación de la parametrización `alpha`/`scale` y la masa en el origen en el README.
- Archivo `AGENTS.md` para agentes de IA con contexto del proyecto.
- Directorio `planning/` con documentación interna de desarrollo.
- 3 nuevos tests del estimador L-momentos (`test_fit_lmoments_exact_moment_matching`, `test_fit_lmoments_raises_on_constant_data`, `test_fit_lmoments_raises_on_too_few_data`).

### Eliminado
- `.opencode/agents/` del tracking de git (archivos conservados en disco, excluidos del historial con `git filter-branch`).

## [0.2.0] - 2026-08-06

### Añadido
- Suite completa de pruebas unitarias para CDF, PPF, PDF, `rvs` y ajuste por L-momentos y máxima verosimilitud (15 tests).
- Integración continua en GitHub Actions para Python 3.8-3.12 (`.github/workflows/ci.yml`).
- Workflow de publicación automática en PyPI (`.github/workflows/publish.yml`).
- Autores y licencia en `pyproject.toml`.

### Cambiado
- `fit_lmoments()` ahora usa la función cuantil analítica exacta (función W de Lambert), mejorando precisión y rendimiento (~500 veces más rápido que la búsqueda numérica).
- Implementación de la inversa analítica de la CDF (PPF) mediante la función W de Lambert.

## [0.1.0] - 2026-03-24

### Añadido
- Primera versión pública de la distribución SQRT-ETmax sobre la interfaz `scipy.stats.rv_continuous`.
- Ajuste por máxima verosimilitud (`fit_custom`) y por L-momentos (`fit_lmoments`).
- Advertencia informativa cuando se emplean L-momentos con series que contienen valores nulos.
- Documentación de uso y pruebas unitarias iniciales.
