# Changelog

Todos los cambios notables de este proyecto se documentarán en este archivo.

El formato sigue [Keep a Changelog](https://keepachangelog.com/es/1.1.0/), y el proyecto se adhiere a [Versionado Semántico](https://semver.org/lang/es/).

## [Sin publicar]

### Añadido
- Suite completa de pruebas unitarias para CDF, PPF, PDF, `rvs` y ajuste por L-momentos y máxima verosimilitud (15 tests).
- Integración continua en GitHub Actions para Python 3.8-3.12 (`.github/workflows/ci.yml`).

## [0.2.0] - 2026-04-20

### Cambiado
- `fit_lmoments()` ahora usa la función cuantil analítica exacta (función W de Lambert), mejorando precisión y rendimiento (~500 veces más rápido que la búsqueda numérica).

### Añadido
- Implementación de la inversa analítica de la CDF (PPF) mediante la función W de Lambert.

## [0.1.0] - 2026-03-24

### Añadido
- Primera versión pública de la distribución SQRT-ETmax sobre la interfaz `scipy.stats.rv_continuous`.
- Ajuste por máxima verosimilitud (`fit_custom`) y por L-momentos (`fit_lmoments`).
- Advertencia informativa cuando se emplean L-momentos con series que contienen valores nulos.
- Documentación de uso y pruebas unitarias iniciales.
