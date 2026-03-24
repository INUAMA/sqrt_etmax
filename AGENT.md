# Instrucciones para Agentes de IA (AI Agent Context)

Este archivo sirve como contexto y sistema de reglas para cualquier agente de inteligencia artificial (Copilot, Cursor, Gemini, etc.) que asista en la lectura o escritura de código en este repositorio.

## Contexto del Proyecto
- **Nombre**: `sqrt_etmax`
- **Propósito**: Implementación de la distribución estadística SQRT-ETmax utilizada en la hidrología española (Norma 5.2-IC, Ministerio de Fomento).
- **Framework**: Python, estructurado como un paquete.
- **Dependencias Base**: `scipy` (específicamente la interfaz `scipy.stats`), `numpy`.

## Reglas de Generación de Código
- La distribución debe heredar siempre de `scipy.stats.rv_continuous` y sobreescribir los métodos privados `_pdf` o `_cdf`.
- Fomentar el uso de operaciones vectorizadas con `numpy` para mejorar el rendimiento.
- La documentación técnica y docstrings deben escribirse preferiblemente en español, usando terminología hidrológica correcta y el formato de docstrings de Google.
- **Pruebas (Tests)**: Todo el código nuevo o las correcciones matemáticas deben estar respaldadas por pruebas unitarias usando el framework `pytest` ubicadas en el directorio `tests/`.