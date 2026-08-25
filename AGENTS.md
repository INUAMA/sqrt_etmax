# AGENTS.md - sqrt_etmax

## Contexto del Proyecto

- **Nombre**: `sqrt_etmax`
- **Propósito**: Implementación de la distribución estadística SQRT-ETmax utilizada en la hidrología española (Norma 5.2-IC, Ministerio de Fomento).
- **Framework**: Python, estructurado como un paquete con layout `src/`.
- **Licencia**: MIT
- **PyPI**: `pip install sqrt_etmax`
- **Repositorio**: https://github.com/INUAMA/sqrt_etmax

## Dependencias

- `scipy>=1.7.0` (específicamente `scipy.stats` y `scipy.special.lambertw`)
- `numpy>=1.20.0`

## Estructura del Repositorio

```
sqrt_etmax/
├── src/sqrt_etmax/
│   ├── __init__.py          # Re-exporta la instancia global sqrt_etmax
│   └── distribution.py      # Clase sqrt_etmax_gen (rv_continuous)
├── tests/
│   └── test_distribution.py # 15 pruebas unitarias
├── .github/workflows/
│   ├── ci.yml               # CI para Python 3.8-3.12
│   └── publish.yml          # Publicación a PyPI vía Trusted Publishing
├── pyproject.toml
├── README.md
├── LICENSE
├── CONTRIBUTING.md
└── CHANGELOG.md
```

## Reglas de Código

- La distribución hereda de `scipy.stats.rv_continuous` y sobreescribe `_cdf`, `_pdf`, `_ppf`, `_logpdf`.
- Usar operaciones vectorizadas con `numpy` para rendimiento.
- Docstrings en español con formato Google, usando terminología hidrológica.
- Todo código nuevo o correcciones matemáticas debe incluir pruebas unitarias en `tests/`.
- Para cambios en la distribución, verificar consistencia CDF-PPF y recuperación de cuantiles.

## Instalación y Pruebas

```bash
# Instalación en modo desarrollo
pip install -e .[test]

# Ejecutar pruebas
pytest tests/ -v

# Verificar construcción
python -m build
```

## Git Workflow

- Antes de empezar trabajo nuevo, crear (o localizar) un issue en GitHub que
  describa la tarea; issues y descripciones de PR se escriben en inglés, con
  el label apropiado (`enhancement`, `documentation`, `bug`, ...).
- Toda PR debe vincularse a su issue con `Closes #N` en la primera línea de
  su descripción.
- Crear una rama de trabajo antes de editar: `git switch -c <tipo>/<descripción>`
- No commitear directamente a `main`
- Hacer push de la rama de trabajo: `git push -u origin <branch>`
- Fusionar mediante Pull Request en GitHub

## Notas de Implementación

- **Soporte**: `[0, ∞)` con masa de probabilidad en el origen: `F(0+) = exp(-k)`.
- **PPF analítica**: Usa la rama -1 de la función W de Lambert (`scipy.special.lambertw`).
- **Ajuste por L-momentos**: brentq sobre τ₂(k) + Gauss-Legendre 256 nodos; PWM insesgado de Hosking (1990).
- **Ajuste por MLE**: Nelder-Mead sobre log-likelihood manual con penalización para parámetros no válidos.

## Orquestación de subagentes (ejecución de planes)

Al ejecutar planes de desarrollo, el agente principal (build) actúa como
orquestador usando los subagentes de `.opencode/agents/`:

| Subagente | Modelo | Rol | Archivos |
|---|---|---|---|
| `implementador` | opencode/mimo-v2.5-free | Núcleo numérico | `src/sqrt_etmax/` |
| `tester` | opencode/mimo-v2.5-free | Tests y pytest | `tests/` |
| `documentador` | opencode/nemotron-3-ultra-free | Documentación | `AGENTS.md`, `CHANGELOG.md`, `planning/` |

Reglas de orquestación:
- Lanzar los subagentes en paralelo (una sola respuesta, varias llamadas
  `task`), cada uno con un prompt completo y autocontenido (los subagentes
  no ven el contexto del orquestador).
- Los ámbitos de archivos son disjuntos: no hay conflictos de edición.
- Dependencia: `tester` escribe los tests en paralelo, pero la ejecución
  final de `pytest` se realiza cuando `implementador` haya terminado.
- El orquestador verifica el trabajo: revisa `git diff` de cada subagente,
  ejecuta `pytest tests/ -v` (deben pasar 18/18) y comprueba la coherencia
  de la documentación antes de commit y push.
