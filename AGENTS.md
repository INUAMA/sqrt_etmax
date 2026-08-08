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

- Crear una rama de trabajo antes de editar: `git switch -c <tipo>/<descripción>`
- No commitear directamente a `main`
- Hacer push de la rama de trabajo: `git push -u origin <branch>`
- Fusionar mediante Pull Request en GitHub

## Notas de Implementación

- **Soporte**: `[0, ∞)` con masa de probabilidad en el origen: `F(0+) = exp(-k)`.
- **PPF analítica**: Usa la rama -1 de la función W de Lambert (`scipy.special.lambertw`).
- **Ajuste por L-momentos**: Integración numérica sobre la PPF + optimización Nelder-Mead.
- **Ajuste por MLE**: Nelder-Mead sobre log-likelihood manual con penalización para parámetros no válidos.
