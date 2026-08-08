# Guía de Contribución

¡Gracias por tu interés en contribuir a `sqrt_etmax`! Todas las aportaciones son bienvenidas.

## ¿Cómo contribuir?

1. Haz un *fork* del repositorio.
2. Crea una rama para tu característica o corrección de error (`git checkout -b feature/nueva-caracteristica`).
3. Escribe tu código (asegúrate de seguir el estilo del proyecto y documentar tus cambios).
4. Escribe o actualiza las pruebas pertinentes en la carpeta `tests/`.
5. Asegúrate de que las pruebas pasan ejecutando `pytest`.
6. Haz *commit* de tus cambios (`git commit -m 'Añade nueva característica'`).
7. Sube los cambios a tu rama (`git push origin feature/nueva-caracteristica`).
8. Abre un *Pull Request* en GitHub.

## Convenciones de Commits

Usamos [Convenciones de Commits](https://www.conventionalcommits.org/es/v1.0.0/):

- `feat:` nueva funcionalidad
- `fix:` corrección de error
- `docs:` cambios en documentación
- `test:` añadir o corregir pruebas
- `refactor:` reestructuración sin cambio de comportamiento
- `ci:` cambios en configuración de CI
- `chore:` tareas de mantenimiento

## Pruebas Unitarias

Usamos `pytest` para garantizar el correcto funcionamiento del paquete. Antes de enviar tu código, instala las dependencias de desarrollo y corre las pruebas:

```bash
pip install -e .[test]
pytest tests/ -v
```

## Publicación de Versiones

### Release Checklist

Antes de publicar una nueva versión:

1. Actualizar la versión en `pyproject.toml` (seguir semver).
2. Actualizar `CHANGELOG.md` con los cambios de la nueva versión.
3. Verificar que todos los tests pasan: `pytest tests/ -v`.
4. Verificar que el paquete se construye correctamente: `python -m build`.
5. Crear un tag de versión: `git tag vX.Y.Z`.
6. Hacer push del tag: `git push origin vX.Y.Z`.
7. Crear un release en GitHub desde el tag.
8. El workflow `publish.yml` publicará automáticamente en PyPI.

## Reporte de Errores

Si encuentras un error o un resultado matemático inesperado en la distribución estadística, por favor abre un *Issue* en GitHub incluyendo:

- Una descripción clara del problema.
- El comportamiento esperado según la literatura (Etoh et al., Norma 5.2-IC).
- Código o datos de ejemplo para reproducir el fallo.

¡Gracias por ayudar a mejorar las herramientas de hidrología!
