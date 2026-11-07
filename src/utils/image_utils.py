# -*- coding: utf-8 -*-
"""
image_utils.py
--------------
Utilidades para cargar y validar imágenes.

Responsabilidad prevista (por implementar):
- Listar las imágenes de una carpeta (o de todas las subcarpetas de
  data/imagenes/) filtrando por las extensiones válidas definidas en
  src.config.EXTENSIONES_VALIDAS.
- Cargar una imagen con Pillow (PIL.Image) y devolverla lista para el
  procesador de CLIP.
- Validar imágenes: comprobar que el fichero existe, que es legible y
  que la extensión está soportada.
- Calcular rutas relativas a la raíz del proyecto para guardarlas en el
  catálogo y en LanceDB, de forma que nada dependa de la ruta absoluta
  del equipo de cada alumno.

Notas:
- Las imágenes serán incorporadas posteriormente por el usuario; este
  módulo no descarga ni genera imágenes.
- En esta fase inicial el módulo solo documenta esa responsabilidad.
"""

try:
    from src.config import IMAGENES_DIR, EXTENSIONES_VALIDAS  # noqa: F401
except ImportError:
    from config import IMAGENES_DIR, EXTENSIONES_VALIDAS  # noqa: F401
