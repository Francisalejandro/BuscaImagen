# -*- coding: utf-8 -*-
"""
config.py
---------
Configuración general del proyecto BuscaImagen.

Define:
- Rutas base del proyecto (siempre relativas a la carpeta raíz).
- Rutas de datos: catálogo e imágenes.
- Ruta de la base de datos LanceDB.
- Parámetros generales (número de resultados, nombre del modelo CLIP,
  nombre de la tabla de la base de datos).

Todos los métodos usan Path.resolve() para devolver rutas absolutas
a partir de rutas relativas a la raíz del proyecto, de modo que el
proyecto funcione igual desde PyCharm, desde la terminal o con pytest.
"""

from pathlib import Path


# ---------------------------------------------------------------------------
# Rutas base
# ---------------------------------------------------------------------------

# Carpeta raíz del proyecto: .../BuscaImagen
# (config.py está en src/, por lo que la raíz es dos niveles arriba)
BASE_DIR = Path(__file__).resolve().parent.parent

# Carpeta de datos
DATA_DIR = BASE_DIR / "data"

# Carpeta con las imágenes organizadas por categoría
IMAGENES_DIR = DATA_DIR / "imagenes"

# Subcarpetas de imágenes
IMAGENES_INSTRUMENTOS_DIR = IMAGENES_DIR / "instrumentos"
IMAGENES_COMPONENTES_DIR = IMAGENES_DIR / "componentes"
IMAGENES_COTIDIANOS_DIR = IMAGENES_DIR / "cotidianos"

# Archivo de catálogo: lista de imágenes con su nombre y descripción
CATALOGO_PATH = DATA_DIR / "catalogo.json"

# Carpeta donde se almacena la base de datos LanceDB
DATABASE_DIR = BASE_DIR / "database"

# ---------------------------------------------------------------------------
# Configuración del modelo y de la base de datos
# ---------------------------------------------------------------------------

# Modelo CLIP de Hugging Face que se usará para generar los embeddings.
# https://huggingface.co/openai/clip-vit-base-patch32
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Nombre de la tabla dentro de LanceDB donde se guardarán los vectores
LANCE_TABLE_NAME = "imagenes"

# Nombre de la tabla de LanceDB con los embeddings de las descripciones
# textuales del catálogo (búsqueda híbrida)
LANCE_TABLE_TEXTOS = "descripciones"

# Número de imágenes más similares a devolver en cada búsqueda
NUM_RESULTADOS = 3

# Pool de candidatos que se extraen de cada modalidad (visual y textual)
# antes de combinar las puntuaciones. El valor por defecto cubre TODA la
# tabla: la combinación "50 % visual + 50 % textual" usa cosenos reales
# con todas las imágenes y descripciones, sin penalizar a 0.0 los
# candidatos ausentes de un pool. El recorte ("min" en image_search)
# evita pedir más candidatos que registros.
CANDIDATOS_POOL = 25

# Pesos de la búsqueda híbrida: similitud visual y textual. Ambas son
# similitudes de coseno en [0, 1], por lo que son directamente comparables
# y su combinación ponderada mantiene la misma escala.
#   puntuacion_final = PESO_VISUAL * sim_visual + PESO_TEXTUAL * sim_textual
PESO_VISUAL = 0.5
PESO_TEXTUAL = 0.5

# Archive con los embeddings generados (vectores + identificadores)
EMBEDDINGS_NPZ_PATH = DATA_DIR / "embeddings_imagenes.npz"

# Archivo JSON con los metadatos de los embeddings (mismo orden que el .npz)
METADATA_EMBEDDINGS_PATH = DATA_DIR / "metadata_embeddings.json"

# Archivo con los embeddings TEXTUALES de las descripciones del catálogo
# (mismo orden que data/catalogo.json; ids incluidos en el .npz)
EMBEDDINGS_TEXTOS_NPZ_PATH = DATA_DIR / "embeddings_descripciones.npz"

# Extensiones de imagen válidas para cargar y validar
EXTENSIONES_VALIDAS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
