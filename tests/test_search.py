# -*- coding: utf-8 -*-
"""
Pruebas de la funcionalidad de búsqueda de imágenes.

En esta fase hay:
- pruebas de estructura (rutas, catálogo, imports),
- pruebas de la búsqueda real: texto -> CLIP -> LanceDB -> top-3
  (requieren CLIP ya en caché y los embeddings generados; tardan unos
  segundos).
"""

import json
from pathlib import Path
from src import config

class TestsConfig:
    """Comprobaciones de la configuración de rutas del proyecto."""

    def test_base_dir_existe(self):
        assert isinstance(config.BASE_DIR, Path)
        assert config.BASE_DIR.name == "BuscaImagen"

    def test_rutas_datos_existen(self):
        assert config.DATA_DIR.exists()
        assert config.IMAGENES_DIR.exists()

    def test_subcarpetas_de_imagenes_existen(self):
        assert config.IMAGENES_INSTRUMENTOS_DIR.exists()
        assert config.IMAGENES_COMPONENTES_DIR.exists()
        assert config.IMAGENES_COTIDIANOS_DIR.exists()

    def test_configuracion_valores(self):
        assert config.CLIP_MODEL_NAME == "openai/clip-vit-base-patch32"
        assert config.LANCE_TABLE_NAME == "imagenes"
        assert config.NUM_RESULTADOS == 3
        assert ".jpg" in config.EXTENSIONES_VALIDAS


class TestsCatalogo:
    """Comprobaciones del catálogo de imágenes."""

    def test_catalogo_existe(self):
        assert config.CATALOGO_PATH.exists()

    def test_catalogo_es_lista_json(self):
        with open(config.CATALOGO_PATH, "r", encoding="utf-8") as f:
            catalogo = json.load(f)
        assert isinstance(catalogo, list)


class TestsImportacion:
    """Comprobaciones de que los módulos se pueden importar."""

    def test_importar_clip_model(self):
        from src.models import clip_model  # noqa: F401

    def test_importar_embedding_generator(self):
        from src.embeddings import embedding_generator  # noqa: F401

    def test_importar_text_embedding(self):
        from src.embeddings import text_embedding  # noqa: F401

    def test_importar_lance_manager(self):
        from src.database.lance_manager import LanceManager  # noqa: F401

    def test_importar_image_search(self):
        from src.search.image_search import buscar_imagenes  # noqa: F401

    def test_importar_image_utils(self):
        from src.utils import image_utils  # noqa: F401


# ---------------------------------------------------------------------------
# Búsqueda real (necesita CLIP ya en caché y embeddings generados)
# ---------------------------------------------------------------------------

class TestsBusquedaReal:
    """Prueba el flujo completo: texto -> CLIP -> LanceDB -> top-3."""

    def test_busqueda_devuelve_top3_con_campos(self):
        from src.search.image_search import buscar_imagenes

        resultados = buscar_imagenes("un microscopio para observar", k=3)
        assert len(resultados) == 3
        esperados = {
            "id", "nombre", "categoria", "descripcion",
            "ruta_imagen", "similitud_visual", "similitud_textual",
            "puntuacion", "distancia",
        }
        for r in resultados:
            assert set(r.keys()) == esperados
            assert 0.0 <= r["similitud_visual"] <= 1.0
            assert 0.0 <= r["puntuacion"] <= 1.0

    def test_busqueda_microscopio_top1(self):
        from src.search.image_search import buscar_imagenes

        resultados = buscar_imagenes("un microscopio", k=3)
        assert resultados[0]["id"] == "microscopio_1"

    def test_busqueda_ordenada_por_puntuacion(self):
        from src.search.image_search import buscar_imagenes

        resultados = buscar_imagenes("una camara fotografica", k=3)
        puntuaciones = [r["puntuacion"] for r in resultados]
        assert puntuaciones == sorted(puntuaciones, reverse=True)

    def test_cadena_vacia_devuelve_lista_vacia(self):
        from src.search.image_search import buscar_imagenes

        assert buscar_imagenes("") == []
        assert buscar_imagenes("   ") == []
