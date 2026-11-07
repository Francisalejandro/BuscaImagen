# -*- coding: utf-8 -*-
"""
Pruebas de la búsqueda híbrida (visual + textual).

Comprueban: artefactos de embeddings textuales correctos y alineados,
tabla "descripciones" íntegra, estructura y coherencia de los resultados
híbridos, el caso que motivó la fase («Componente que emite luz» → LED),
el tratamiento de consultas vacías y el control de pesos.

Requiere data/embeddings_descripciones.npz y las dos tablas de LanceDB
(ejecutar antes generar_embeddings_texto.py y cargar_lancedb.py).
"""

import json

import numpy as np

from src import config
from src.database.lance_manager import LanceManager

CATALOGO_PATH = config.CATALOGO_PATH
EMBEDDINGS_TEXTOS_NPZ_PATH = config.EMBEDDINGS_TEXTOS_NPZ_PATH


def _ids_catalogo():
    with open(CATALOGO_PATH, encoding="utf-8") as f:
        return [e["id"] for e in json.load(f)]


class TestsEmbeddingsTextuales:
    """El .npz de descripciones existe, está completo y normalizado."""

    def test_npz_textual_existe(self):
        assert EMBEDDINGS_TEXTOS_NPZ_PATH.exists()

    def test_npz_textual_25_por_512(self):
        with np.load(EMBEDDINGS_TEXTOS_NPZ_PATH) as datos:
            assert datos["embeddings"].shape == (25, 512)

    def test_npz_textual_normalizado(self):
        with np.load(EMBEDDINGS_TEXTOS_NPZ_PATH) as datos:
            normas = np.linalg.norm(datos["embeddings"], axis=1)
        assert float(np.abs(normas - 1.0).max()) < 1e-4

    def test_ids_alineados_con_catalogo(self):
        with np.load(EMBEDDINGS_TEXTOS_NPZ_PATH) as datos:
            ids_npz = [str(i) for i in datos["ids"]]
        assert ids_npz == _ids_catalogo()


class TestsTablaDescripciones:
    """La tabla 'descripciones' de LanceDB está íntegra."""

    def test_tabla_tiene_veinticinco(self):
        from src.search.image_search import obtener_gestor_textos

        manager = obtener_gestor_textos()
        n, ids = manager.contar_registros(detallado=True)
        assert n == 25
        assert ids == _ids_catalogo()

    def test_tabla_imagenes_intacta(self):
        from src.search.image_search import obtener_gestor

        manager = obtener_gestor()
        n, ids = manager.contar_registros(detallado=True)
        assert n == 25
        assert ids == _ids_catalogo()


class TestsBusquedaHibrida:
    """Estructura, coherencia y orden del método híbrido."""

    def test_estructura_campos(self):
        from src.search.image_search import buscar_imagenes_hibrida

        resultados = buscar_imagenes_hibrida("un microscopio", k=3)
        assert len(resultados) == 3
        esperados = {
            "id", "nombre", "categoria", "descripcion", "ruta_imagen",
            "similitud_visual", "similitud_textual", "puntuacion", "distancia",
        }
        for fila in resultados:
            assert set(fila.keys()) == esperados
            for campo in ("similitud_visual", "similitud_textual", "puntuacion"):
                assert 0.0 <= fila[campo] <= 1.0001

    def test_puntuacion_coherente_con_pesos(self):
        from src.config import PESO_TEXTUAL, PESO_VISUAL
        from src.search.image_search import buscar_imagenes_hibrida

        resultados = buscar_imagenes_hibrida("una camara fotografica", k=3)
        for fila in resultados:
            esperada = PESO_VISUAL * fila["similitud_visual"] \
                + PESO_TEXTUAL * fila["similitud_textual"]
            assert abs(fila["puntuacion"] - esperada) < 1e-3

    def test_resultados_ordenados_descendente(self):
        from src.search.image_search import buscar_imagenes_hibrida

        resultados = buscar_imagenes_hibrida("Dispositivo para capturar fotografías", k=3)
        puntuaciones = [f["puntuacion"] for f in resultados]
        assert puntuaciones == sorted(puntuaciones, reverse=True)

    def test_consulta_vacia_devuelve_lista_vacia(self):
        from src.search.image_search import (
            buscar_imagenes_hibrida,
            buscar_solo_visual,
        )

        for consulta in ("", "   "):
            assert buscar_imagenes_hibrida(consulta) == []
            assert buscar_solo_visual(consulta) == []

    def test_ruta_con_imagen_correspondiente(self):
        from src.config import BASE_DIR
        from src.search.image_search import buscar_imagenes_hibrida

        resultados = buscar_imagenes_hibrida("un microscopio", k=3)
        for fila in resultados:
            assert (BASE_DIR / fila["ruta_imagen"]).exists()
            assert fila["ruta_imagen"].startswith("data/imagenes/")

    def test_caso_componente_que_emite_luz_devuelve_led(self):
        """Caso que motivó la fase híbrida: «Componente que emite luz»."""
        from src.search.image_search import buscar_imagenes_hibrida

        resultados = buscar_imagenes_hibrida("Componente que emite luz", k=3)
        assert resultados[0]["id"] == "led_1"

    def test_peso_visual_1_regresa_al_metodo_visual(self):
        """Con peso visual 1.0 el ranking del híbrido sigue al visual."""
        from src.search.image_search import (
            buscar_imagenes_hibrida,
            buscar_solo_visual,
        )

        vis = buscar_solo_visual("una camara fotografica", k=3)
        hib = buscar_imagenes_hibrida("una camara fotografica", k=3, peso_visual=1.0)
        assert hib[0]["id"] == vis[0]["id"]
        assert hib[0]["puntuacion"] == vis[0]["similitud_visual"]

    def test_buscar_imagenes_api_hibrida_por_defecto(self):
        from src.search.image_search import buscar_imagenes

        resultados = buscar_imagenes("un microscopio", k=3)
        assert len(resultados) == 3
        assert "puntuacion" in resultados[0]
