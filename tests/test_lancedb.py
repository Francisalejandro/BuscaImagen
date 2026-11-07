# -*- coding: utf-8 -*-
"""
Pruebas de la fase de LanceDB.

Verifican (sobre una base de datos temporal, sin tocar database/ del
proyecto):
- Que existen exactamente 25 registros.
- Que cada vector tiene 512 dimensiones.
- Que los metadatos están correctamente asociados a sus imágenes.
- Que no existen registros duplicados.
- Que la base de datos persiste después de cerrar y volver a abrir.
- Que una búsqueda vectorial devuelve resultados válidos.
- Que el proceso de carga es idempotente (no crea duplicados al repetirse).
"""

import json

import numpy as np
import pytest

from src.config import (
    EMBEDDINGS_NPZ_PATH,
    METADATA_EMBEDDINGS_PATH,
    NUM_RESULTADOS,
)
from src.database.lance_manager import LanceManager


@pytest.fixture(scope="module")
def manager(tmp_path_factory):
    """Base de datos temporal con los 25 registros cargados una sola vez."""
    ruta = tmp_path_factory.mktemp("bd_pruebas")
    gestor = LanceManager(ruta_bd=ruta / "database_practica")
    gestor.conectar()
    registros = gestor.cargar_registros_desde_archivos(
        EMBEDDINGS_NPZ_PATH, METADATA_EMBEDDINGS_PATH
    )
    accion, _ = gestor.insertar_o_actualizar(registros)
    assert accion == "creada"
    yield gestor
    gestor.cerrar()


# ---------------------------------------------------------------------------
# Verificaciones obligatorias
# ---------------------------------------------------------------------------

def test_veinticinco_registros(manager):
    assert manager.contar_registros() == 25


def test_vectores_512_dimensiones(manager):
    flecha = manager.tabla.to_arrow()
    vectores = [np.asarray(v, dtype=np.float32) for v in flecha.column("vector").to_pylist()]
    assert len(vectores) == 25
    assert all(v.shape == (512,) for v in vectores)


def test_metadatos_asociados(manager):
    with open(METADATA_EMBEDDINGS_PATH, encoding="utf-8") as f:
        metadatos = json.load(f)["entradas"]
    for entrada in metadatos:
        fila = manager.recuperar_por_id(entrada["id"])
        assert len(fila) == 1, f"El id {entrada['id']} no aparece una única vez"
        r = fila.iloc[0]
        assert str(r["id"]) == entrada["id"]
        assert r["nombre"] == entrada["nombre"]
        assert r["categoria"] == entrada["categoria"]
        assert r["descripcion"] == entrada["descripcion"]
        assert r["ruta_imagen"] == entrada["ruta"]


def test_metadato_de_una_fila_concreta(manager):
    fila = manager.recuperar_por_id("microscopio_1")
    r = fila.iloc[0]
    assert r["nombre"] == "microscopio"
    assert r["categoria"] == "instrumentos"
    assert "pequeños" in r["descripcion"]
    assert r["ruta_imagen"] == "data/imagenes/instrumentos/microscopio_1.jpg"


def test_sin_duplicados(manager):
    _, ids = manager.contar_registros(detallado=True)
    assert len(ids) == len(set(ids)) == 25


def test_persistencia_tras_reabrir(manager):
    """Cierra la conexión y la reabre: los datos siguen ahí."""
    ruta = manager.ruta_bd
    manager.cerrar()

    gestor_fresco = LanceManager(ruta_bd=ruta)
    gestor_fresco.conectar()
    assert gestor_fresco.tabla_existe()
    assert gestor_fresco.contar_registros() == 25
    # el vector persistido coincide con el NPZ original (sin modificaciones)
    _, ids = gestor_fresco.contar_registros(detallado=True)
    fila = gestor_fresco.recuperar_por_id(ids[0])
    vector_bd = np.asarray(fila.iloc[0]["vector"], dtype=np.float32)
    with np.load(EMBEDDINGS_NPZ_PATH) as datos:
        ids_npz = [str(i) for i in datos["ids"]]
        vector_npz = datos["embeddings"][ids_npz.index(ids[0])]
    assert np.allclose(vector_bd, vector_npz, atol=1e-6)


def test_busqueda_vectorial_valida(manager):
    with np.load(EMBEDDINGS_NPZ_PATH) as datos:
        ids_npz = [str(i) for i in datos["ids"]]
        indice = ids_npz.index("microscopio_1")
        vector = datos["embeddings"][indice]
    resultados = manager.buscar_similares(vector, k=NUM_RESULTADOS)
    assert len(resultados) == NUM_RESULTADOS
    assert resultados[0]["id"] == "microscopio_1"  # autodistancia ⇒ primero
    distancias = [r["distancia"] for r in resultados]
    assert distancias == sorted(distancias)  # ordenados de más a menos cercano
    for r in resultados:
        assert set(r.keys()) == {
            "id", "nombre", "categoria", "descripcion",
            "ruta_imagen", "distancia",
        }


def test_insercion_repetida_no_duplica(manager):
    registros = manager.cargar_registros_desde_archivos(
        EMBEDDINGS_NPZ_PATH, METADATA_EMBEDDINGS_PATH
    )
    accion, _ = manager.insertar_o_actualizar(registros)
    assert accion == "sin_cambios"
    assert manager.contar_registros() == 25


def test_reconstruccion_forzada_sin_duplicados(manager):
    registros = manager.cargar_registros_desde_archivos(
        EMBEDDINGS_NPZ_PATH, METADATA_EMBEDDINGS_PATH
    )
    accion, ids = manager.insertar_o_actualizar(registros, reconstruir=True)
    assert accion == "reconstruida"
    assert len(ids) == len(set(ids)) == 25
