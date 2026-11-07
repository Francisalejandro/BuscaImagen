# -*- coding: utf-8 -*-
"""
cargar_lancedb.py
-----------------
Carga en LanceDB (local, database/) los dos conjuntos de embeddings:

1. Tabla "imagenes"    : embeddings visuales CLIP de las 25 imágenes
                         (data/embeddings_imagenes.npz + metadatos).
2. Tabla "descripciones": embeddings textuales CLIP de las 25
                          descripciones del catálogo
                          (data/embeddings_descripciones.npz), usados
                          por la búsqueda híbrida.

Ambas cargas son idempotentes (creada / sin_cambios / reconstruida) y
nunca generan registros duplicados.

Uso (desde la carpeta BuscaImagen):
    python cargar_lancedb.py [-h] [--reconstruir]

Tras la carga ejecuta las verificaciones obligatorias: 25 registros,
512 dimensiones, metadatos asociados, sin duplicados, persistencia tras
reabrir la base y búsqueda vectorial válida en las dos tablas.
"""

import argparse
import json
import os
import sys

# Asegurar ejecución desde la raíz del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd

from src.config import (
    EMBEDDINGS_NPZ_PATH,
    EMBEDDINGS_TEXTOS_NPZ_PATH,
    LANCE_TABLE_TEXTOS,
    METADATA_EMBEDDINGS_PATH,
    NUM_RESULTADOS,
)
from src.database.lance_manager import CAMPOS_METADATO, LanceManager


def cargar_tabla_imagenes(manager, reconstruir=False):
    """Carga idempotente de la tabla de imágenes (embeddings visuales)."""
    registros = manager.cargar_registros_desde_archivos(
        EMBEDDINGS_NPZ_PATH, METADATA_EMBEDDINGS_PATH
    )
    return manager.insertar_o_actualizar(registros, reconstruir=reconstruir)


def cargar_tabla_textos(manager, reconstruir=False):
    """Carga idempotente de la tabla de descripciones (embeddings textuales)."""
    if not EMBEDDINGS_TEXTOS_NPZ_PATH.exists():
        raise RuntimeError(
            "Falta data/embeddings_descripciones.npz. Ejecuta antes "
            "generar_embeddings_texto.py."
        )
    with np.load(EMBEDDINGS_TEXTOS_NPZ_PATH) as datos:
        ids = [str(i) for i in datos["ids"]]
        matriz = datos["embeddings"].astype(np.float32)
    registros = pd.DataFrame({
        "id": ids,
        "vector": [matriz[i] for i in range(len(ids))],
    })
    return manager.insertar_o_actualizar(registros, reconstruir=reconstruir)


def verificar(manager_imagenes):
    """Verificaciones obligatorias de las dos tablas.

    Returns: True si todas pasan.
    """
    print()
    print("=" * 60)
    print("VERIFICACIONES OBLIGATORIAS")
    print("=" * 60)
    resultados_ok = []

    # --- Tabla de imágenes -------------------------------------------------
    print("[TABLA imagenes]")
    n = manager_imagenes.contar_registros()
    ok1 = n == 25
    print(f"[1] Registros = 25                       : {n}  -> {'OK' if ok1 else 'FALLO'}")
    resultados_ok.append(ok1)

    flecha = manager_imagenes.tabla.to_arrow()
    vectores = np.array(
        [np.asarray(v, dtype=np.float32) for v in flecha.column("vector").to_pylist()]
    )
    ok2 = vectores.shape[1] == 512
    print(f"[2] Dimensión de los vectores = 512      : {vectores.shape}  -> {'OK' if ok2 else 'FALLO'}")
    resultados_ok.append(ok2)

    with open(METADATA_EMBEDDINGS_PATH, encoding="utf-8") as f:
        metadatos = json.load(f)["entradas"]
    esperados = {e["id"]: e for e in metadatos}
    correspondencia_ok = True
    for identificador, entrada in esperados.items():
        fila = manager_imagenes.recuperar_por_id(identificador)
        if fila.empty:
            correspondencia_ok = False
            break
        r = fila.iloc[0]
        igual = (
            str(r["id"]) == entrada["id"]
            and r["nombre"] == entrada["nombre"]
            and r["categoria"] == entrada["categoria"]
            and r["descripcion"] == entrada["descripcion"]
            and r["ruta_imagen"] == entrada["ruta"]
        )
        if not igual:
            print(f"     Campo distinto en {identificador}")
            correspondencia_ok = False
            break
    print(f"[3] Metadatos asociados a sus imágenes    : {'OK' if correspondencia_ok else 'FALLO'}")
    resultados_ok.append(correspondencia_ok)

    _, ids = manager_imagenes.contar_registros(detallado=True)
    sin_duplicados = len(ids) == len(set(ids))
    print(f"[4] Sin ids duplicados ({len(set(ids))} únicos) : {'OK' if sin_duplicados else 'FALLO'}")
    resultados_ok.append(sin_duplicados)

    ruta = manager_imagenes.ruta_bd
    manager_imagenes.cerrar()
    manager_fresco = LanceManager(ruta_bd=ruta)
    manager_fresco.conectar()
    n_fresco = manager_fresco.contar_registros()
    ok5 = n_fresco == 25
    print(f"[5] Persiste tras abrir/cerrar conexión  : {n_fresco} registros  -> {'OK' if ok5 else 'FALLO'}")
    resultados_ok.append(ok5)

    with np.load(EMBEDDINGS_NPZ_PATH) as datos:
        indice = [str(i) for i in datos["ids"]].index("microscopio_1")
        vector_consulta = datos["embeddings"][indice]
    resultados = manager_fresco.buscar_similares(vector_consulta, k=NUM_RESULTADOS)
    top1_correcto = bool(resultados) and resultados[0]["id"] == "microscopio_1"
    campos_ok = all(c in resultados[0] for c in CAMPOS_METADATO) if resultados else False
    ok6 = top1_correcto and campos_ok
    print(f"[6] Búsqueda vectorial devuelve top-{NUM_RESULTADOS}      : {'OK' if ok6 else 'FALLO'}")
    print("    Consulta: vector de 'microscopio_1'")
    for fila in resultados:
        print(f"      - {fila['id']:<22} ({fila['categoria']}) distancia={fila['distancia']:.4f}")
    resultados_ok.append(ok6)

    # --- Tabla de descripciones --------------------------------------------
    print("[TABLA descripciones]")
    manager_textos = LanceManager(ruta_bd=ruta, nombre_tabla=LANCE_TABLE_TEXTOS)
    manager_textos.conectar()
    n_textos = manager_textos.contar_registros()
    ok7 = n_textos == 25
    print(f"[7] Descripciones = 25                    : {n_textos}  -> {'OK' if ok7 else 'FALLO'}")
    resultados_ok.append(ok7)

    if n_textos:
        flecha_txt = manager_textos.tabla.to_arrow()
        ids_txt = [str(v) for v in flecha_txt.column("id").to_pylist()]
        vectores_txt = np.array(
            [np.asarray(v, dtype=np.float32) for v in flecha_txt.column("vector").to_pylist()]
        )
        ok8 = vectores_txt.shape[1] == 512
        print(f"[8] Dimensión textuales = 512             : {vectores_txt.shape}  -> {'OK' if ok8 else 'FALLO'}")
        resultados_ok.append(ok8)

        normas = np.linalg.norm(vectores_txt, axis=1)
        ok9 = float(normas.max() - normas.min()) < 1e-3 and abs(normas.max() - 1.0) < 1e-3
        print(f"[9] Norma L2 = 1 (normalizados)           : min={normas.min():.4f} max={normas.max():.4f}  -> {'OK' if ok9 else 'FALLO'}")
        resultados_ok.append(ok9)

        ok10 = len(ids_txt) == len(set(ids_txt))
        print(f"[10] Sin ids duplicados ({len(set(ids_txt))} únicos)    : {'OK' if ok10 else 'FALLO'}")
        resultados_ok.append(ok10)

        # Búsqueda textual válida: consulta con el vector de la descripción
        # de camara_1 debe devolver camara_1 como primer resultado.
        with np.load(EMBEDDINGS_TEXTOS_NPZ_PATH) as datos:
            indice_txt = [str(i) for i in datos["ids"]].index("camara_1")
            vector_txt = datos["embeddings"][indice_txt]
        resultados_txt = manager_textos.buscar_similares(vector_txt, k=3, campos=["id"])
        top1_txt = bool(resultados_txt) and resultados_txt[0]["id"] == "camara_1"
        ok11 = top1_txt
        print(f"[11] Búsqueda textual válida (top-3)      : {'OK' if ok11 else 'FALLO'}")
        print("    Consulta: vector de la descripción de 'camara_1'")
        for fila in resultados_txt:
            print(f"      - {fila['id']:<22} distancia={fila['distancia']:.4f}")
        resultados_ok.append(ok11)

    manager_textos.cerrar()

    print()
    if all(resultados_ok):
        print("OK Todas las verificaciones pasaron.")
        return True
    print("FALLO: hay verificaciones que no pasan. Revisa el detalle de arriba.")
    return False


def main():
    parser = argparse.ArgumentParser(description="BuscaImagen - Carga en LanceDB")
    parser.add_argument(
        "--reconstruir", action="store_true",
        help="borra las tablas existentes y las reconstruye desde los archivos",
    )
    argumentos = parser.parse_args()
    reconstruir = argumentos.reconstruir

    print("=" * 60)
    print("BuscaImagen - Carga de embeddings en LanceDB")
    print("=" * 60)

    # --- Tabla de imágenes
    manager = LanceManager()
    manager.conectar()
    print(f"Base de datos : {manager.ruta_bd}")

    accion_imagenes, ids = cargar_tabla_imagenes(manager, reconstruir=reconstruir)
    n, _ = manager.contar_registros(detallado=True)
    print()
    print("RESUMEN")
    print(f"Tabla imagenes     : {accion_imagenes}")
    print(f"Registros imagenes : {n}")

    # --- Tabla de descripciones
    manager_textos = LanceManager(nombre_tabla=LANCE_TABLE_TEXTOS)
    manager_textos.conectar()
    try:
        accion_textos, _ = cargar_tabla_textos(manager_textos, reconstruir=reconstruir)
        n_textos, _ = manager_textos.contar_registros(detallado=True)
        print(f"Tabla descripciones: {accion_textos}")
        print(f"Registros descripciones: {n_textos}")
    except RuntimeError as e:
        print(f"ATENCIÓN: {e}")
    print(f"Carpeta de base de datos: database/ (LanceDB local)")

    todo_ok = verificar(manager)
    sys.exit(0 if todo_ok else 1)


if __name__ == "__main__":
    main()
