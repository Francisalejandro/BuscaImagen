# -*- coding: utf-8 -*-
"""
embedding_generator.py
----------------------
Generación de embeddings de imágenes con CLIP.

Responsabilidad:
- Leer data/catalogo.json.
- Cargar las imágenes utilizando sus rutas relativas a la raíz del proyecto.
- Procesar cada imagen con el procesador CLIP (src.models.clip_model).
- Generar el embedding visual con model.get_image_features().
- Normalizar el vector (norma L2) para comparar por similitud del coseno.
- Guardar los resultados en:
    * data/embeddings_imagenes.npz   → matriz de vectores + ids
    * data/metadata_embeddings.json  → metadatos en el mismo orden

Notas:
- Los errores de imagen (no existe, fichero dañado, etc.) no detienen el
  proceso: se recogen en una lista y se informan al final.
- Si el .npz ya existe, los ids ya calculados se reutilizan y no se vuelven a
  procesar (evita generar vectores duplicados innecesariamente).
- Cada vector tiene 512 dimensiones (CLIP ViT-B/32).
"""

import json

import numpy as np

try:
    from src.config import (
        BASE_DIR,
        CATALOGO_PATH,
        EMBEDDINGS_NPZ_PATH,
        METADATA_EMBEDDINGS_PATH,
        CLIP_MODEL_NAME,
    )
    from src.models.clip_model import cargar_clip
except ImportError:
    # Permite ejecutar el módulo directamente desde PyCharm
    from config import (
        BASE_DIR,
        CATALOGO_PATH,
        EMBEDDINGS_NPZ_PATH,
        METADATA_EMBEDDINGS_PATH,
        CLIP_MODEL_NAME,
    )
    from models.clip_model import cargar_clip

from PIL import Image


def cargar_catalogo():
    """Devuelve la lista de entradas de data/catalogo.json."""
    with open(CATALOGO_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def cargar_imagen(ruta_relativa):
    """Carga y devuelve una imagen RGB desde su ruta relativa al proyecto.

    Raises:
        FileNotFoundError, OSError, ValueError: si la imagen no existe,
        está dañada o no puede abrirse (lo gestiona el llamador).
    """
    ruta_absoluta = (BASE_DIR / ruta_relativa).resolve()
    imagen = Image.open(ruta_absoluta)
    return imagen.convert("RGB")


def calcular_embedding_imagen(imagen, processor, modelo, dispositivo):
    """Calcula el embedding normalizado de una imagen ya cargada (PIL).

    Returns:
        np.ndarray float32 de 512 componentes con norma 1.
    """
    import torch

    entradas = processor(images=imagen, return_tensors="pt")
    entradas = {k: v.to(dispositivo) for k, v in entradas.items()}
    with torch.no_grad():
        salida = modelo.get_image_features(**entradas)

    # transformers v4 devuelve un Tensor (batch,512); v5 devuelve un
    # BaseModelOutputWithPooling donde pooler_output ES el embedding
    # ya proyectado a 512 dimensiones.
    if isinstance(salida, torch.Tensor):
        caracteristicas = salida
    else:
        caracteristicas = salida.pooler_output

    vector = caracteristicas[0].cpu().numpy().astype(np.float32)
    norma = np.linalg.norm(vector)
    if norma > 0:
        vector = vector / norma  # normalización L2 (similitud del coseno)
    return vector


def generar_embeddings_imagenes(mostrar_progreso=False):
    """Genera (o completa) los embeddings de todas las imágenes del catálogo.

    - Reutiliza los ids presentes en embeddings_imagenes.npz si existe.
    - Procesa solo las imágenes que faltan.
    - Guarda .npz (vectores + ids) y .json (metadatos) en el mismo orden.

    Returns:
        dict con el resumen del proceso:
            dim, total, ok, reutilizados, fallo, errores (lista de dicts),
            vectores (np.ndarray), ids (list[str]), metadatos, npz_path, json_path
    """
    catalogo = cargar_catalogo()

    # --- 1. Reanudar: cargar vectores ya calculados (evita duplicar trabajo)
    existentes = {}
    if EMBEDDINGS_NPZ_PATH.exists():
        with np.load(EMBEDDINGS_NPZ_PATH) as data:
            for i, identificador in enumerate(data["ids"]):
                existentes[str(identificador)] = data["embeddings"][i]
        if mostrar_progreso:
            print(f"[EXTRAER] Ya hay {len(existentes)} embeddings calculados de anteriores ejecuciones.")

    processor, modelo, dispositivo = None, None, None
    vector_nuevo, ids, metadatos, errores, reutilizados = {}, [], [], [], 0

    # --- 2. Procesar cada imagen del catálogo
    for i, entrada in enumerate(catalogo, 1):
        identificador = entrada["id"]
        ruta_relativa = entrada["ruta"]

        # conservar los metadatos siempre (en el orden del catálogo)
        metadatos.append({
            "id": identificador,
            "nombre": entrada["nombre"],
            "categoria": entrada["categoria"],
            "descripcion": entrada["descripcion"],
            "ruta": entrada["ruta"],
        })

        if identificador in existentes:
            ids.append(identificador)
            vector_nuevo[identificador] = existentes[identificador]
            reutilizados += 1
            if mostrar_progreso:
                print(f"  [{i:2d}/{len(catalogo)}] {identificador} ... ya calculado, se reutiliza")
            continue

        try:
            imagen = cargar_imagen(ruta_relativa)
            if processor is None:  # carga perezosa: solo si hace falta
                processor, modelo, dispositivo = cargar_clip()
            vector = calcular_embedding_imagen(imagen, processor, modelo, dispositivo)
            vector_nuevo[identificador] = vector
            ids.append(identificador)
            if mostrar_progreso:
                print(f"  [{i:2d}/{len(catalogo)}] {identificador} ... OK")
        except (FileNotFoundError, OSError, ValueError) as e:
            errores.append({"id": identificador, "ruta": ruta_relativa, "error": str(e)})
            if mostrar_progreso:
                print(f"  [{i:2d}/{len(catalogo)}] {identificador} ... FALLO: {e}")

    # --- 3. Montar la matriz final (en el orden del catálogo)
    if vector_nuevo:
        dimension = len(next(iter(vector_nuevo.values())))
        embeddings = np.array(
            [vector_nuevo[identificador] for identificador in ids], dtype=np.float32
        ).reshape(len(ids), -1)
    else:
        dimension = 512
        embeddings = np.empty((0, dimension), dtype=np.float32)

    return {
        "dim": dimension,
        "total": len(catalogo),
        "ok": sum(1 for i in ids if i in vector_nuevo and i not in existentes),
        "reutilizados": reutilizados,
        "fallo": len(errores),
        "errores": errores,
        "vectores": embeddings,
        "ids": ids,
        "metadatos": metadatos,
        "npz_path": EMBEDDINGS_NPZ_PATH,
        "json_path": METADATA_EMBEDDINGS_PATH,
    }


def guardar_embeddings(resumen):
    """Guarda el .npz (vectores + ids) y el .json (metadatos, mismo orden)."""
    np.savez(
        resumen["npz_path"],
        embeddings=resumen["vectores"],
        ids=np.array(resumen["ids"]),
    )
    with open(resumen["json_path"], "w", encoding="utf-8") as f:
        json.dump(
            {
                "modelo": CLIP_MODEL_NAME,
                "dim": resumen["dim"],
                "total": len(resumen["ids"]),
                "entradas": resumen["metadatos"],
            },
            f,
            ensure_ascii=False,
            indent=4,
        )
