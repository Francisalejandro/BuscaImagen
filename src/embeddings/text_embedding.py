# -*- coding: utf-8 -*-
"""
text_embedding.py
-----------------
Generación de embeddings TEXTUALES de las descripciones del catálogo
con CLIP (misma torre de texto que ya se usa para las consultas).

Responsabilidad:
- Leer data/catalogo.json (descripciones reales del proyecto; nunca se
  inventan textos).
- Codificar la descripción de cada imagen con model.get_text_features()
  del MISMO modelo openai/clip-vit-base-patch32 (no se descarga otro).
- Normalizar cada vector con la norma L2 (mismo criterio que los
  embeddings visuales → similitud del coseno comparable).
- Guardar data/embeddings_descripciones.npz con matriz (25, 512) + ids,
  en el orden de data/catalogo.json.

Notas:
- Idempotente: si el .npz ya existe, los ids ya calculados se reutilizan
  (igual que en la generación de embeddings visuales).
- Los errores por descripción no detienen el proceso: se recogen y se
  informan al final.
- No usa coincidencias literales de texto: los embeddings se comparan
  después por similitud del coseno en el espacio de CLIP.
"""

import json

import numpy as np

try:
    from src.config import (
        CATALOGO_PATH,
        CLIP_MODEL_NAME,
        EMBEDDINGS_TEXTOS_NPZ_PATH,
    )
    from src.embeddings.embedding_generator import cargar_catalogo
    from src.models.clip_model import cargar_clip
except ImportError:
    # Permite ejecutar el módulo directamente desde PyCharm
    from config import (
        CATALOGO_PATH,
        CLIP_MODEL_NAME,
        EMBEDDINGS_TEXTOS_NPZ_PATH,
    )
    from embeddings.embedding_generator import cargar_catalogo
    from models.clip_model import cargar_clip


def calcular_embedding_texto(texto, processor, modelo, dispositivo):
    """Calcula el embedding normalizado de un texto con la torre de texto CLIP.

    Se usa el texto tal cual (sin plantillas): las descripciones del
    catálogo ya son frases completas y el objetivo es comparar consultas
    y descripciones en el mismo espacio textual.

    Returns:
        np.ndarray float32 de 512 componentes con norma 1.

    Raises:
        ValueError: si el texto está vacío.
    """
    import torch

    if not texto or not texto.strip():
        raise ValueError("No se puede calcular el embedding de un texto vacío.")

    entradas = processor(text=[texto.strip()], return_tensors="pt", padding=True)
    entradas = {campo: valor.to(dispositivo) for campo, valor in entradas.items()}
    with torch.no_grad():
        salida = modelo.get_text_features(**entradas)

    # transformers v4 devuelve un Tensor; v5 devuelve BaseModelOutputWithPooling
    # con el embedding proyectado a 512 dimensiones en .pooler_output.
    if isinstance(salida, torch.Tensor):
        caracteristicas = salida
    else:
        caracteristicas = salida.pooler_output

    vector = caracteristicas[0].cpu().numpy().astype(np.float32)
    norma = np.linalg.norm(vector)
    if norma > 0:
        vector = vector / norma  # normalización L2 (similitud del coseno)
    return vector


def generar_embeddings_textos(mostrar_progreso=False):
    """Genera (o completa) los embeddings de las descripciones del catálogo.

    - Lee CATALOGO_PATH y codifica cada campo "descripcion" REAL.
    - Reutiliza los ids presentes en embeddings_descripciones.npz si existe.
    - Procesa solo las descripciones que faltan.
    - Guarda el .npz con orden fijo: el de data/catalogo.json.

    Returns:
        dict con el resumen: dim, total, ok, reutilizados, fallo, errores,
        vectores, ids, descripciones, npz_path.
    """
    catalogo = cargar_catalogo()

    # --- 1. Reanudar: cargar vectores ya calculados (evita duplicar trabajo)
    existentes = {}
    if EMBEDDINGS_TEXTOS_NPZ_PATH.exists():
        with np.load(EMBEDDINGS_TEXTOS_NPZ_PATH) as datos:
            for i, identificador in enumerate(datos["ids"]):
                existentes[str(identificador)] = datos["embeddings"][i]
        if mostrar_progreso:
            print(
                f"[TEXTO] Ya hay {len(existentes)} embeddings textuales de "
                "ejecuciones anteriores, se reutilizan."
            )

    processor, modelo, dispositivo = None, None, None
    vectores_nuevos, ids, descripciones, errores, reutilizados = {}, [], [], [], 0

    # --- 2. Codificar cada descripción (en el orden del catálogo)
    for i, entrada in enumerate(catalogo, 1):
        identificador = entrada["id"]
        texto = entrada["descripcion"]
        ids.append(identificador)
        descripciones.append(texto)

        if identificador in existentes:
            vectores_nuevos[identificador] = existentes[identificador]
            reutilizados += 1
            if mostrar_progreso:
                print(f"  [{i:2d}/{len(catalogo)}] {identificador} ... ya calculado, se reutiliza")
            continue

        try:
            if processor is None:  # carga perezosa: solo si hace falta
                processor, modelo, dispositivo = cargar_clip()
            vector = calcular_embedding_texto(texto, processor, modelo, dispositivo)
            vectores_nuevos[identificador] = vector
            if mostrar_progreso:
                print(f"  [{i:2d}/{len(catalogo)}] {identificador} ... OK")
        except (ValueError, OSError, RuntimeError) as e:
            errores.append({"id": identificador, "error": str(e)})
            if mostrar_progreso:
                print(f"  [{i:2d}/{len(catalogo)}] {identificador} ... FALLO: {e}")

    ids_validos = [i for i in ids if i in vectores_nuevos]
    if ids_validos:
        dimension = len(next(iter(vectores_nuevos.values())))
        embeddings = np.array(
            [vectores_nuevos[i] for i in ids_validos], dtype=np.float32
        ).reshape(len(ids_validos), -1)
    else:
        dimension = 512
        embeddings = np.empty((0, dimension), dtype=np.float32)

    # --- 3. Guardar (solo si todas las descripciones del catálogo quedaron)
    if not errores:
        np.savez(
            EMBEDDINGS_TEXTOS_NPZ_PATH,
            embeddings=embeddings,
            ids=np.array(ids_validos),  # unicode: se carga sin allow_pickle
        )

    return {
        "modelo": CLIP_MODEL_NAME,
        "dim": dimension,
        "total": len(catalogo),
        "ok": sum(1 for i in ids_validos if i in vectores_nuevos and i not in existentes),
        "reutilizados": reutilizados,
        "fallo": len(errores),
        "errores": errores,
        "vectores": embeddings,
        "ids": ids_validos,
        "descripciones": descripciones,
        "npz_path": EMBEDDINGS_TEXTOS_NPZ_PATH,
        "catalogo_path": CATALOGO_PATH,
    }
