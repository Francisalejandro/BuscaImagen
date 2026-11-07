# -*- coding: utf-8 -*-
"""
image_search.py
---------------
Lógica de búsqueda de BuscaImagen: dado un texto en lenguaje natural,
recupera las NUM_RESULTADOS imágenes más relevantes combinando dos
modalidades (búsqueda híbrida) o solo la visual.

Modalidades
-----------
VISUAL: coseno entre el embedding de la consulta y los embeddings de las
IMÁGENES (columna vector de la tabla "imagenes"). Es la búsqueda clásica
de las fases anteriores; es fuerte con consultas de nombres de objetos
("un microscopio") y débil con paráfrasis conceptuales en español (la
torre de texto de CLIP se entrenó predominantemente con pares en
inglés; ver §Limitaciones del README).

TEXTUAL: coseno entre el embedding de la consulta y los embeddings de
las descripciones REALES del catálogo (tabla "descripciones"). Es más
robusta con consultas funcionales en español ("Componente que emite
luz"), redactadas de forma parecida a las descripciones.

HÍBRIDA (por defecto):
    puntuacion_final = PESO_VISUAL * sim_visual + PESO_TEXTUAL * sim_textual

Ambas similitudes son cosenos (rango matemático [-1, 1]) convertidos y
recortados al intervalo [0, 1] por _similitud_desde_distancia; con los
datos actuales no surgen cosenos negativos, de modo que las escalas son
directamente comparables y la media ponderada conserva esa escala.
Configurar PESO_VISUAL / PESO_TEXTUAL en src/config.py.

Consultas
---------
- Visual:      la consulta se procesa con plantillas ("{q}",
               "a photo of {q}", "imagen de {q}") elegidas empíricamente;
               se promedian los embeddings normalizados.
- Textual:     la consulta se usa tal cual (sin plantillas): las
               descripciones del catálogo son frases completas en
               español y la comparación se hace dentro del espacio
               textual, no por coincidencia literal.

Datos
-----
Si falta la tabla "imagenes" se reconstruye desde
data/embeddings_imagenes.npz + data/metadata_embeddings.json.
Si falta la tabla "descripciones" se construye desde
data/embeddings_descripciones.npz (ids alineados con el catálogo).
"""

import numpy as np
import pandas as pd
import torch

try:
    from src.config import (
        CANDIDATOS_POOL,
        EMBEDDINGS_NPZ_PATH,
        EMBEDDINGS_TEXTOS_NPZ_PATH,
        LANCE_TABLE_TEXTOS,
        METADATA_EMBEDDINGS_PATH,
        NUM_RESULTADOS,
        PESO_TEXTUAL,
        PESO_VISUAL,
    )
    from src.database.lance_manager import LanceManager
    from src.embeddings.text_embedding import calcular_embedding_texto
    from src.models.clip_model import cargar_clip
except ImportError:
    # Permite ejecutar el módulo directamente desde PyCharm
    from config import (
        CANDIDATOS_POOL,
        EMBEDDINGS_NPZ_PATH,
        EMBEDDINGS_TEXTOS_NPZ_PATH,
        LANCE_TABLE_TEXTOS,
        METADATA_EMBEDDINGS_PATH,
        NUM_RESULTADOS,
        PESO_TEXTUAL,
        PESO_VISUAL,
    )
    from database.lance_manager import LanceManager
    from embeddings.text_embedding import calcular_embedding_texto
    from models.clip_model import cargar_clip

# Plantillas de la modalidad VISUAL, elegidas empíricamente con el
# catálogo actual (fase de diagnóstico): "a photo of {q}" e "imagen de
# {q}" maximizan la similitud con la imagen objetivo; el texto crudo
# estabiliza la consulta cuando ya viene bien redactada.
PLANTILLAS_VISUALES = [
    "{q}",
    "a photo of {q}",
    "imagen de {q}",
]

CAMPOS_RESULTADO = [
    "id", "nombre", "categoria", "descripcion", "ruta_imagen",
    "similitud_visual", "similitud_textual", "puntuacion", "distancia",
]


def _embedding_texto(descripcion, processor, modelo, dispositivo, plantillas=PLANTILLAS_VISUALES):
    """Embedding de la consulta combinando plantillas y renormalizando."""
    frases = [plantilla.replace("{q}", descripcion.strip()) for plantilla in plantillas]
    entradas = processor(
        text=frases, return_tensors="pt", padding=True, truncation=True
    )
    entradas = {campo: valor.to(dispositivo) for campo, valor in entradas.items()}
    with torch.no_grad():
        salida = modelo.get_text_features(**entradas)
    caracteristicas = salida if hasattr(salida, "shape") else salida.pooler_output
    mat = caracteristicas.cpu().numpy().astype("float32")  # (T, 512)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)     # normalizar cada plantilla
    vector = mat.mean(axis=0)                             # promedio
    vector = vector / np.linalg.norm(vector)              # renormalizar
    return vector


def obtener_gestor(crear_si_falta=True):
    """Devuelve un LanceManager de la tabla de IMÁGENES conectado.

    Si la tabla no existe, la reconstruye desde los archivos de
    embeddings visuales (idempotente, sin duplicados).
    """
    manager = LanceManager()
    manager.conectar()
    if manager.tabla_existe():
        return manager
    if not crear_si_falta:
        raise RuntimeError(
            f"No existe la tabla '{manager.nombre_tabla}' en {manager.ruta_bd}. "
            "Ejecuta cargar_lancedb.py."
        )
    registros = manager.cargar_registros_desde_archivos(
        EMBEDDINGS_NPZ_PATH, METADATA_EMBEDDINGS_PATH
    )
    manager.insertar_o_actualizar(registros)
    return manager


def obtener_gestor_textos(crear_si_falta=True):
    """Devuelve un LanceManager de la tabla de DESCRIPCIONES conectado.

    Si la tabla no existe, la construye desde embeddings_descripciones.npz
    (columnas: id, vector). Requiere haber ejecutado antes
    generar_embeddings_texto.py.
    """
    manager = LanceManager(nombre_tabla=LANCE_TABLE_TEXTOS)
    manager.conectar()
    if manager.tabla_existe():
        return manager
    if not crear_si_falta:
        raise RuntimeError(
            f"No existe la tabla '{manager.nombre_tabla}' en {manager.ruta_bd}. "
            "Ejecuta generar_embeddings_texto.py y luego cargar_lancedb.py."
        )
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
    manager.insertar_o_actualizar(registros)
    return manager


def _similitud_desde_distancia(distancia):
    """Convierte _distance de LanceDB (L2 AL CUADRADO) a coseno en [0, 1].

    Con vectores unitarios: d² = 2 - 2·cos  ⇒  cos = 1 - d²/2 = 1 - d/2.
    (Verificado empíricamente con el catálogo en la fase de diagnóstico.)
    """
    return max(0.0, min(1.0, 1.0 - float(distancia) / 2.0))


def buscar_solo_visual(descripcion, k=NUM_RESULTADOS, manager=None):
    """Búsqueda clásica solo visual (comparativa con el método híbrido).

    Returns:
        lista de dicts {id, nombre, categoria, descripcion, ruta_imagen,
        distancia, similitud_visual}. La clave 'similitud_visual' contiene
        la única similitud de este método.
    """
    if not descripcion or not descripcion.strip():
        return []

    processor, modelo, dispositivo = cargar_clip()
    vector = _embedding_texto(descripcion, processor, modelo, dispositivo)

    if manager is None:
        manager = obtener_gestor()
    if not manager.tabla_existe():
        manager.conectar()

    resultados = manager.buscar_similares(vector, k=k)
    for fila in resultados:
        fila["similitud_visual"] = round(
            _similitud_desde_distancia(fila.pop("distancia")), 4
        )
    return resultados


def _fila_hibrida(identificador, sim_visual, sim_textual, puntuacion,
                  filas_visuales, manager_visual):
    """Junta la puntuación híbrida con los metadatos del catálogo."""
    fila = filas_visuales.get(identificador)
    if fila is None:
        registro = manager_visual.recuperar_por_id(identificador)
        if registro.empty:
            return None  # puntuación sin imagen asociada: se descarta
        r = registro.iloc[0]
        fila = {
            "id": str(r["id"]),
            "nombre": r["nombre"],
            "categoria": r["categoria"],
            "descripcion": r["descripcion"],
            "ruta_imagen": r["ruta_imagen"],
            "distancia": None,
        }
    salida = {campo: fila[campo] for campo in
              ("id", "nombre", "categoria", "descripcion", "ruta_imagen")}
    salida["similitud_visual"] = round(sim_visual, 4)
    salida["similitud_textual"] = round(sim_textual, 4)
    salida["puntuacion"] = round(puntuacion, 4)
    salida["distancia"] = fila.get("distancia")
    return salida


def buscar_imagenes_hibrida(descripcion, k=NUM_RESULTADOS, manager_visual=None,
                            manager_textos=None, peso_visual=PESO_VISUAL):
    """Búsqueda híbrida visual + textual (método principal del proyecto).

    Proceso:
    1. Embedding de la consulta con plantillas (modalidad visual) y tal
       cual (modalidad textual), ambos con la misma torre de texto CLIP.
    2. Se extrae un pool de candidatos de cada modalidad con LanceDB
       (coseno descendente). El pool se recorta al número real de
       registros: `pool = min(CANDIDATOS_POOL, nº de registros)`; con
       el valor por defecto (25) el pool cubre TODA la tabla y cada
       modalidad puntúa con su coseno real, sin penalizaciones por
       ausencia.
    3. Se combinan las similitudes de coseno (convertidas y recortadas al
       intervalo [0, 1] por _similitud_desde_distancia) de las dos
       modalidades con la media ponderada de los pesos configurados:
            puntuacion = PESO_VISUAL * sim_visual
                       + PESO_TEXTUAL * sim_textual
       (un candidato ausente de una modalidad puntúa 0.0 en ella; con el
       pool por defecto esa situación no se produce).
    4. Se ordena por puntuación descendente y se devuelven los primeros k.

    Returns:
        lista de dicts con CAMPOS_RESULTADO ordenados de más a menos
        relevantes. 'distancia' es la distancia L2 visual (None si el
        candidato solo entró por la modalidad textual).
    """
    if not descripcion or not descripcion.strip():
        return []

    peso_visual = float(peso_visual)
    peso_textual = 1.0 - peso_visual

    processor, modelo, dispositivo = cargar_clip()
    vector_visual = _embedding_texto(descripcion, processor, modelo, dispositivo)
    vector_textual = calcular_embedding_texto(descripcion, processor, modelo, dispositivo)

    if manager_visual is None:
        manager_visual = obtener_gestor()
    if manager_textos is None:
        manager_textos = obtener_gestor_textos()
    if not manager_visual.tabla_existe():
        manager_visual.conectar()
    if not manager_textos.tabla_existe():
        manager_textos.conectar()

    # --- modalidad visual: pool de candidatos con metadatos
    pool_visual = min(CANDIDATOS_POOL, manager_visual.contar_registros())
    filas_visuales = {}
    sims_visuales = {}
    for fila in manager_visual.buscar_similares(vector_visual, k=pool_visual):
        distancia = fila.pop("distancia")
        identificador = str(fila["id"])
        fila["distancia"] = distancia
        filas_visuales[identificador] = fila
        sims_visuales[identificador] = _similitud_desde_distancia(distancia)

    # --- modalidad textual: pool de candidatos (id + coseno)
    pool_textual = min(CANDIDATOS_POOL, manager_textos.contar_registros())
    sims_textuales = {}
    for fila in manager_textos.buscar_similares(vector_textual,
                                                k=pool_textual, campos=["id"]):
        sims_textuales[str(fila["id"])] = _similitud_desde_distancia(fila["distancia"])

    # --- combinación ponderada sobre la unión de candidatos
    combinados = []
    for identificador in set(sims_visuales) | set(sims_textuales):
        sim_vis = sims_visuales.get(identificador, 0.0)
        sim_txt = sims_textuales.get(identificador, 0.0)
        puntuacion = peso_visual * sim_vis + peso_textual * sim_txt
        combinados.append((identificador, sim_vis, sim_txt, puntuacion))
    combinados.sort(key=lambda t: t[3], reverse=True)

    salida = []
    for identificador, sim_vis, sim_txt, puntuacion in combinados[:k]:
        fila = _fila_hibrida(identificador, sim_vis, sim_txt, puntuacion,
                             filas_visuales, manager_visual)
        if fila is not None:
            salida.append(fila)
    return salida


def buscar_imagenes(descripcion, k=NUM_RESULTADOS, manager=None,
                    manager_textos=None, peso_visual=PESO_VISUAL):
    """API principal de búsqueda: híbrida por defecto.

    Para la búsqueda clásica solo visual usar buscar_solo_visual().
    """
    return buscar_imagenes_hibrida(
        descripcion, k=k, manager_visual=manager,
        manager_textos=manager_textos, peso_visual=peso_visual,
    )
