# -*- coding: utf-8 -*-
"""
respuesta_generador.py
----------------------
Generación de la «Respuesta basada en los resultados»: un resumen breve
y coherente construido EXCLUSIVAMENTE con las descripciones almacenadas
en LanceDB y asociadas a los tres resultados de la búsqueda híbrida.

Es procesamiento de texto determinista (extracción/recomposición):
- no usa ningún modelo generativo,
- no usa APIs externas ni servidores,
- no inventa características: las descripciones aparecen tal cual,
  salvo dos limpiezas controladas (punto final y nombre repetido),
- no modifica la búsqueda (visual, textual ni híbrida).

Lógica de composición
---------------------
1. Sin resultados o sin descripciones utilizables → None (la interfaz
   indicará que no hay información suficiente).
2. Descripciones duplicadas (misma frase en varios resultados): se cita
   cada frase una sola vez y se informa el nº de coincidencias
   adicionales, sin añadir contenido.
3. Redundancia controlada: si la descripción empieza repitiendo el
   nombre del objeto (p. ej. nombre "diodo" + "Diodo que emite luz …"),
   se elimina esa repetición inicial (comparación sin mayúsculas ni
   tildes). La primera palabra restante pasa a minúscula (las
   descripciones del catálogo no contienen nombres propios).
4. Apertura según las categorías REALES de los resultados (una única
   categoría o la lista mezclada); nunca se deduce que es).
5. Cuerpo: «nombre (descripción limpiada)» unidos por «; » y con «y»
   delante del último. La respuesta queda ligada a la consulta del
   usuario («Los resultados encontrados para «consulta» …»).
"""

import re
import unicodedata

LIMITE_CONSULTA = 100


def _sin_tildes_mayusculas(texto):
    """Minúsculas sin tildes ni diéresis (para comparar, no para mostrar)."""
    texto = unicodedata.normalize("NFKD", texto.strip().lower())
    return "".join(c for c in texto if not unicodedata.combining(c))


def _mapear_normalizado(texto):
    """Normaliza texto letra a letra conservando su posición original.

    Returns:
        (normalizado, indices): cadena sin mayúsculas/tildes y la lista
        indices tal que indices[i] es la posición en `texto` del carácter
        (no combinante) número i de `normalizado`.
    """
    normalizado, indices = [], []
    for posicion, caracter in enumerate(texto):
        for c in unicodedata.normalize("NFKD", caracter):
            if not unicodedata.combining(c):
                normalizado.append(c.lower())
                indices.append(posicion)
    return "".join(normalizado), indices


def _quitar_nombre_inicial(nombre, descripcion):
    """Quita el nombre si la descripción empieza repitiéndolo.

    Ej.: nombre "diodo" + "Diodo que emite luz al pasar corriente"
    → "que emite luz al pasar corriente". La comparación es sin
    mayúsculas ni tildes y exige que el nombre ocupe el inicio completo
    (después solo puede seguir espacio). Si no coincide, se devuelve la
    descripción intacta.
    """
    texto = descripcion.strip()
    palabras = [p for p in re.split(r"[\s_]+", _sin_tildes_mayusculas(nombre)) if p]
    if not palabras:
        return texto
    patron = r"^[\s_]*" + r"[\s_]+".join(re.escape(p) for p in palabras)
    normalizado, indices = _mapear_normalizado(texto)
    coincidencia = re.match(patron, normalizado)
    if coincidencia is None:
        return texto
    fin_normalizado = coincidencia.end()
    # debe terminar en borde (fin o espacio) para no cortar p. ej.
    # nombre "led" al inicio de "led...": con la descripción empezando en
    # otra palabra nunca coincide el prefijo completo, así que es seguro.
    if fin_normalizado < len(normalizado) and normalizado[fin_normalizado] not in (" ",):
        return texto
    if fin_normalizado >= len(indices):
        return ""  # la descripción era solo el nombre repetido
    indice_original = indices[fin_normalizado]
    return texto[indice_original:].lstrip(" :,")


def _limpiar_descripcion(descripcion):
    """Limpiezas seguras: punto final y minúscula inicial."""
    texto = descripcion.strip().rstrip(".").strip()
    if texto and texto[0].isupper() and len(texto) > 1 and texto[1].islower():
        texto = texto[0].lower() + texto[1:]
    return texto


def generar_respuesta(consulta, resultados):
    """Construye la respuesta breve a partir de los resultados híbridos.

    Args:
        consulta: texto de la consulta del usuario.
        resultados: lista de dicts devuelta por buscar_imagenes_hibrida()
            (cada fila contiene nombre, categoria y descripcion de LanceDB).

    Returns:
        str con la respuesta, o None si no hay resultados o las
        descripciones no aportan información suficiente.
    """
    if not consulta or not str(consulta).strip():
        return None
    if not resultados:
        return None

    # --- 1. Recoger descripciones reales (ya vienen de LanceDB)
    pares = []
    for fila in resultados:
        nombre = str(fila.get("nombre", "")).strip()
        descripcion = fila.get("descripcion")
        descripcion = str(descripcion).strip() if descripcion else ""
        if nombre and descripcion:
            pares.append((nombre, descripcion))
    if not pares:
        return None  # información insuficiente

    # --- 2. Descripciones duplicadas: una frase se cita una sola vez
    omitidas_extra = 0
    unicos, vistas = [], set()
    for nombre, descripcion in pares:
        clave = _sin_tildes_mayusculas(descripcion)
        if clave in vistas:
            omitidas_extra += 1
            continue
        vistas.add(clave)
        unicos.append((nombre, descripcion))

    # --- 3. Limpieza de cada mención
    menciones = []
    for nombre, descripcion in unicos:
        texto = _limpiar_descripcion(descripcion)
        texto = _quitar_nombre_inicial(nombre, texto)
        texto = _limpiar_descripcion(texto)  # por si quedó punto tras el recorte
        if texto:
            texto = texto[0].lower() + texto[1:]
            menciones.append(f"{nombre} ({texto})")
        else:
            menciones.append(nombre)

    # --- 4. Apertura según las categorías reales de los resultados
    consulta_limpia = str(consulta).strip()
    if len(consulta_limpia) > LIMITE_CONSULTA:
        consulta_limpia = consulta_limpia[:LIMITE_CONSULTA].rstrip() + "…"
    categorias = []
    for fila in resultados:
        categoria = str(fila.get("categoria", "")).strip()
        if categoria and categoria not in categorias:
            categorias.append(categoria)
    if len(categorias) == 1:
        tipo = f"objetos de la categoría «{categorias[0]}»"
    else:
        tipo = "objetos de las categorías " + ", ".join(
            f"«{c}»" for c in categorias
        )

    cuerpo = "; ".join(menciones[:-1])
    cuerpo = f"{cuerpo} y {menciones[-1]}" if len(menciones) > 1 else menciones[0]

    respuesta = (
        f"Los resultados encontrados para «{consulta_limpia}» corresponden "
        f"a {tipo}: {cuerpo}."
    )

    # --- 5. Coincidencias omitidas por duplicidad (sin inventar)
    if omitidas_extra:
        respuesta += (
            f" Además, {omitidas_extra} resultado(s) adicional(es) "
            "repiten descripciones ya mencionadas."
        )
    return respuesta
