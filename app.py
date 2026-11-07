# -*- coding: utf-8 -*-
"""
app.py
------
Punto de entrada de la aplicación Streamlit de BuscaImagen.

Uso:
    python -m streamlit run app.py

La aplicación permite escribir una descripción en lenguaje natural y
muestra las 3 imágenes más relevantes según la BÚSQUEDA HÍBRIDA:

    puntuacion = peso_visual * sim_visual + peso_textual * sim_textual

- sim_visual   : coseno consulta ↔ embedding de la imagen (tabla imagenes).
- sim_textual  : coseno consulta ↔ embedding de la descripción del
                 catálogo (tabla descripciones).
El peso visual se puede ajustar desde el panel lateral. Todo el
procesamiento es local (CLIP + LanceDB, sin APIs de pago).
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import streamlit as st

from src.config import BASE_DIR as RUTA_BASE, NUM_RESULTADOS, PESO_VISUAL
from src.search.image_search import buscar_imagenes_hibrida, obtener_gestor
from src.search.respuesta_generador import generar_respuesta

# ---------------------------------------------------------------------------
# Configuración de la página
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="BuscaImagen",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 BuscaImagen")
st.caption(
    "Búsqueda de imágenes con descripciones en lenguaje natural mediante "
    "búsqueda híbrida (embedding visual + embedding textual, CLIP + LanceDB, "
    "procesamiento 100 % local)."
)


# ---------------------------------------------------------------------------
# Recursos en caché (el modelo CLIP y la base de datos se cargan una sola vez)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Cargando CLIP y la base de datos (la primera vez tarda más)...")
def recursos():
    manager_visual = obtener_gestor()
    return manager_visual


try:
    MGR = recursos()
    N = MGR.contar_registros()
except RuntimeError as e:
    st.error(f"No se pudo preparar la base de datos: {e}")
    st.stop()

st.info(f"Índice listo: **{N}** imágenes indexadas en LanceDB.", icon="✅")

# ---------------------------------------------------------------------------
# Panel lateral: pesos de la búsqueda híbrida e información
# ---------------------------------------------------------------------------

st.sidebar.header("Peso de la búsqueda híbrida")
peso_visual = st.sidebar.slider(
    "Peso visual (el textual es el complemento)",
    min_value=0.0,
    max_value=1.0,
    value=float(PESO_VISUAL),
    step=0.05,
)
st.sidebar.caption(
    f"puntuación = {peso_visual:.2f} · similitud_visual "
    f"+ {1.0 - peso_visual:.2f} · similitud_textual\n\n"
    "- **1.00** = solo visual (comportamiento clásico)\n"
    "- **0.50** = híbrido equilibrado (recomendado)\n"
    "- **0.00** = solo textual"
)

st.sidebar.header("Acerca de")
st.sidebar.markdown(
    """
    **BuscaImagen** busca las 3 imágenes más relacionadas con tu
    descripción combinando:

    - **Modalidad visual** (tabla `imagenes`): consulta ↔ imagen CLIP.
    - **Modalidad textual** (tabla `descripciones`): consulta ↔ descripción
      real del catálogo.

    **CLIP** `openai/clip-vit-base-patch32` · **LanceDB** local
    (`database/`) · **Streamlit** como interfaz.

    *Procesamiento 100 % local, sin APIs de pago.*
    """
)

# ---------------------------------------------------------------------------
# Interfaz de búsqueda
# ---------------------------------------------------------------------------

consulta = st.text_input(
    "Describe la imagen que buscas",
    placeholder="Ej.: un microscopio · Componente que emite luz · "
                "Dispositivo para capturar fotografías",
)

col_izq, col_der = st.columns([1, 4])
with col_izq:
    boton = st.button("Buscar", type="primary", use_container_width=True)

if boton and consulta.strip():
    try:
        with st.spinner("Buscando imágenes similares (híbrido visual + textual)..."):
            resultados = buscar_imagenes_hibrida(
                consulta, k=NUM_RESULTADOS, manager_visual=MGR,
                peso_visual=peso_visual,
            )
    except RuntimeError as e:
        st.error(f"Error de búsqueda: {e}")
        resultados = []

    if not resultados:
        st.warning("No se encontraron resultados para esa descripción.")
    else:
        st.subheader(f"Top {len(resultados)} resultados para “{consulta}”")
        columnas = st.columns(len(resultados), gap="medium")
        for posicion, (columna, fila) in enumerate(zip(columnas, resultados), 1):
            ruta_img = (RUTA_BASE / fila["ruta_imagen"]).resolve()
            with columna:
                st.markdown(f"**#{posicion}**")
                if os.path.exists(ruta_img):
                    st.image(str(ruta_img), use_container_width=True)
                else:
                    st.error(f"Imagen no encontrada: {fila['ruta_imagen']}")
                st.markdown(f"**{fila['nombre']}**")
                st.markdown(
                    f"`{fila['categoria']}` · puntuación "
                    f"**{fila['puntuacion']:.4f}**"
                )
                st.progress(fila["puntuacion"], text="puntuación híbrida")
                st.caption(
                    f"visual: {fila['similitud_visual']:.4f} · "
                    f"textual: {fila['similitud_textual']:.4f}"
                )
                st.write(fila["descripcion"])
                st.caption(f"Ruta: {fila['ruta_imagen']}")

        # ---------------------------------------------------------------
        # Respuesta basada en los resultados (extractiva, sin IA generativa)
        # ---------------------------------------------------------------
        respuesta = generar_respuesta(consulta, resultados)
        st.subheader("Respuesta basada en los resultados")
        if respuesta:
            st.info(respuesta, icon="💬")
        else:
            st.warning(
                "No hay información suficiente en las descripciones "
                "recuperadas para elaborar una respuesta."
            )
elif boton:
    st.warning("Escribe una descripción antes de buscar.")
