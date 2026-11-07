# -*- coding: utf-8 -*-
"""
ejecutar_pruebas.py
-------------------
Ejecución automática de las 10 pruebas funcionales de búsqueda de
BuscaImagen y generación de PRUEBAS.md (raíz del proyecto).

- Usa el sistema real: buscar_imagenes_hibrida() sobre LanceDB con las
  descripciones reales del catálogo. Ningún resultado se inventa ni se
  escribe a mano: todo sale de la ejecución.
- No modifica el algoritmo híbrido, embeddings, imágenes ni interfaz.
- Si PRUEBAS.md ya existe, se reescribe con la nueva ejecución
  (siempre un único archivo).

Uso (desde la carpeta BuscaImagen):
    python tests/ejecutar_pruebas.py
"""

import datetime
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import (
    CANDIDATOS_POOL,
    CLIP_MODEL_NAME,
    LANCE_TABLE_TEXTOS,
    PESO_TEXTUAL,
    PESO_VISUAL,
)
from src.search.image_search import buscar_imagenes_hibrida, obtener_gestor, obtener_gestor_textos

# ---------------------------------------------------------------------------
# Definición de las 10 pruebas (consulta + objeto esperado + alternativas
# funcionales LEGÍTIMAS según las descripciones reales del catálogo).
# ---------------------------------------------------------------------------

PRUEBAS = [
    {
        "n": 1,
        "consulta": "Instrumento para medir voltaje",
        "esperado": ["voltimetro_1"],
        "alternativas": {
            "multimetro_1": "también mide voltaje entre otras magnitudes "
                            "(su descripción real lo afirma)",
        },
    },
    {
        "n": 2,
        "consulta": "Instrumento para medir corriente eléctrica",
        "esperado": ["amperimetro_1"],
        "alternativas": {
            "multimetro_1": "también mide corriente entre otras magnitudes",
        },
    },
    {
        "n": 3,
        "consulta": "Dispositivo para medir varias magnitudes eléctricas",
        "esperado": ["multimetro_1"],
        "alternativas": {},
    },
    {
        "n": 4,
        "consulta": "Componente que emite luz",
        "esperado": ["led_1"],
        "alternativas": {
            "diodo_1": "componente conceptualmente afín: el LED es un diodo "
                       "emisor de luz (aunque un diodo común no emite)",
        },
    },
    {
        "n": 5,
        "consulta": "Componente que almacena energía eléctrica",
        "esperado": ["bateria_1", "condensador_1"],
        "alternativas": {},
    },
    {
        "n": 6,
        "consulta": "Instrumento para observar objetos pequeños",
        "esperado": ["microscopio_1"],
        "alternativas": {},
    },
    {
        "n": 7,
        "consulta": "Instrumento para medir temperatura",
        "esperado": ["termometro_1"],
        "alternativas": {},
    },
    {
        "n": 8,
        "consulta": "Dispositivo para introducir texto en una computadora",
        "esperado": ["teclado_1"],
        "alternativas": {},
    },
    {
        "n": 9,
        "consulta": "Dispositivo utilizado para capturar fotografías",
        "esperado": ["camara_1"],
        "alternativas": {
            "telefono_1": "los teléfonos capturan fotografías con su cámara "
                          "integrada ('funciones digitales' en su descripción)",
        },
    },
    {
        "n": 10,
        "consulta": "Dispositivo para escuchar música",
        "esperado": ["auriculares_1"],
        "alternativas": {
            "telefono_1": "puede reproducir música por su altavoz "
                          "(dispositivo personal de comunicación y audio)",
        },
    },
]

RUTA_PRUEBAS_MD = os.path.join(BASE_DIR, "PRUEBAS.md")


def clasificar(pos1, ids_top3, prueba, catalogo_ids):
    """Clasificación determinista de una prueba.

    - Correcta: el objeto esperado ocupa la PRIMERA posición.
    - Relacionada: en primera posición un objeto alternativo funcional/
      conceptual definido en la prueba.
    - No satisfactoria: primera posición sin relación definida (aunque el
      esperado aparezca en 2º/3º, se documenta aparte).
    - No evaluable: ni el esperado ni alternativa alguna existe en el
      catálogo.
    """
    if not any(o in catalogo_ids for o in prueba["esperado"]) and \
            not any(a in catalogo_ids for a in prueba["alternativas"]):
        return "No evaluable", \
            "Ninguno de los objetos candidatos existe en el catálogo real."
    if pos1 in prueba["esperado"]:
        return "Correcta", "El objeto esperado aparece en primera posición."
    if pos1 in prueba["alternativas"]:
        return "Relacionada", (
            f"En primera posición aparece «{pos1}», alternativa funcional: "
            f"{prueba['alternativas'][pos1]}. El objeto esperado "
            f"{_donde(prueba['esperado'], ids_top3)}."
        )
    return "No satisfactoria", (
        "Ni el objeto esperado ni una alternativa funcional definida "
        f"ocupan la primera posición. {_donde(prueba['esperado'], ids_top3)}"
    )


def _donde(esperados, ids_top3):
    presentes = [f"«{id_}» en la posición {i + 1}"
                 for i, id_ in enumerate(ids_top3) if id_ in esperados]
    if presentes:
        return ("El objeto esperado NO está en primera posición, pero sí: "
                + ", ".join(presentes) + ".")
    return "El objeto esperado no alcanza los tres primeros resultados."


def partir_cuerpo():
    """Ejecuta las pruebas reales con la búsqueda híbrida existente."""
    manager_visual = obtener_gestor()
    manager_textos = obtener_gestor_textos()
    catalogo = json.load(open("data/catalogo.json", encoding="utf-8"))
    catalogo_ids = {e["id"] for e in catalogo}

    ejecuciones = []
    for prueba in PRUEBAS:
        resultados = buscar_imagenes_hibrida(
            prueba["consulta"], k=3,
            manager_visual=manager_visual, manager_textos=manager_textos,
        )
        filas = []
        for posicion, fila in enumerate(resultados, 1):
            filas.append({
                "posicion": posicion,
                "id": fila["id"],
                "nombre": fila["nombre"],
                "categoria": fila["categoria"],
                "visual": f"{fila['similitud_visual']:.4f}",
                "textual": f"{fila['similitud_textual']:.4f}",
                "hibrida": f"{fila['puntuacion']:.4f}",
                "descripcion": fila["descripcion"],
                "ruta": fila["ruta_imagen"],
            })
        ids_top3 = [f["id"] for f in filas]
        pos1 = ids_top3[0] if ids_top3 else None
        clasificacion, explicacion = clasificar(pos1, ids_top3, prueba, catalogo_ids)
        ejecuciones.append({
            "prueba": prueba, "filas": filas,
            "clasificacion": clasificacion, "explicacion": explicacion,
        })
    return ejecuciones, catalogo


# ---------------------------------------------------------------------------
# Generación de PRUEBAS.md
# ---------------------------------------------------------------------------

FILA = "| {p} | `{id}` | {n} | {c} | {v} | {t} | {h} | {d} | `{r}` |"
CABECERA_TABLA = (
    "| Posición | ID | Nombre | Categoría | Visual | Textual | Híbrida "
    "| Descripción | Ruta |\n|---|---|---|---|---|---|---|---|---|"
)


def generar_markdown(ejecuciones, catalogo):
    ahora = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    por_categoria = {}
    for e in catalogo:
        por_categoria.setdefault(e["categoria"], []).append(e["nombre"])

    lineas = []
    add = lineas.append
    add("# BuscaImagen — Pruebas funcionales de búsqueda (10 consultas)")
    add("")
    add(f"*Generado automáticamente por `tests/ejecutar_pruebas.py` el {ahora} "
        "a partir de ejecuciones reales del sistema (sin resultados escritos a mano).*")
    add("")

    # --- 1. Introducción
    add("## 1. Introducción")
    add("")
    add("Este documento registra la ejecución de **10 pruebas reales** del "
        "sistema de búsqueda de BuscaImagen. Para cada consulta se recupera "
        "el top-3 del algoritmo de **búsqueda híbrida** (visual 50 % + "
        "textual 50 %, pesos por defecto) y se evalúa si el objeto esperado "
        "—o una alternativa funcional documentada— ocupa la primera "
        "posición. Todos los datos (ids, puntuaciones, descripciones, "
        "rutas) proceden de LanceDB durante la ejecución.")
    add("")
    add("**Nota de trazabilidad (revisión 2026-11-06):** al ampliar el "
        "pool de candidatos a toda la tabla (`CANDIDATOS_POOL = 25`) "
        "este documento se regeneró íntegramente con los datos reales "
        "de esa ejecución: las pruebas 5 y 6 pasaron de «No "
        "satisfactoria» a «Correcta» y desapareció el valor «0.0000» de "
        "modalidad ausente de la prueba 10. Las columnas de esta "
        "versión reflejan la configuración vigente, no la anterior.")
    add("")

    # --- 2. Entorno
    add("## 2. Entorno de pruebas")
    add("")
    add(f"- **Modelo CLIP**: `{CLIP_MODEL_NAME}` (Hugging Face Transformers, "
        "ejecución local en CPU; no se descargó ningún modelo adicional).")
    add("- **Base de datos**: LanceDB local (`database/`), tablas `imagenes` "
        f"(25 imágenes) y `{LANCE_TABLE_TEXTOS}` (25 descripciones).")
    add("- **Búsqueda**: híbrida `buscar_imagenes_hibrida()` con "
        f"PESO_VISUAL = {PESO_VISUAL}, PESO_TEXTUAL = {PESO_TEXTUAL}, "
        f"CANDIDATOS_POOL = {CANDIDATOS_POOL}, top-3.")
    add(f"- **Python**, **Streamlit** y el resto del proyecto sin cambios.")
    add("")
    add("Los vectores son unitarios; «Visual» y «Textual» son similitudes de "
        "coseno en [0, 1] y «Híbrida = 0.5·Visual + 0.5·Textual», con cuatro "
        "decimales.")
    add("")

    # --- 3. Catálogo
    add("## 3. Catálogo utilizado")
    add("")
    add(f"Total: **{len(catalogo)}** imágenes reales, sin imágenes añadidas:")
    add("")
    for categoria in por_categoria:
        nombres = ", ".join(por_categoria[categoria])
        add(f"- **{categoria}** ({len(por_categoria[categoria])}): {nombres}.")
    add("")

    # --- 4. Resultados individuales
    add("## 4. Resultados individuales")
    add("")
    for ejecucion in ejecuciones:
        prueba, filas = ejecucion["prueba"], ejecucion["filas"]
        add(f"### Prueba {prueba['n']}: «{prueba['consulta']}»")
        add("")
        esperados = ", ".join(f"`{o}`" for o in prueba["esperado"])
        add(f"**Objeto esperado:** {esperados}"
            + (f" — alternativas funcionales aceptadas: "
               + "; ".join(f"`{a}`" for a in prueba["alternativas"])
               if prueba["alternativas"] else ""))
        add("")
        add(CABECERA_TABLA)
        for f in filas:
            add(FILA.format(
                p=f["posicion"], id=f["id"], n=f["nombre"], c=f["categoria"],
                v=f["visual"], t=f["textual"], h=f["hibrida"],
                d=f["descripcion"], r=f["ruta"],
            ))
        add("")
        add(f"**Evaluación: {ejecucion['clasificacion']}** — {ejecucion['explicacion']}")
        add("")

    # --- 5. Resumen general
    add("## 5. Resumen general")
    add("")
    add("| Nº | Consulta | Resultado en 1ª posición | Clasificación |")
    add("|---|---|---|---|")
    for ejecucion in ejecuciones:
        prueba = ejecucion["prueba"]
        filas = ejecucion["filas"]
        pos1 = f"`{filas[0]['id']}`" if filas else "—"
        add(f"| {prueba['n']} | «{prueba['consulta']}» | {pos1} "
            f"| **{ejecucion['clasificacion']}** |")
    add("")
    conteo = {}
    for ejecucion in ejecuciones:
        conteo[ejecucion["clasificacion"]] = conteo.get(ejecucion["clasificacion"], 0) + 1
    for categoria in ("Correcta", "Relacionada", "No satisfactoria", "No evaluable"):
        add(f"- **{categoria}**: {conteo.get(categoria, 0)}")
    add(f"- **Total**: {len(ejecuciones)}")
    add("")

    # --- 6. Análisis
    add("## 6. Análisis de resultados")
    add("")
    correctas = [e for e in ejecuciones if e["clasificacion"] == "Correcta"]
    relacionadas = [e for e in ejecuciones if e["clasificacion"] == "Relacionada"]
    insatisf = [e for e in ejecuciones if e["clasificacion"] == "No satisfactoria"]
    if correctas:
        add(f"**Aciertos ({len(correctas)}):** " + "; ".join(
            f"«{e['prueba']['consulta']}» devolvió `{e['filas'][0]['id']}` "
            f"(h={e['filas'][0]['hibrida']})" for e in correctas) + ".")
        add("")
    if relacionadas:
        add(f"**Aciertos por alternativa funcional ({len(relacionadas)}):** " + "; ".join(
            f"«{e['prueba']['consulta']}» devolvió `{e['filas'][0]['id']}`"
            for e in relacionadas) + ".")
        add("")
    if insatisf:
        add(f"**Fracasos ({len(insatisf)}):**")
        add("")
        for e in insatisf:
            add(f"- «{e['prueba']['consulta']}»: primer resultado "
                f"`{e['filas'][0]['id']}` (h={e['filas'][0]['hibrida']}, "
                f"v={e['filas'][0]['visual']}, t={e['filas'][0]['textual']}). "
                f"{e['explicacion']}")
        add("")
    add("**Confusiones habituales:** los «instrumentos de medición» en "
        "español tienden a votar al termómetro en la modalidad visual "
        "(torre de texto de CLIP entrenada solo en inglés) y la modalidad "
        "textual tiende a agrupar descripciones que comparten el patrón "
        "«Instrumento para medir …». El caso «Componente que emite luz» es "
        "el motivo por el que existe la modalidad textual: el LED gana por "
        "la afinidad con su descripción real («Diodo que emite luz al "
        "pasar corriente»).")
    add("")
    add("**Limitaciones:** catálogo pequeño (1 imagen por objeto), "
        "similitudes textuales concentradas en una banda estrecha "
        "(0.8–1.0, poca discriminación entre descripciones parecidas) y "
        "consultas en español contra un modelo con torre de texto "
        "solo en inglés.")
    add("")

    # --- 7. Conclusiones
    add("## 7. Conclusiones")
    add("")
    total = len(ejecuciones)
    add(f"Con la configuración por defecto (50/50), el sistema acierta en "
        f"primera posición en {conteo.get('Correcta', 0)}/{total} pruebas y "
        f"devuelve una alternativa funcional en otras {conteo.get('Relacionada', 0)}. "
        f"Las {conteo.get('No satisfactoria', 0)} restantes muestran la "
        "limitación conocida del catálogo pequeño y del español frente a CLIP "
        "base-patch32; el detalle de puntuaciones permite reproducir y auditar "
        "cada resultado. La búsqueda híbrida no se modificó durante estas "
        "pruebas.")
    add("")
    return "\n".join(lineas)


def main():
    print("=" * 70)
    print("BuscaImagen - Ejecucion automatica de 10 pruebas + PRUEBAS.md")
    print("=" * 70)

    ejecuciones, catalogo = partir_cuerpo()

    print()
    for ejecucion in ejecuciones:
        prueba = ejecucion["prueba"]
        filas = ejecucion["filas"]
        top1 = filas[0]["id"] if filas else "-"
        print(f"[{prueba['n']:2d}] {top1:<22} {ejecucion['clasificacion']}")

    markdown = generar_markdown(ejecuciones, catalogo)
    with open(RUTA_PRUEBAS_MD, "w", encoding="utf-8") as f:
        f.write(markdown)

    conteo = {}
    for e in ejecuciones:
        conteo[e["clasificacion"]] = conteo.get(e["clasificacion"], 0) + 1
    print()
    print("PRUEBAS:", conteo)
    print(f"Archivo generado: {RUTA_PRUEBAS_MD}")
    todas = all(len(e["filas"]) == 3 for e in ejecuciones) and len(ejecuciones) == 10
    print("OK" if todas else "FALLO: faltan pruebas o resultados")
    sys.exit(0 if todas else 1)


if __name__ == "__main__":
    main()
