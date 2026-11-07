# -*- coding: utf-8 -*-
"""
comparar_metodos.py
-------------------
Prueba comparativa del método de búsqueda ANTIGUO (solo visual) frente
al NUEVO (híbrido visual + textual), con consultas conceptuales en
español de dificultad creciente.

Muestra, para cada consulta:
    - top-3 del método solo visual con sus similitudes (coseno).
    - top-3 del método híbrido con sus componentes (visual, textual) y
      la puntuación combinada.
    - la comparación (coincidencias entre ambos top-3, cambio del 1º).

Uso (desde la carpeta BuscaImagen):
    python comparar_metodos.py
"""

import os
import re
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import NUM_RESULTADOS, PESO_VISUAL
from src.search.image_search import (
    buscar_imagenes_hibrida,
    buscar_solo_visual,
    obtener_gestor,
    obtener_gestor_textos,
)

# Consultas obligatorias de la prueba comparativa (paráfrasis conceptuales
# del catálogo real; cada una tiene un objetivo claro en data/catalogo.json).
CONSULTAS = [
    ("Componente que emite luz",                              "led_1 / diodo_1"),
    ("Instrumento para medir voltaje",                        "voltimetro_1 / multimetro_1"),
    ("Objeto utilizado para observar microorganismos",        "microscopio_1"),
    ("Dispositivo para introducir texto en una computadora",  "teclado_1"),
    ("Instrumento para medir temperatura",                    "termometro_1"),
    ("Componente que almacena energía eléctrica",             "condensador_1 / bateria_1"),
    ("Objeto utilizado para realizar cálculos",               "calculadora_1"),
    ("Instrumento para medir corriente eléctrica",            "amperimetro_1 / multimetro_1"),
    ("Dispositivo para capturar fotografías",                 "camara_1"),
    ("Objeto que permite escuchar música",                    "auriculares_1"),
]


def formato_puntuacion(fila):
    if "puntuacion" in fila:
        return (f" [{fila['puntuacion']:.4f}: v={fila['similitud_visual']:.4f}"
                f" t={fila['similitud_textual']:.4f}]")
    return f" [{fila['similitud_visual']:.4f}]"


def main():
    print("=" * 100)
    print("BuscaImagen - Comparativa: solo visual frente a híbrido "
          f"(peso visual = {PESO_VISUAL})")
    print("=" * 100)

    manager_visual = obtener_gestor()
    try:
        manager_textos = obtener_gestor_textos()
    except RuntimeError as e:
        print("ATENCIÓN:", e)
        sys.exit(1)

    filas_salida = []
    for consulta, objetivo_esperado in CONSULTAS:
        vis = buscar_solo_visual(consulta, k=NUM_RESULTADOS, manager=manager_visual)
        hib = buscar_imagenes_hibrida(
            consulta, k=NUM_RESULTADOS,
            manager_visual=manager_visual, manager_textos=manager_textos,
            peso_visual=PESO_VISUAL,
        )
        ids_vis = [f["id"] for f in vis]
        ids_hib = [f["id"] for f in hib]
        mismo_top1 = ids_vis and ids_hib and ids_vis[0] == ids_hib[0]
        top1_mejora = ids_hib and ids_vis and ids_hib[0] in ids_vis[1:] and ids_hib[0] != ids_vis[0]

        filas_salida.append({
            "consulta": consulta,
            "esperado": objetivo_esperado,
            "visual": ids_vis,
            "hibrido": ids_hib,
            "misma_puntuacion": None,
            "mismo_top1": mismo_top1,
            "top1_mejora": top1_mejora,
        })

        print()
        print(f"CONSULTA: {consulta}")
        print(f"  Esperado : {objetivo_esperado}")
        print("  Solo visual:")
        for f in vis:
            print(f"    - {f['id']:<24} sim={f['similitud_visual']:.4f}")
        print("  Híbrido:")
        for f in hib:
            print(f"    - {f['id']:<24} puntuación={f['puntuacion']:.4f}"
                  f" (visual={f['similitud_visual']:.4f},"
                  f" textual={f['similitud_textual']:.4f})")
        print("  Comparación:")
        print(f"    top-1 visual -> {ids_vis[0] if ids_vis else '-'} | "
              f"top-1 híbrido -> {ids_hib[0] if ids_hib else '-'}"
              f" ({'IGUAL' if mismo_top1 else 'CAMBIA'})")

    # --- Resumen ---------------------------------------------------------
    print()
    print("=" * 100)
    print("RESUMEN COMPARATIVO")
    print("=" * 100)
    print(f"{'consulta':<52} {'top-1 visual':<22} {'top-1 híbrido':<22} delta")
    for r in filas_salida:
        tv = r["visual"][0] if r["visual"] else "-"
        th = r["hibrido"][0] if r["hibrido"] else "-"
        coincide = "top-1 igual" if r["mismo_top1"] else ("top-1 cambia" if "-" not in (tv, th) else "sin resultados")
        print(f"{r['consulta'][:51]:<52} {tv:<22} {th:<22} {coincide}")
    n_igual = sum(1 for r in filas_salida if r["mismo_top1"])
    print()
    print(f"Consultas con el mismo top-1 en ambos métodos : {n_igual}/{len(CONSULTAS)}")
    print(f"Consultas donde el híbrido cambia el top-1    : "
          f"{sum(1 for r in filas_salida if not r['mismo_top1'])}/{len(CONSULTAS)}")
    print()
    print("Nota: el objetivo esperado de cada consulta está indicado en la")
    print("columna 'Esperado'. Un método es mejor cuanto más arriba sitúe")
    print("los objetivos reales; la métrica completa por consulta está en el")
    print("detalle de arriba (scores de cada modalidad).")


if __name__ == "__main__":
    main()
