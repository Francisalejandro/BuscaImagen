# -*- coding: utf-8 -*-
"""
generar_embeddings_texto.py
---------------------------
Script de la fase de búsqueda híbrida: genera los embeddings textuales
de las 25 descripciones de data/catalogo.json con la torre de texto de
CLIP (mismo modelo openai/clip-vit-base-patch32, ejecución local en
CPU, sin métodos de pago ni contadores de idioma).

Uso (desde la carpeta BuscaImagen):
    python generar_embeddings_texto.py

Salidas:
    data/embeddings_descripciones.npz  (embeddings + ids, orden del catálogo)

Es idempotente: si ya existen, se reutilizan y solo se añaden los que
falten (nunca duplica vectores).
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.embeddings.text_embedding import generar_embeddings_textos


def main():
    print("=" * 60)
    print("BuscaImagen - Embeddings textuales de las descripciones")
    print("=" * 60)

    resumen = generar_embeddings_textos(mostrar_progreso=True)

    print()
    print("RESUMEN")
    print(f"Modelo          : {resumen['modelo']}")
    print(f"Dimensiones     : {resumen['dim']}")
    print(f"Total catalogo  : {resumen['total']}")
    print(f"Calculados ahora: {resumen['ok']}")
    print(f"Reutilizados    : {resumen['reutilizados']}")
    print(f"Con fallo       : {resumen['fallo']}")
    print(f"Guardado en     : {resumen['npz_path']}")

    if resumen["errores"]:
        print()
        print("DESCRIPCIONES CON FALLO:")
        for e in resumen["errores"]:
            print(f"  - {e['id']}: {e['error']}")
        sys.exit(1)

    # Verificación básica del resultado
    import numpy as np

    with np.load(resumen["npz_path"]) as datos:
        n, dim = datos["embeddings"].shape
        normas = np.linalg.norm(datos["embeddings"], axis=1)
    print()
    print("VERIFICACION")
    print(f"Forma guardada  : ({n}, {dim})  -> {'OK' if n == 25 and dim == 512 else 'FALLO'}")
    print(f"Norma min/max   : {normas.min():.6f} / {normas.max():.6f}  "
          f"-> {'OK' if abs(normas.max() - 1.0) < 1e-4 else 'FALLO'}")
    print("OK Embeddings textuales listos. Ejecuta cargar_lancedb.py para")
    print("  integrarlos en la tabla 'descripciones' de LanceDB.")
    sys.exit(0)


if __name__ == "__main__":
    main()
