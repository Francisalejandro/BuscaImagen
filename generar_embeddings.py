import json
import os
import sys

# Comprobar que se ejecuta desde la raíz del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.embeddings.embedding_generator import (
    generar_embeddings_imagenes,
    guardar_embeddings,
)


def main():
    print("=" * 60)
    print("BuscaImagen - Generación de embeddings de imágenes (CLIP)")
    print("=" * 60)

    resumen = generar_embeddings_imagenes(mostrar_progreso=True)

    if not resumen["ids"]:
        print(f"\nERROR No se procesó ninguna imagen ({resumen['fallo']} fallos). "
              "No se han guardado archivos.")
        sys.exit(1)

    guardar_embeddings(resumen)

    print()
    print("=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"Imágenes del catálogo : {resumen['total']}")
    print(f"Procesadas ahora      : {resumen['ok']}")
    print(f"Reutilizadas (cache)  : {resumen['reutilizados']}")
    print(f"Fallidas              : {resumen['fallo']}")
    for e in resumen["errores"]:
        print(f"  - {e['id']}: {e['error']}")
    print(f"Dimensión de vectores : {resumen['dim']}")
    print(f"Guardado en           : {resumen['npz_path']}")
    print(f"Metadatos en          : {resumen['json_path']}")

    # --- Verificación: npz y json con el mismo orden y vectores normalizados
    print()
    print("VERIFICACIÓN")
    import numpy as np

    with np.load(resumen["npz_path"]) as data:
        vectores = data["embeddings"]
        ids_npz = [str(i) for i in data["ids"]]
    with open(resumen["json_path"], encoding="utf-8") as f:
        metadatos = json.load(f)["entradas"]
    ids_json = [e["id"] for e in metadatos]

    mismo_orden = ids_npz == ids_json
    normas = np.linalg.norm(vectores, axis=1)
    normalizados = bool(np.allclose(normas, 1.0, atol=1e-5))
    misma_dimension = vectores.shape[1] == resumen["dim"]

    print(f"ids npz == ids json (mismo orden): {mismo_orden}")
    print(f"Todos los vectores normalizados (||v||=1): {normalizados}")
    print(f"Dimensión coherente: {vectores.shape} OK" if misma_dimension
          else f"Dimensión incoherente: {vectores.shape}")
    if mismo_orden and normalizados and misma_dimension:
        print("\nOK Todo correcto: los vectores y los metadatos están "
              "alineados y listos para LanceDB.")
    else:
        print("\nERROR ERROR de correspondencia. Revisa los pasos.")
        sys.exit(1)


if __name__ == "__main__":
    main()
