# Documentación técnica — BuscaImagen

Documento de referencia para desarrollo y mantenimiento. Complementa al
`README.md` (instalación, uso y evaluación resumida). Última actualización:
fase de búsqueda híbrida visual + textual.

## 1. Arquitectura del sistema

```
┌──────────────┐   get_image_features    ┌─────────────────────────────┐
│ 25 imágenes  │ ──────────────────────► │ data/embeddings_imagenes.npz│
│ catalogo.json│   512-d, norma L2       │ + metadata_embeddings.json  │
└──────────────┘                         └──────────────┬──────────────┘
                                                        │ cargar_lancedb.py
┌──────────────┐   get_text_features                   ▼
│ 25 descrip-  │ ──────────────────────► ┌─────────────────────────────┐
│ ciones reales│   512-d, norma L2       │    LanceDB  database/       │
│ catalogo.json│ → embeddings_descrip-   │ tablas: imagenes (id, nom-  │
└──────┬───────┘   ciones.npz           │ bre, categoria, descripcion,│
       │                                │ ruta_imagen, vector) y      │
       │                                │ descripciones (id, vector)  │
       │                                └──────────────┬──────────────┘
       │                                               │ buscar_similares()
       ▼                                               ▼
   consulta ──► CLIP texto ──► src/search/image_search.py
                             visual (plantillas) + textual (texto tal cual)
                             puntuacion = 0.5·cos(q,imagen) + 0.5·cos(q,desc)
                                                │ top-3
                                                ▼
                                          app.py (Streamlit)
```

Módulos y dependencias internas:

| Módulo | Depende de | Rol |
|---|---|---|
| `src/config.py` | — | Rutas, nombre del modelo, pesos, tablas, pool |
| `src/models/clip_model.py` | config | `cargar_clip()` con `lru_cache`; autodetección cpu/cuda; `model.eval()` |
| `src/embeddings/embedding_generator.py` | config, clip_model | Embeddings visuales; soporta transformers v4 (tensor) y v5 (`pooler_output`) |
| `src/embeddings/text_embedding.py` | config, clip_model, embedding_generator (`cargar_catalogo`) | Embeddings textuales de las descripciones reales |
| `src/database/lance_manager.py` | config | Clase `LanceManager(ruta_bd=None, nombre_tabla=None)`: conecta, carga idempotente, cuenta, busca |
| `src/search/image_search.py` | config, lance_manager, text_embedding, clip_model | `buscar_solo_visual()`, `buscar_imagenes_hibrida()`, API `buscar_imagenes()` |
| `app.py` | config, image_search | Streamlit; deslizador de peso visual; errores por imagen faltante |
| `tests/test_*.py` | src | 53 pruebas pytest (escrituras de BD en rutas temporales; `database/` solo lectura) |

## 2. Flujo de generación de embeddings

### Visuales (`generar_embeddings.py` → `embedding_generator.py`)

1. Lee `data/catalogo.json` (25 entradas, orden fijo del catálogo).
2. Si `data/embeddings_imagenes.npz` ya existe, **reutiliza** los ids ya
   calculados (reanudación sin duplicados).
3. Por imagen: `catalogo.json["ruta"]` (relativa a la raíz) → PIL RGB →
   `processor(...)` → `get_image_features` → toma `pooler_output` en
   transformers v5 (objeto) o el tensor directo en v4 → norma L2.
4. Guarda `.npz` (`embeddings` (25,512) float32 + `ids`) y
   `metadata_embeddings.json` (id, nombre, categoria, descripcion, ruta —
   **mismo orden** que los vectores). Las imágenes originales solo se leen.

### Textuales (`generar_embeddings_texto.py` → `text_embedding.py`)

1. Lee el mismo catálogo y codifica el campo **`descripcion` real** de cada
   entrada (nunca textos inventados; no usa coincidencias literales: todo va
   por embeddings).
2. `calcular_embedding_texto()`: texto tal cual (sin plantillas, porque las
   descripciones ya son frases completas) → `get_text_features` →
   `pooler_output` (v5) → norma L2.
3. Guarda `data/embeddings_descripciones.npz` (`embeddings` (25,512) + `ids`
   unicode, **alineado 1:1 con el orden del catálogo**). Idempotente.
4. Mismo modelo `openai/clip-vit-base-patch32`, en CPU: ni se descarga ni se
   cambia otro modelo.

## 3. Estructura de los datos

| Archivo | Contenido | Esquema |
|---|---|---|
| `data/catalogo.json` | Catálogo definitivo | `[{id, nombre, categoria, descripcion, archivo, ruta, titulo_original, fuente, licencia, autor, url_imagen}, …]` (Wikimedia Commons) |
| `data/embeddings_imagenes.npz` | Vectores visuales | `embeddings: float32 (25,512)` (norma 1) + `ids: str[25]` |
| `data/metadata_embeddings.json` | Metadatos visuales | `{modelo, dim: 512, total: 25, entradas: [{id, nombre, categoria, descripcion, ruta}, …]}` (mismo orden que el npz) |
| `data/embeddings_descripciones.npz` | Vectores textuales | `embeddings: float32 (25,512)` (norma 1) + `ids: str[25]` (orden del catálogo) |

Los 25 ids son `*_1` y las rutas apuntan a `data/imagenes/{categoria}/`.

## 4. Esquema de LanceDB

Base local: `database/` (LanceDB 0.41.0, sin servidor). Dos tablas:

**`imagenes`** (una fila por imagen):

| Columna | Tipo | Contenido |
|---|---|---|
| id | str (`*`) | `balanza_1`, `microscopio_1`, … |
| nombre | str | nombre corto (`microscopio`) |
| categoria | str | `instrumentos` / `componentes` / `cotidianos` |
| descripcion | str | descripción real del catálogo |
| ruta_imagen | str | `data/imagenes/<categoria>/<id>.jpg` |
| vector | float32[512] | embedding visual, norma 1 |

**`descripciones`** (una fila por descripción): `id` (str, mismo id que la
tabla de imágenes) + `vector` (float32[512], norma 1).

Nota clave de versión: en lancedb 0.41, `_distance` de `tabla.search()`
es la **L2 al cuadrado**. Con vectores unitarios `d² = 2 − 2·cos`, de modo
que `cos = 1 − _distance/2`. La clase `LanceManager.buscar_similares()`
devuelve la distancia sin tocar; las conversiones a coseno están
centralizadas en `src/search/image_search._similitud_desde_distancia()`.

`LanceManager` admite `nombre_tabla` parametrizado (por defecto
`imagenes`) y cambió el select de metadatos en `buscar_similares()`
(`campos=None` usa los 5 campos de imagen; para la tabla de descripciones
se pasa `campos=["id"]`).

## 5. Algoritmo de búsqueda híbrida y fórmula

`src/search/image_search.py`:

1. **Consulta vacía** (o solo espacios) → `[]` inmediato (ambos métodos).
2. **Embedding de consulta (visual)**: plantillas `"{q}"`, `"a photo of {q}"`,
   `"imagen de {q}"` (elegidas empíricamente en la fase de diagnóstico) →
   `padding=True, truncation=True` → normalizar cada plantilla → promedio →
   renormalizar.
3. **Embedding de consulta (textual)**: texto tal cual, una sola frase →
   normalizado (función `calcular_embedding_texto()`).
4. **Candidatos**: `tabla.search(vec).limit(pool)` en cada tabla, con
   `pool = min(CANDIDATOS_POOL, nº de registros de la tabla)`. Con el
   valor por defecto de `CANDIDATOS_POOL` (25) el pool cubre toda la
   tabla: cada modalidad puntúa con su coseno real y no existen
   "ausencias". La similitud se convierte a coseno con
   `1 − _distance/2`.
5. **Combinación ponderada** sobre la unión de ids:

```
puntuacion(u) = PESO_VISUAL · sim_visual(u) + PESO_TEXTUAL · sim_textual(u)
```

   Todo coseno; candidatos ausentes de una modalidad puntúan 0.0 en ella
   (situación que con el pool por defecto —tabla completa— no ocurre;
   nunca se combinan distancias brutas de escalas distintas).
6. **Orden** descendente por puntuación; top-`NUM_RESULTADOS=3`.
7. **Metadatos**: el metadato (nombre, categoría, descripción, ruta) se toma
   de la fila de la tabla de imágenes; si el id solo entró por la modalidad
   textual, se recupera con `recuperar_por_id()`; si ni siquiera existe en
   la tabla de imágenes, se descarta el candidato (defensa contra registros
   desalineados).
8. **Salida** por resultado: `id, nombre, categoria, descripcion,
   ruta_imagen, similitud_visual, similitud_textual, puntuacion, distancia`
   (distancia = L2² visual, o `None` si no entró por esa modalidad).

Pesos por defecto en `src/config.py`: `PESO_VISUAL = 0.5`,
`PESO_TEXTUAL = 0.5`. Con `PESO_VISUAL = 1.0` el método híbrido es
matemáticamente el método clásico solo visual (comprobado por test).

Si falta la tabla de descripciones y su `.npz`, `obtener_gestor_textos()`
lanza un `RuntimeError` claro («ejecuta generar_embeddings_texto.py») que la
app muestra como error, sin colapsar.

## 6. Procedimiento de evaluación (`comparar_metodos.py`)

1. Se fijan 10 consultas obligatorias (paráfrasis conceptuales del catálogo)
   con su objetivo esperado en `CONSULTAS` (led/cámara/termómetro/…).
2. Para cada consulta se ejecutan:
   - `buscar_solo_visual(consulta, k=3)` — método clásico.
   - `buscar_imagenes_hibrida(consulta, k=3, peso_visual=0.5)` — método nuevo.
3. Se imprime el top-3 de cada método con sus puntuaciones: solo visual
   muestra `similitud_visual`; el híbrido muestra `puntuacion` desglosada
   en componentes `v=` (visual) y `t=` (textual).
4. Se marca si el top-1 de ambos métodos coincide o cambia.
5. Resumen final: nº de consultas con top-1 igual/cambiado. La comparación
   con el objetivo esperado corre a cargo del lector con la tabla del
   README (no se ocultan los fallos: p. ejemplo, «observar
   microorganismos» sigue fallando en el híbrido).

Resultados medidos (pesos 50/50, ejecutados en esta fase): híbrido top-1
correcto 6/10 frente a 4/10 del método visual; corrige «Componente que emite
luz» (LED pasa de 2º-empate a 1º con componente textual 0.9101) y «medir
corriente eléctrica» (amperímetro); no aciertan aún «medir voltaje»,
«microorganismos», «almacena energía eléctrica» y «realizar cálculos».

## 7. Estructura definitiva del proyecto

Véase el árbol completo en el README. Reglas de mantenimiento:

- **No eliminar**: `data/catalogo.json`, las imágenes definitivas, los tres
  artefactos de embeddings, `database/`, `src/`, `tests/`,
  `app.py`, `generar_embeddings.py`, `generar_embeddings_texto.py`,
  `cargar_lancedb.py`, `comparar_metodos.py`, `requirements.txt`,
  `README.md`, `.gitignore`, `docs/ARQUITECTURA.md`.
- Eliminados en la limpia de esta fase (ya no existen, comprobado que nada
  los importaba): `descargar_faltantes.py` (descarga de fotos, completada),
  `probar_busqueda.py` (demo interactiva cubierta por la app y por
  `comparar_metodos.py`), `INSTRUCCIONES_EMBEDDINGS.md` (contenido
  migrado a este documento y al README) y todas las carpetas
  `__pycache__/` y `.pytest_cache/` (regenerables, ya en `.gitignore`).
- El original de descarga del usuario (`D:\Escuela\Inteligencia Artificial
  \BuscaImagen\descargar.py`, `descriptions.json`) está **fuera** del
  proyecto y no se ha tocado.

## 8. Problemas frecuentes (resueltos durante el desarrollo)

| Problema | Causa | Solución aplicada |
|---|---|---|
| `401 Client Error / Repository Not Found` al cargar CLIP | Nombre mal escrito (`clip-vit-base-p32`) | Modelo exacto `openai/clip-vit-base-patch32` en config |
| `UnicodeEncodeError: charmap codec` en consola | Windows usa cp1252 | Salida de los scripts solo con caracteres ASCII (OK / FALLO) |
| Vectores de 38 400 dimensiones | transformers **v5**: `get_image_features`/`get_text_features` devuelven `BaseModelOutputWithPooling` | Los generadores toman `.pooler_output` (512-d); soportan v4 y v5 |
| `ValueError: expected sequence of length 9 at dim 1` | Tokenizar varias plantillas sin `padding` | `padding=True, truncation=True` en el procesado de consultas |
| `ValueError: Object arrays cannot be loaded when allow_pickle=False` | npz textual guardado con `ids` de `dtype=object` | Guardar `ids=np.array(ids)` (unicode) en `text_embedding.py` |
| El método visual empataba al ruido con paráfrasis en español («Componente que emite luz») | Torre de texto CLIP entrenada predominantemente con pares en inglés | Búsqueda híbrida con embeddings textuales de las descripciones reales |
| `_distance` parecía inconsistent | En lancedb 0.41 es la L2 al cuadrado | Conversión central `cos = 1 − _distance/2` + verificación empírica |
| Aplicación sin resultados tras limpiar tablas | Tabla o npz faltantes | `obtener_gestor()` / `obtener_gestor_textos()` reconstruyen o explican qué script ejecutar; `app.py` muestra el mensaje como `st.error` |

## 9. Verificaciones de la fase (11, en `cargar_lancedb.py`)

1–6. Tabla `imagenes`: 25 registros, vectors 512-d, metadatos asociados al
JSON, sin duplicados, persistencia al reabrir, búsqueda vectorial válida
(microscopio top-1 por vector exacto). 7–11. Tabla `descripciones`:
25 descripciones, 512-d, normas L2 = 1, sin duplicados, búsqueda textual
válida (vector de la descripción de `camara_1` → top-1 `camara_1`).
