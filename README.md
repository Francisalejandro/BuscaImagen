# BuscaImagen 🔎

**Búsqueda de imágenes mediante descripciones en lenguaje natural.**
Proyecto que recupera las **3 imágenes más relevantes** para una
frase escrita en español, combinando similitud **visual** y similitud
**textual** (búsqueda híbrida). Procesamiento **100 % local**: sin APIs de
pago, sin servidores externos, sin SQL.

| I | II | III |
|---|---|---|
| ¿Qué es? | Un buscador semántico de imágenes | ¿Cómo? CLIP + LanceDB + Streamlit |
| ¿Con qué datos? | 25 imágenes reales en 3 categorías | ¿Puntúa cómo? media 50 % visual + 50 % textual |

## 1. Descripción general

El usuario escribe una frase («Componente que emite luz», «Instrumento para
medir temperatura»…) y el sistema devuelve las 3 imágenes del catálogo con
mayor puntuación híbrida. Detrás:

1. Un **único modelo CLIP** transforma *imágenes y textos* en vectores
   comparables del mismo espacio de 512 dimensiones.
2. Una base de datos **vectorial local** (LanceDB) almacena y consulta esos
   vectores.
3. Una combinación ponderada de dos similitudes (visual y textual) decide el
   orden.
4. Streamlit presenta los resultados y una **respuesta extractiva** breve
   construida solo con las descripciones reales recuperadas.

## 2. Problema que aborda

Buscar una imagen exige, normalmente, saber su **nombre de archivo** o
palabras clave exactas. BuscaImagen sustituye eso por **descripciones libres**:
si el usuario no recuerda cómo se llamaba el objeto («algo que introduce texto
en una computadora») pero sí cuál es su función, la búsqueda semántica lo
encuentra. El caso motivador de este proyecto: la consulta «Componente que
emite luz» —ninguna imagen se llamaba así— se resuelve emparejando la frase
con la **descripción almacenada** del LED («Diodo que emite luz al pasar
corriente»).

## 3. Objetivos académicos

- Aplicar **representaciones multimodales** (CLIP) que unen el espacio de las
  imágenes y el del lenguaje natural.
- Gestionar una **base de datos vectorial local** con persistencia y cargas
  idempotentes (11 verificaciones obligatorias tras cada carga).
- Diseñar y **evaluar empíricamente** un algoritmo de búsqueda híbrida con
  pesos configurables, sin afirmar mejoras que los datos no demuestren.
- Producir **evidencia reproducible**: `PRUEBAS.md` se regenera desde LanceDB
  en cada ejecución y `comparar_metodos.py` compara dos métodos bajo las
  mismas condiciones.

## 4. Tecnologías utilizadas

| Componente | Tecnología | Versión verificada |
|---|---|---|
| Lenguaje | Python | 3.13.0 |
| Deep learning | PyTorch (build CPU) | 2.15.0+cpu |
| Modelo CLIP | Hugging Face Transformers | 5.22.1 |
| Modelo preentrenado | [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32) | — |
| Base vectorial | LanceDB (local, en `database/`) | 0.41.0 |
| Interfaz | Streamlit | 1.67.0 |
| Datos numéricos | NumPy / Pandas / Pillow | 2.5.4 / 3.1.1 / 13.0.0 |
| Pruebas | pytest | 9.2.0 |

## 5. Arquitectura general

```
                ┌────────────────────── CLIP (único modelo, 512-d compartido) ──────────────────┐
 Imágenes (25) ─►│ get_image_features → vector 512-d, L2 → data/embeddings_imagenes.npz         │
 Descripciones ─►│ get_text_features  → vector 512-d, L2 → data/embeddings_descripciones.npz    │
 (catalogo.json) └──────────────────────────────────────────────────────────────────────────────┘
                                                │  python cargar_lancedb.py
                                                ▼
                ┌────────────────────── LanceDB (database/, local) ─────────────────────────────┐
                │ tabla "imagenes"      (id, nombre, categoria, descripcion, ruta_imagen, vector)│
                │ tabla "descripciones" (id, vector)                                             │
                └────────────────────────────────────────────────────────────────────────────────┘
                                                │
 Consulta del usuario ──► CLIP texto ──► BÚSQUEDA HÍBRIDA (src/search/image_search.py)
   (2 embeddings:          con plantillas (visual) y tal cual (textual)
   visual + textual)       puntuacion = PESO_VISUAL·cos(consulta, imagen)
                                        + PESO_TEXTUAL·cos(consulta, descripcion)
                                                │ top-3
                                                ▼
                     Streamlit (app.py): 3 tarjetas + respuesta extractiva
                     (src/search/respuesta_generador.py)
```

### Recorrido de una consulta (paso a paso, con el módulo responsable)

| # | Paso | Módulo |
|---|---|---|
| 1 | El usuario escribe la descripción y pulsa Buscar (o ajusta el peso visual en el panel lateral) | `app.py` |
| 2 | Se cargan CLIP y LanceDB una sola vez (caché por sesión) | `app.py` → `src/models/clip_model.py`, `src/database/lance_manager.py` |
| 3 | Se generan los 2 embeddings de la consulta: visual (plantillas) y textual (texto tal cual) | `src/search/image_search.py` + `src/embeddings/text_embedding.py` |
| 4 | LanceDB devuelve el pool de candidatos por similitud en cada modalidad | `src/database/lance_manager.py` (`buscar_similares`) |
| 5 | `_distance` (L2²) se convierte a coseno y se recorta a [0, 1] | `src/search/image_search.py` (`_similitud_desde_distancia`) |
| 6 | Se combinan las puntuaciones con `p = peso_visual·v + (1−peso_visual)·t` sobre la unión de ids | `src/search/image_search.py` |
| 7 | Orden descendente y selección de los 3 primeros | `src/search/image_search.py` |
| 8 | Metadatos (nombre, categoría, descripción, ruta) de cada resultado desde la tabla `imagenes` | `src/search/image_search.py` (`_fila_hibrida` → `recuperar_por_id`) |
| 9 | Respuesta extractiva usando SOLO las descripciones de los 3 resultados | `src/search/respuesta_generador.py` |
| 10 | Presentación: tarjetas con imagen y puntuaciones + respuesta en `st.info` | `app.py` |

### Responsabilidad de cada módulo

| Módulo | Responsabilidad única |
|---|---|
| `src/config.py` | Todas las rutas (relativas a la raíz) y parámetros: modelo, tablas, `NUM_RESULTADOS=3`, `CANDIDATOS_POOL=25`, pesos |
| `src/models/clip_model.py` | Carga única y cacheada (`@lru_cache`) de CLIP y su procesador; autodetección CPU/GPU |
| `src/embeddings/embedding_generator.py` | Embeddings **visuales** de las 25 imágenes → `embeddings_imagenes.npz` + `metadata_embeddings.json` |
| `src/embeddings/text_embedding.py` | Embeddings **textuales** de las 25 descripciones → `embeddings_descripciones.npz` |
| `src/database/lance_manager.py` | Conexión, carga **idempotente** (`creada`/`sin_cambios`/`reconstruida`), búsqueda vectorial |
| `src/search/image_search.py` | Búsqueda visual, textual e híbrida; conversión distancia→coseno |
| `src/search/respuesta_generador.py` | Respuesta extractiva determinista desde las descripciones recuperadas |
| `src/utils/image_utils.py` | Utilidades auxiliares de imagen |

## 6. Funcionamiento de CLIP

**Qué es y qué problema resuelve.** CLIP (*Contrastive Language-Image
Pretraining*, Radford et al., 2021) entrena simultáneamente una **torre
visual** (ViT-B/32) y una **torre de texto** (transformer) con un objetivo
contrastivo: emparejar millones de pares imagen-frase acercando sus
representaciones y separando las no correspondientes. Resultado: un espacio
de embeddings **compartido** donde un texto y una imagen que describen lo
mismo quedan cerca aunque sean datos de naturaleza distinta.

**Puntos clave para este proyecto (todos verificados en el código):**

- **Representaciones vectoriales multimodales y espacio compartido**: las 25
  imágenes y las 25 descripciones se proyectan con el mismo
  `openai/clip-vit-base-patch32` al mismo espacio de **512 dimensiones**
  (`get_image_features` / `get_text_features`). Eso hace posible comparar una
  *consulta textual* contra una *imagen* (modalidad visual) y otra *consulta
  textual* contra otra *frase* (modalidad textual) con la misma métrica,
  sin traducciones ni coincidencia de palabras.
- **Las 512 dimensiones**: es la dimensión del embedding compartido de este
  modelo (ViT-B/32). Cada dimensión es una codificación latente aprendida
  que no es interpretable individualmente; lo relevante es la dirección del
  vector, que codifica el *significado* de la imagen o frase completa.
- **Normalización L2**: dividir un vector por su norma euclídea (‖v‖ =
  √Σxᵢ²) hasta dejarlo con longitud 1. Con vectores unitarios el producto
  escalar **es** el coseno: cos(u,v) = u·v. Para vectores unitarios
  además ‖u−v‖² = 2 − 2·cos(u,v), identidad que el proyecto usa en la
  conversión de LanceDB (§9).
- **Limitación con consultas en español** (justificada y contrastada): la torre
  de texto de CLIP se entrenó con pares imagen-texto recogidos en internet,
  **predominantemente en inglés**. No es correcto decir que «no entiende
  español»: el encoder mapea igual cualquier cadena y **en este proyecto las
  consultas en español funcionan (8/10 pruebas funcionales correctas)**; lo que
  se observa es que los **matices finos** entre paráfrasis españolas
  («medir voltaje» frente a descripciones que empiezan igual: «Instrumento
  para medir…») se pierden, como documentan las 2 pruebas «No satisfactoria»
  de `PRUEBAS.md` y las bandas de similitud medidas (§14). Por eso el híbrido
  apoya la comparación en descripciones escritas *también en español*.

La primera vez, Hugging Face descarga el modelo a la caché local del usuario
(`~/.cache/huggingface`, unos 600 MB; **requiere conexión a internet solo esa
vez**); después todo funciona desde la caché, 100 % offline y en CPU
(autodetección en `src/models/clip_model.py`).

## 7. Búsqueda visual

1. La consulta se expande con **3 plantillas** implementadas realmente en
   `PLANTILLAS_VISUALES` (`src/search/image_search.py`): `"{q}"`,
   `"a photo of {q}"`, `"imagen de {q}"` (elegidas empíricamente en la fase
   de diagnóstico con este catálogo).
2. Cada plantilla se codifica con la torre de texto, **se normaliza** (L2) y
   las tres se **promedian**; el promedio se **renormaliza**. El resultado es
   robusto frente al estilo de redacción.
3. Ese vector se compara por **coseno** contra los 25 vectores de las
   **imágenes** (tabla `imagenes`), recuperados vía LanceDB (§10).

Es la búsqueda «clásica» de las primeras fases del proyecto: fuerte con
nombres o apariencia («un microscopio», «una placa blanca con muchos
agujeros»), débil con paráfrasis conceptuales en español (§6).

## 8. Búsqueda textual

1. La consulta se usa **tal cual, sin plantillas**
   (`calcular_embedding_texto`): las descripciones del catálogo son frases
   completas en español y la comparación es dentro del propio espacio
   textual.
2. Ese vector se compara por **coseno** contra los 25 vectores de las
   **descripciones reales** del catálogo (tabla `descripciones`,
   idéntica en tamaño y orden al catálogo).

La comparación es **semántica, no literal**: no hay coincidencia de cadenas
en ningún punto del sistema.

## 9. Algoritmo de búsqueda híbrida

Implementación exacta de `buscar_imagenes_hibrida()`:

1. **Dos embeddings de la consulta** (visual con plantillas §7, textual tal
   cual §8), ambos con la misma torre de texto del mismo modelo.
2. **Pool de candidatos**: LanceDB devuelve, por cada modalidad, los
   `k = min(CANDIDATOS_POOL, nº de registros)` más próximos ordenados por
   distancia. Con el valor por defecto (`CANDIDATOS_POOL = 25`) el pool
   cubre la **tabla completa**: cada modalidad puntúa a todos los registros
   con su coseno real. Para catálogos más grandes el parámetro es
   configurable en `src/config.py`.
3. **Conversión de métrica**: LanceDB 0.41 devuelve en `_distance` la
   **distancia L2 al cuadrado** (verificado empíricamente con este catálogo y
   documentado en `docs/ARQUITECTURA.md`). Para vectores unitarios se cumple
   d² = 2 − 2·cos, de modo que `coseno = 1 − d²/2`. Esta conversión está
   centralizada en `_similitud_desde_distancia()`, que además **recorta** el
   resultado al intervalo [0, 1].
4. **Combinación ponderada** sobre la **unión** de ids de ambos pools:

   ```
   puntuacion_final = PESO_VISUAL · sim_visual + PESO_TEXTUAL · sim_textual
                    = 0.5 · cos(consulta, imagen) + 0.5 · cos(consulta, descripcion)   # por defecto
   ```
5. **Top-3**: orden descendente por puntuación; los primeros
   `NUM_RESULTADOS = 3`. El peso se puede cambiar en `src/config.py` o desde
   el deslizador del panel lateral de la app (`PESO_VISUAL`).

**Los pesos y sus límites (importante leerlo con precisión):**

- Ambas similitudes son las transformadas de la misma métrica (coseno
  convertida y recortada a [0, 1]); la media ponderada conserva esa escala y
  `PESO_VISUAL = 1.0` reproduce exactamente el método clásico solo visual y
  `0.0` sería solo textual.
- El coseno de por sí **no** vive solo en [0, 1]: su rango matemático es
  **[−1, 1]**. En este catálogo las similitudes observadas caen dentro de
  banda positiva (visual ≈ 0.19–0.29 entre consulta e imagen; textual
  ≈ 0.83–1.00 entre consulta y descripción; entre descripciones del catálogo
  el par más alejado mide ≈ 0.73), y el recorte del punto 3 garantiza el
  intervalo [0, 1] por construcción.
- La normalización L2 **no** hace que las dos modalidades tengan
  *distribuciones* equivalentes: la visual se mueve en una banda estrecha
  (0.19–0.29) y la textual en otra más alta y concentrada (0.83–1.00). Son
  escalas **comparables** (ambas coseno en [0, 1]), pero **no equivalentes
  estadísticamente**: en la práctica las diferencias textuales entre
  candidatos (≈0.1–0.2) pesan más que las visuales (≈0.02–0.05), de modo que
  con pesos 50/50 el **orden final tiende a dominarlo la textual**. Es un
  hecho medido, no un error de implementación.
- Los pesos permiten **controlar** la contribución de cada modalidad, pero **no
  son una garantía de precisión**: la evidencia de qué mejora el híbrido está
  en las ejecuciones (§17 y §18).

**Candidato ausente de una modalidad.** Si un id apareciera en un pool y no
en el otro (solo posible reduciendo `CANDIDATOS_POOL` por debajo del tamaño
de la tabla), el código le asigna `0.0` en la modalidad ausente, documentado
en el propio docstring. Con la configuración por defecto esa situación no
ocurre. Si un id del pool textual no tiene fila en la tabla de imágenes, el
resultado se descarta (`_fila_hibrida` devuelve `None`).

## 10. Papel de LanceDB

- Base de datos **vectorial, local y persistente** en `database/`: guarda las
  dos tablas y ejecuta la búsqueda por distancia (`tabla.search(...).
  limit(k)`), sin servidor, sin SQL, sin servicios externos. La decisión de
  usarla: es una base **vectorial** real que se integra en una carpeta del proyecto (un
  proyecto universitario portátil), a diferencia de un servidor de BD o un
  índice in-memory que desaparece al cerrar el proceso.
- **Esquema**: tabla `imagenes` (id, nombre, categoria, descripcion,
  ruta_imagen, vector float32 de 512) y tabla `descripciones` (id, vector).
  Están **separadas** porque tienen distinto esquema (metadatos completos
  frente a solo ids) y distinta modalidad; comparten carpeta e id común.
- **Cargas idempotentes** (`creada` / `sin_cambios` / `reconstruida`): si los
  ids coinciden no se toca nada; si difieren se reconstruye por completo
  (transacción única `create_table(mode="create")`), de modo que **nunca se
  duplican registros**. `cargar_lancedb.py` ejecuta **11 verificaciones
  obligatorias** tras la carga (§15).
- LanceDB no calcula coseno: devuelve `_distance` = **L2 al cuadrado**. La
  conversión a coseno es del proyecto (`_similitud_desde_distancia`), correcta
  **porque todos los vectores están L2-normalizados antes de almacenarse**
  (normas = 1: los generadores los comprueban al guardar y los tests
  verifican la del archivo textual; el diagnóstico del proyecto las midió
  también en la tabla de imágenes). Con vectores no normalizados esa
  fórmula NO valdría; está documentado en `docs/ARQUITECTURA.md`.

## 11. Papel de Streamlit

`app.py` es la interfaz: caja de texto + botón Buscar, resultados en 3
columnas con la imagen (verificando existencia de la ruta y mostrando un
`st.error` localizado si faltara), nombre, categoría, descripción, ruta y las
tres puntuaciones (híbrida, con barra de progreso; visual y textual con 4
decimales), deslizador en el panel lateral para el peso visual y sección
«Respuesta basada en los resultados» (`st.info`) generada de forma
extractiva. El modelo y la base se cachean con `st.cache_resource` (carga
única por sesión). La decisión de Streamlit: es la interfaz web en Python
más directa para construir paneles interactivos **sin escribir HTML/JS** y
sin servidor de aplicaciones adicional, compatible con el requisito «todo
local».

## 12. Estructura del proyecto

```
BuscaImagen/
├── data/                                 ← datos (se suben a Git)
│   ├── imagenes/
│   │   ├── instrumentos/      (10 imágenes: amperimetro, balanza, microscopio,
│   │   │                       multimetro, osciloscopio, pipeta, probeta, regla,
│   │   │                       termometro, voltimetro)
│   │   ├── componentes/       (8: bateria, circuito_integrado, condensador, diodo,
│   │   │                       led, protoboard, resistencia, transistor)
│   │   └── cotidianos/        (7: auriculares, calculadora, camara, reloj, raton,
│   │                           telefono, teclado)
│   ├── catalogo.json                  25 registros: id, nombre, categoria,
│   │                                  descripcion real, ruta relativa,
│   │                                  titulo_original, fuente, licencia, autor
│   ├── embeddings_imagenes.npz        vectores visuales (25, 512) + ids
│   ├── metadata_embeddings.json       metadatos en el mismo orden
│   └── embeddings_descripciones.npz   vectores textuales (25, 512) + ids
├── database/                          LanceDB local (GENERADO, no se sube a Git;
│                                      se reconstruye con cargar_lancedb.py)
├── src/
│   ├── config.py                      rutas y parámetros del híbrido
│   ├── models/clip_model.py           carga cacheada de CLIP (autodetección CPU/GPU)
│   ├── embeddings/
│   │   ├── embedding_generator.py     embeddings VISUALES de las imágenes
│   │   └── text_embedding.py          embeddings TEXTUALES de las descripciones
│   ├── database/lance_manager.py      LanceManager: conexión, carga idempotente, búsqueda
│   ├── search/
│   │   ├── image_search.py            búsqueda visual, textual e híbrida
│   │   └── respuesta_generador.py     respuesta extractiva
│   └── utils/image_utils.py           utilidades auxiliares de imagen
├── tests/                             53 pruebas pytest + ejecutar_pruebas.py
│   ├── test_search.py   (16)          configuración, catálogo, imports, búsqueda real
│   ├── test_lancedb.py  (9)           LanceDB en rutas temporales (nunca toca database/)
│   ├── test_hibrida.py  (14)          embeddings textuales, tabla descripciones, híbrido
│   ├── test_respuesta.py (14)         respuesta extractiva (incl. integración con la búsqueda)
│   └── ejecutar_pruebas.py            10 pruebas funcionales → regenera PRUEBAS.md
├── docs/ARQUITECTURA.md               documentación técnica detallada
├── app.py                             interfaz Streamlit
├── generar_embeddings.py              (idempotente) embeddings visuales
├── generar_embeddings_texto.py        (idempotente) embeddings textuales
├── cargar_lancedb.py                  carga ambas tablas + 11 verificaciones [--reconstruir]
├── comparar_metodos.py                comparativa solo-visual vs híbrido (10 consultas)
├── PRUEBAS.md                         registro real de las 10 pruebas funcionales
├── requirements.txt
├── .gitignore
└── README.md                          (este documento)
```

## 13. Conjunto de datos

- **25 imágenes reales**, una por objeto, en 3 categorías:
  **instrumentos** (10), **componentes** (8) y **objetos cotidianos** (7).
- Todas las imágenes proceden de **Wikimedia Commons** y `catalogo.json`
  registra por cada una `titulo_original`, `fuente` (URL de la página de la
  imagen), `licencia` (CC BY-SA, versión 3.0 o 4.0 según la imagen) y
  `autor`. Ese fichero es la autoridad sobre las condiciones de uso; las
  imágenes se incluyen en el repositorio tal como las usa el proyecto y
  **no se modifican**.
- Cada registro tiene: `id` (ej. `led_1`, `termometro_1`), `nombre`,
  `categoria`, `descripcion` (frase real, redactada como descripción
  funcional, ej. «Instrumento para medir la masa de objetos») y `ruta`
  relativa al proyecto (ej. `data/imagenes/componentes/led_1.jpg`).
- Las **rutas son relativas y portables** (se resuelven con `Path.resolve()`
  contra la raíz del proyecto en `src/config.py`); el repositorio no depende
  de unidades absolutas.

## 14. Instalación y configuración

1. **Python 3.13** y entorno virtual (recomendado):
   ```bash
   cd BuscaImagen
   python -m venv .venv
   # Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
   ```
2. Instalar la build de PyTorch **CPU** (muchísimo más ligera que la CUDA) y
   después el resto de dependencias:
   ```bash
   pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```
   (la variante `+cpu` de torch solo existe en ese índice; por eso PyTorch se
   instala aparte de `requirements.txt`).
3. **No se necesita ningún token ni clave.** En la primera ejecución Hugging
   Face descarga el modelo (~600 MB) a la caché local del usuario
   (`~/.cache/huggingface`); esa descarga **requiere internet**. A partir de
   ahí todo el sistema funciona sin conexión (comprobado: el proyecto se
   ejecuta offline salvo esa primera descarga). Quien no disponga de esa
   caché debe contar con esa descarga inicial.

## 15. Generación de embeddings y carga de la base de datos

```bash
python generar_embeddings.py          # embeddings visuales (idempotente: reanuda)
python generar_embeddings_texto.py    # embeddings textuales (idempotente: reanuda)
python cargar_lancedb.py              # carga LanceDB + 11 verificaciones
```

- Los dos generadores **reutilizan** los ids ya calculados en el `.npz`
  correspondiente (no recalculan ni duplican trabajo) y guardan vectores +
  ids en el mismo orden que `data/catalogo.json`.
- `cargar_lancedb.py` crea las tablas `imagenes` y `descripciones` de forma
  idempotente (`creada`/`sin_cambios`/`reconstruida`; `--reconstruir` fuerza
  la reconstrucción completa) y ejecuta 11 verificaciones: correspondencia
  catálogo↔imágenes, 25 registros, 512 dimensiones, norma L2 = 1, sin ids
  duplicados, persistencia tras reabrir la conexión, búsqueda visual y
  textual válidas, tablas intactas.

**Qué se incluye en GitHub y qué se genera localmente:**

| Incluido en el repositorio | Generado localmente (si falta) |
|---|---|
| 25 imágenes + `catalogo.json` | modelo CLIP en la caché de HF (descarga única con internet) |
| embeddings visuales y textuales (`.npz` + json) | `database/` (reconstruir: `python cargar_lancedb.py`) |
| todo el código, tests y documentación | `__pycache__`, `.streamlit/` etc. (excluidos por `.gitignore`) |

La base de datos **no se sube** al repositorio porque es un artefacto
determinista derivado de `data/`: si se reconstruye con `cargar_lancedb.py`
es bit a bit equivalente a la original y las 11 verificaciones lo confirman.

## 16. Ejecución de la aplicación

```bash
python -m streamlit run app.py
```

Se abre en `http://localhost:8501` (para detenerla, `Ctrl+C`). Siempre desde
la carpeta `BuscaImagen`. En el panel lateral puede ajustarse el peso visual
(1.00 = solo visual · 0.50 = híbrido equilibrado por defecto · 0.00 = solo
textual); no se necesita configurar nada más.

## 17. Ejecución de las pruebas

El proyecto distingue **tres niveles de verificación** (los resultados
siguientes son los comprobables, no una promesa):

| Tipo | Cómo | Qué cobertura |
|---|---|---|
| **Unitarias e integración** (`pytest`) | `python -m pytest tests/ -v` | **53 pruebas** — config (4), catálogo/imports (8), búsqueda real contra LanceDB (4), LanceDB propia en rutas temporales (9: escritura/lectura/persistencia/duplicados), embeddings textuales y tabla descripciones (6), híbrido (8), respuesta extractiva e integración con la búsqueda (14). Los tests que **escriben** en LanceDB usan rutas temporales (nunca modifican `database/`); los que leen la base real (`database/`) la consultan en modo solo lectura |
| **Funcionales (sistema real)** | `python tests/ejecutar_pruebas.py` | Las **10 consultas fijas** contra el híbrido real; regenera `PRUEBAS.md` (sobreescritura, nunca duplicado) |
| **Verificación de la interfaz** | puntual, con `streamlit.testing.v1.AppTest` | arranque sin excepciones, flujo de una consulta, consulta vacía, imagen faltante y error de BD (realizada en la fase de validación; no forma parte de la suite permanente) |

`PRUEBAS.md` (real, generado el 2026-11-06) registra: 10 consultas
ejecutadas con pesos 50/50 y pool = tabla completa; **8 clasificadas
Correcta** (objeto esperado en primera posición: pruebas 2, 4, 5, 6, 7, 8, 9
y 10) y **2 No satisfactoria** (prueba 1 «Instrumento para medir voltaje» →
top-1 `termometro_1` y prueba 3 «Dispositivo para medir varias magnitudes
eléctricas» → top-1 `calculadora_1`). El criterio está documentado en el
propio fichero: *Correcta* = esperado en 1ª posición; *Relacionada* =
alternativa funcional legítima en 1ª (en esta ejecución: 0); *No
satisfactoria* = ni esperado ni alternativa en 1ª posición; *No evaluable*
= sin datos suficientes (0). Cada prueba registra el top-3 completo con
puntuaciones a 4 decimales, descripciones y rutas, y el análisis explica
las 2 limitaciones (§20).

## 18. Evaluación comparativa de métodos (comparar_metodos.py)

Esta evaluación **es distinta de las pruebas funcionales** de `PRUEBAS.md`:
usa sus propias 10 consultas (varias de las cuales difieren en la redacción
y en el objeto esperado) y compara **dos métodos bajo las mismas
condiciones** (mismos datos, mismo modelo, mismos pesos y mismo pool):
la búsqueda **solo visual** frente a la **híbrida**. La métrica es el
top-1 de cada método y qué contiene cada top-3.

```bash
python comparar_metodos.py
```

Resultado real (ejecutado con la configuración vigente: pesos 50/50, pool =
tabla completa):

| Consulta (comparativa) | Esperado | Top-1 solo visual | Top-1 híbrido |
|---|---|---|---|
| Componente que emite luz | led_1 / diodo_1 | termometro_1 ✗ | **led_1 ✓** |
| Instrumento para medir voltaje | voltimetro_1 / multimetro_1 | termometro_1 ✗ | termometro_1 ✗ |
| Objeto utilizado para observar microorganismos | microscopio_1 | telefono_1 ✗ | **microscopio_1 ✓** |
| Dispositivo para introducir texto en una computadora | teclado_1 | teclado_1 ✓ | teclado_1 ✓ |
| Instrumento para medir temperatura | termometro_1 | termometro_1 ✓ | termometro_1 ✓ |
| Componente que almacena energía eléctrica | condensador_1 / bateria_1 | termometro_1 ✗ | **condensador_1 ✓** |
| Objeto utilizado para realizar cálculos | calculadora_1 | telefono_1 ✗ | teclado_1 ✗ |
| Instrumento para medir corriente eléctrica | amperimetro_1 / multimetro_1 | termometro_1 ✗ | **amperimetro_1 ✓** |
| Dispositivo para capturar fotografías | camara_1 | camara_1 ✓ | camara_1 ✓ |
| Objeto que permite escuchar música | auriculares_1 | auriculares_1 ✓ | auriculares_1 ✓ |

**Lectura honesta:** con el top-1 como criterio, el híbrido acierta
**8/10** y el solo visual **4/10**; el híbrido nunca *empeora* el objeto que
el visual tenía correcto. Los 2 fracasos del híbrido son
«Instrumento para medir voltaje» (el termómetro es el nº 1 real de la
modalidad visual para esa frase) y «Objeto utilizado para realizar
cálculos» (el teclado desbanca a la calculadora en el espacio textual).
El detalle completo (top-3 y puntuaciones por consulta y método) es la
salida del script.

### Diferencias entre las dos evaluaciones (no confundirlas)

| Aspecto | Pruebas funcionales (`PRUEBAS.md`) | Comparativa (`comparar_metodos.py`) |
|---|---|---|
| Objetivo | validar el sistema completo | comparar 2 métodos |
| Consultas | 10 fijas (con sus alternativas funcionales) | 10 fijas propias (objetivos, algunas distintas) |
| Métodos | híbrido (así se sirve en la app) | solo visual + híbrido |
| Criterio | esperado/alternativa en 1ª posición | top-1 de cada método |
| Salida | `PRUEBAS.md` regenerado | tabla y detalle por consola |

En particular, **no son la misma lista de consultas**: la comparativa usa
«Objeto utilizado para observar microorganismos» y «Objeto utilizado para
realizar cálculos» mientras las funcionales usan «Instrumento para observar
objetos pequeños» y «Dispositivo para medir varias magnitudes eléctricas».
Los resultados no deben mezclarse ni promediarse.

## 19. Ejemplos de consultas (verificados con ejecución real)

En la caja de texto de la app (peso 0.50, top-3 con sus puntuaciones reales):

| Consulta | Resultados (id, puntuación híbrida) |
|---|---|
| `un microscopio` | microscopio_1 0.5661 · led_1 0.5451 · balanza_1 0.5337 |
| `Componente que emite luz` | led_1 0.5703 (¡caso que motivó la fase!) |
| `Instrumento para medir temperatura` | termometro_1 0.6397 |
| `Dispositivo para capturar fotografías` | camara_1 0.6295 |
| `una camara fotografica` | camara_1 0.5774 |
| `una placa blanca con muchos agujeros` | protoboard_1 0.5456 |

## 20. Limitaciones conocidas

1. **Torre de texto de CLIP predominantemente en inglés** (§6): los
   matices finos entre paráfrasis españolas se pierden; las 2 pruebas
   insatisfactorias de `PRUEBAS.md` y 1 de la comparativa se explican por
   ese límite.
2. **Bandas de similitud**: el coseno consulta↔imagen se concentra en
   ≈ 0.19–0.29 y el consulta↔descripción en ≈ 0.83–1.00; entre descripciones
   del propio catálogo el par más alejado mide ≈ 0.73. La poca dispersión
   relativa implica que detalles de ±0.01 pueden cambiar el orden (poca
   discriminación entre descripciones parecidas).
3. **La modalidad textual tiende a dominar el ranking** (mayor dispersión
   entre candidatos) con pesos 50/50 — un hecho medido, no un error.
4. **Catálogo pequeño (1 imagen por objeto; 25)**: los márgenes de ±0.001
   que separan candidatos son ruido; con varias imágenes por objeto la
   señal visual sería más estable.
5. La imagen `reloj_1.jpg` es una foto pequeña (dorso de un reloj) y compite
   mal en la modalidad visual.
6. `_distance` de LanceDB 0.41 es la L2 **al cuadrado** (§9); cualquier
   actualización de versión debe revisar esa conversión.

## 21. Conclusiones y posibles mejoras

**Conclusión**: el sistema funciona de forma local y reproducible,
evaluado con 53 pruebas automatizadas, 10 pruebas funcionales con
resultados reales (8/10) y una comparativa de métodos (híbrido 8/10 frente
a visual 4/10). Los fracasos restantes están documentados con sus causas
medidas.

**Mejoras posibles (no implementadas):** ampliar el catálogo (varias fotos
por objeto), revisar el manejo de «candidato ausente de un pool» si el
catálogo creciese (mostrar «no candidato» en vez de 0.0), considerar un
modelo CLIP multilingüe para el español (implicaría cambiar de modelo, hoy
fuera de los requisitos), o una estrategia de normalización por modalidad.
No se ha aplicado ninguna para no alterar el comportamiento evaluado.

## 22. Cuestiones académicas (solicitadas por el profesor)

**1. Diferencia entre buscar por nombre de archivo y por contenido de imagen.**
Buscar por nombre de archivo significa *coincidencia de cadenas* por el
nombre del fichero o una palabra clave etiquetada a mano. Buscar por
contenido es *interpretar la imagen*: los nombres de fichero solo sirven de
identificación; la modalidad visual
empareja la consulta con el **contenido** de la imagen a través de su
vector de 512-d, y por eso una consulta sin relación con ningún nombre de
fichero («una placa blanca con muchos agujeros») recupera correctamente la
imagen del protoboard (ejecución real de §19). La diferencia es la que hay
entre un *grep sobre nombres de fichero* y una *búsqueda semántica sobre el
contenido*: la parte semántica la proporcionan los embeddings.

**2. Qué sucede si se busca únicamente mediante descripciones textuales.**
En este proyecto, esa búsqueda correspondería a poner el deslizador en
`peso_visual = 0.0` (peso textual = 1.0). Las descripciones con sus
embeddings textuales se usan aquí
como **complemento** dentro del híbrido, no como motor único. Lo que los
datos demuestran: (a) cuando la consulta es una paráfrasis funcional en
español, la modalidad textual es la que sitúa al objeto correcto arriba
(medido en la comparativa: «Componente que emite luz» textual = 0.9101
frente a visual 0.2305; en la prueba funcional 5, el condensador era nº 1
textual); (b) el catálogo de descripciones es pequeño (25 frases de un
estilo parecido) y el coseno entre ellas se concentra en una banda alta
(0.73–0.98 entre pares), de modo que un sistema *solo textual* heredaría
esa baja discriminación. La valoración como limitación está en §20 y como
**hipótesis no experimentada**: el comportamiento exacto con `peso_visual =
0.0` no fue objeto de una evaluación sistemática en este proyecto.

**3. Cuándo mejoran el resultado los embeddings visuales.**
Teóricamente, cuando la consulta describe la **apariencia** de la imagen o
cuando el objeto descrito no aparece bien representado en el vocabulario de
una descripción textual corta, o son sinónimos que la torre de texto aunque
sea «en inglés» sí convierte (p. ej. «una camara fotografica» con errores
ortográficos). En los experimentos de este proyecto: el método solo visual
acierta en 4/10 de la comparativa, y en esas 4 consultas el híbrido
también acierta («introducir texto», «medir temperatura», «capturar
fotografías» y «escuchar música»); el híbrido acierta además los 4
conceptuales (desempeño 8/10). Es decir, el híbrido no degrada ningún
caso que el visual ya acertaba, y añade los conceptuales. Ese resultado
es exactamente lo que mide `comparar_metodos.py`; el README no
afirma más allá de lo que la ejecución demuestra.

## 23. Fuentes y referencias

- **CLIP**: Radford, A. et al. (2021). *Learning Transferable Visual Models
  From Natural Language Supervision.* <https://arxiv.org/abs/2103.00020>.
  Modelo y fichas oficiales:
  <https://huggingface.co/openai/clip-vit-base-patch32>
  y <https://github.com/openai/CLIP>.
- **Hugging Face Transformers**:
  <https://huggingface.co/docs/transformers>.
- **LanceDB** (documentación y repositorio):
  <https://docs.lancedb.com> · <https://github.com/lancedb/lancedb>.
- **Streamlit** (documentación, incl. `streamlit.testing`):
  <https://docs.streamlit.io>.
- **Imágenes del conjunto de datos**: Wikimedia Commons; por cada imagen,
  `data/catalogo.json` registra `titulo_original`, `fuente` (URL verificable
  de la página de la imagen), `licencia` (CC BY-SA 3.0 / 4.0) y `autor` —
  ese fichero es la referencia de procedencia, no este README.
- PyTorch (instalación de la build CPU):
  <https://pytorch.org/get-started/locally/>.

---

Más detalle técnico: **[docs/ARQUITECTURA.md](docs/ARQUITECTURA.md)** ·
Resultados reales de las pruebas funcionales: **[PRUEBAS.md](PRUEBAS.md)**.
