# BuscaImagen — Pruebas funcionales de búsqueda (10 consultas)

*Generado automáticamente por `tests/ejecutar_pruebas.py` el 2026-11-06 19:15 a partir de ejecuciones reales del sistema (sin resultados escritos a mano).*

## 1. Introducción

Este documento registra la ejecución de **10 pruebas reales** del sistema de búsqueda de BuscaImagen. Para cada consulta se recupera el top-3 del algoritmo de **búsqueda híbrida** (visual 50 % + textual 50 %, pesos por defecto) y se evalúa si el objeto esperado —o una alternativa funcional documentada— ocupa la primera posición. Todos los datos (ids, puntuaciones, descripciones, rutas) proceden de LanceDB durante la ejecución.

**Nota de trazabilidad (revisión 2026-11-06):** al ampliar el pool de candidatos a toda la tabla (`CANDIDATOS_POOL = 25`) este documento se regeneró íntegramente con los datos reales de esa ejecución: las pruebas 5 y 6 pasaron de «No satisfactoria» a «Correcta» y desapareció el valor «0.0000» de modalidad ausente de la prueba 10. Las columnas de esta versión reflejan la configuración vigente, no la anterior.

## 2. Entorno de pruebas

- **Modelo CLIP**: `openai/clip-vit-base-patch32` (Hugging Face Transformers, ejecución local en CPU; no se descargó ningún modelo adicional).
- **Base de datos**: LanceDB local (`database/`), tablas `imagenes` (25 imágenes) y `descripciones` (25 descripciones).
- **Búsqueda**: híbrida `buscar_imagenes_hibrida()` con PESO_VISUAL = 0.5, PESO_TEXTUAL = 0.5, CANDIDATOS_POOL = 25, top-3.
- **Python**, **Streamlit** y el resto del proyecto sin cambios.

Los vectores son unitarios; «Visual» y «Textual» son similitudes de coseno en [0, 1] y «Híbrida = 0.5·Visual + 0.5·Textual», con cuatro decimales.

## 3. Catálogo utilizado

Total: **25** imágenes reales, sin imágenes añadidas:

- **instrumentos** (10): balanza, microscopio, multimetro, osciloscopio, pipeta, probeta, regla, termometro, voltimetro, amperimetro.
- **cotidianos** (7): calculadora, reloj, telefono, camara, auriculares, teclado, raton.
- **componentes** (8): resistencia, condensador, diodo, transistor, led, circuito_integrado, bateria, protoboard.

## 4. Resultados individuales

### Prueba 1: «Instrumento para medir voltaje»

**Objeto esperado:** `voltimetro_1` — alternativas funcionales aceptadas: `multimetro_1`

| Posición | ID | Nombre | Categoría | Visual | Textual | Híbrida | Descripción | Ruta |
|---|---|---|---|---|---|---|---|---|
| 1 | `termometro_1` | termometro | instrumentos | 0.2574 | 0.9261 | 0.5918 | Instrumento para medir la temperatura | `data/imagenes/instrumentos/termometro_1.jpg` |
| 2 | `balanza_1` | balanza | instrumentos | 0.2144 | 0.9507 | 0.5826 | Instrumento para medir la masa de objetos | `data/imagenes/instrumentos/balanza_1.jpg` |
| 3 | `multimetro_1` | multimetro | instrumentos | 0.2303 | 0.9308 | 0.5806 | Instrumento para medir voltaje, corriente y resistencia | `data/imagenes/instrumentos/multimetro_1.jpg` |

**Evaluación: No satisfactoria** — Ni el objeto esperado ni una alternativa funcional definida ocupan la primera posición. El objeto esperado no alcanza los tres primeros resultados.

### Prueba 2: «Instrumento para medir corriente eléctrica»

**Objeto esperado:** `amperimetro_1` — alternativas funcionales aceptadas: `multimetro_1`

| Posición | ID | Nombre | Categoría | Visual | Textual | Híbrida | Descripción | Ruta |
|---|---|---|---|---|---|---|---|---|
| 1 | `amperimetro_1` | amperimetro | instrumentos | 0.2498 | 0.9859 | 0.6178 | Instrumento para medir la intensidad de corriente eléctrica | `data/imagenes/instrumentos/amperimetro_1.jpg` |
| 2 | `multimetro_1` | multimetro | instrumentos | 0.2590 | 0.9453 | 0.6022 | Instrumento para medir voltaje, corriente y resistencia | `data/imagenes/instrumentos/multimetro_1.jpg` |
| 3 | `osciloscopio_1` | osciloscopio | instrumentos | 0.2607 | 0.9423 | 0.6015 | Instrumento para visualizar señales eléctricas | `data/imagenes/instrumentos/osciloscopio_1.jpg` |

**Evaluación: Correcta** — El objeto esperado aparece en primera posición.

### Prueba 3: «Dispositivo para medir varias magnitudes eléctricas»

**Objeto esperado:** `multimetro_1`

| Posición | ID | Nombre | Categoría | Visual | Textual | Híbrida | Descripción | Ruta |
|---|---|---|---|---|---|---|---|---|
| 1 | `calculadora_1` | calculadora | cotidianos | 0.2439 | 0.9214 | 0.5826 | Dispositivo electrónico para realizar cálculos | `data/imagenes/cotidianos/calculadora_1.jpg` |
| 2 | `amperimetro_1` | amperimetro | instrumentos | 0.2530 | 0.9116 | 0.5823 | Instrumento para medir la intensidad de corriente eléctrica | `data/imagenes/instrumentos/amperimetro_1.jpg` |
| 3 | `voltimetro_1` | voltimetro | instrumentos | 0.2516 | 0.9124 | 0.5820 | Instrumento para medir la diferencia de potencial eléctrico | `data/imagenes/instrumentos/voltimetro_1.jpg` |

**Evaluación: No satisfactoria** — Ni el objeto esperado ni una alternativa funcional definida ocupan la primera posición. El objeto esperado no alcanza los tres primeros resultados.

### Prueba 4: «Componente que emite luz»

**Objeto esperado:** `led_1` — alternativas funcionales aceptadas: `diodo_1`

| Posición | ID | Nombre | Categoría | Visual | Textual | Híbrida | Descripción | Ruta |
|---|---|---|---|---|---|---|---|---|
| 1 | `led_1` | led | componentes | 0.2305 | 0.9101 | 0.5703 | Diodo que emite luz al pasar corriente | `data/imagenes/componentes/led_1.jpg` |
| 2 | `regla_1` | regla | instrumentos | 0.2129 | 0.8541 | 0.5335 | Instrumento para medir longitudes | `data/imagenes/instrumentos/regla_1.jpg` |
| 3 | `amperimetro_1` | amperimetro | instrumentos | 0.2194 | 0.8450 | 0.5322 | Instrumento para medir la intensidad de corriente eléctrica | `data/imagenes/instrumentos/amperimetro_1.jpg` |

**Evaluación: Correcta** — El objeto esperado aparece en primera posición.

### Prueba 5: «Componente que almacena energía eléctrica»

**Objeto esperado:** `bateria_1`, `condensador_1`

| Posición | ID | Nombre | Categoría | Visual | Textual | Híbrida | Descripción | Ruta |
|---|---|---|---|---|---|---|---|---|
| 1 | `condensador_1` | condensador | componentes | 0.2139 | 0.9736 | 0.5938 | Componente electrónico que almacena carga eléctrica | `data/imagenes/componentes/condensador_1.jpg` |
| 2 | `transistor_1` | transistor | componentes | 0.2358 | 0.9384 | 0.5871 | Componente electrónico que amplifica o conmuta señales | `data/imagenes/componentes/transistor_1.jpg` |
| 3 | `voltimetro_1` | voltimetro | instrumentos | 0.2386 | 0.9344 | 0.5865 | Instrumento para medir la diferencia de potencial eléctrico | `data/imagenes/instrumentos/voltimetro_1.jpg` |

**Evaluación: Correcta** — El objeto esperado aparece en primera posición.

### Prueba 6: «Instrumento para observar objetos pequeños»

**Objeto esperado:** `microscopio_1`

| Posición | ID | Nombre | Categoría | Visual | Textual | Híbrida | Descripción | Ruta |
|---|---|---|---|---|---|---|---|---|
| 1 | `microscopio_1` | microscopio | instrumentos | 0.2259 | 0.9933 | 0.6096 | Instrumento para observar objetos muy pequeños | `data/imagenes/instrumentos/microscopio_1.jpg` |
| 2 | `balanza_1` | balanza | instrumentos | 0.2250 | 0.9695 | 0.5972 | Instrumento para medir la masa de objetos | `data/imagenes/instrumentos/balanza_1.jpg` |
| 3 | `termometro_1` | termometro | instrumentos | 0.2558 | 0.9344 | 0.5951 | Instrumento para medir la temperatura | `data/imagenes/instrumentos/termometro_1.jpg` |

**Evaluación: Correcta** — El objeto esperado aparece en primera posición.

### Prueba 7: «Instrumento para medir temperatura»

**Objeto esperado:** `termometro_1`

| Posición | ID | Nombre | Categoría | Visual | Textual | Híbrida | Descripción | Ruta |
|---|---|---|---|---|---|---|---|---|
| 1 | `termometro_1` | termometro | instrumentos | 0.2920 | 0.9873 | 0.6397 | Instrumento para medir la temperatura | `data/imagenes/instrumentos/termometro_1.jpg` |
| 2 | `multimetro_1` | multimetro | instrumentos | 0.2608 | 0.9216 | 0.5912 | Instrumento para medir voltaje, corriente y resistencia | `data/imagenes/instrumentos/multimetro_1.jpg` |
| 3 | `regla_1` | regla | instrumentos | 0.2816 | 0.8991 | 0.5903 | Instrumento para medir longitudes | `data/imagenes/instrumentos/regla_1.jpg` |

**Evaluación: Correcta** — El objeto esperado aparece en primera posición.

### Prueba 8: «Dispositivo para introducir texto en una computadora»

**Objeto esperado:** `teclado_1`

| Posición | ID | Nombre | Categoría | Visual | Textual | Híbrida | Descripción | Ruta |
|---|---|---|---|---|---|---|---|---|
| 1 | `teclado_1` | teclado | cotidianos | 0.2652 | 0.8728 | 0.5690 | Periférico de entrada con teclas para escribir | `data/imagenes/cotidianos/teclado_1.jpg` |
| 2 | `balanza_1` | balanza | instrumentos | 0.2107 | 0.9043 | 0.5575 | Instrumento para medir la masa de objetos | `data/imagenes/instrumentos/balanza_1.jpg` |
| 3 | `osciloscopio_1` | osciloscopio | instrumentos | 0.2334 | 0.8800 | 0.5567 | Instrumento para visualizar señales eléctricas | `data/imagenes/instrumentos/osciloscopio_1.jpg` |

**Evaluación: Correcta** — El objeto esperado aparece en primera posición.

### Prueba 9: «Dispositivo utilizado para capturar fotografías»

**Objeto esperado:** `camara_1` — alternativas funcionales aceptadas: `telefono_1`

| Posición | ID | Nombre | Categoría | Visual | Textual | Híbrida | Descripción | Ruta |
|---|---|---|---|---|---|---|---|---|
| 1 | `camara_1` | camara | cotidianos | 0.2560 | 0.9868 | 0.6214 | Dispositivo para capturar fotografías | `data/imagenes/cotidianos/camara_1.jpg` |
| 2 | `telefono_1` | telefono | cotidianos | 0.2397 | 0.8545 | 0.5471 | Dispositivo portátil para comunicaciones y funciones digitales | `data/imagenes/cotidianos/telefono_1.jpg` |
| 3 | `calculadora_1` | calculadora | cotidianos | 0.2111 | 0.8828 | 0.5470 | Dispositivo electrónico para realizar cálculos | `data/imagenes/cotidianos/calculadora_1.jpg` |

**Evaluación: Correcta** — El objeto esperado aparece en primera posición.

### Prueba 10: «Dispositivo para escuchar música»

**Objeto esperado:** `auriculares_1` — alternativas funcionales aceptadas: `telefono_1`

| Posición | ID | Nombre | Categoría | Visual | Textual | Híbrida | Descripción | Ruta |
|---|---|---|---|---|---|---|---|---|
| 1 | `auriculares_1` | auriculares | cotidianos | 0.2643 | 0.9479 | 0.6061 | Dispositivo para escuchar audio de forma personal | `data/imagenes/cotidianos/auriculares_1.jpg` |
| 2 | `balanza_1` | balanza | instrumentos | 0.2151 | 0.8907 | 0.5529 | Instrumento para medir la masa de objetos | `data/imagenes/instrumentos/balanza_1.jpg` |
| 3 | `microscopio_1` | microscopio | instrumentos | 0.2060 | 0.8969 | 0.5514 | Instrumento para observar objetos muy pequeños | `data/imagenes/instrumentos/microscopio_1.jpg` |

**Evaluación: Correcta** — El objeto esperado aparece en primera posición.

## 5. Resumen general

| Nº | Consulta | Resultado en 1ª posición | Clasificación |
|---|---|---|---|
| 1 | «Instrumento para medir voltaje» | `termometro_1` | **No satisfactoria** |
| 2 | «Instrumento para medir corriente eléctrica» | `amperimetro_1` | **Correcta** |
| 3 | «Dispositivo para medir varias magnitudes eléctricas» | `calculadora_1` | **No satisfactoria** |
| 4 | «Componente que emite luz» | `led_1` | **Correcta** |
| 5 | «Componente que almacena energía eléctrica» | `condensador_1` | **Correcta** |
| 6 | «Instrumento para observar objetos pequeños» | `microscopio_1` | **Correcta** |
| 7 | «Instrumento para medir temperatura» | `termometro_1` | **Correcta** |
| 8 | «Dispositivo para introducir texto en una computadora» | `teclado_1` | **Correcta** |
| 9 | «Dispositivo utilizado para capturar fotografías» | `camara_1` | **Correcta** |
| 10 | «Dispositivo para escuchar música» | `auriculares_1` | **Correcta** |

- **Correcta**: 8
- **Relacionada**: 0
- **No satisfactoria**: 2
- **No evaluable**: 0
- **Total**: 10

## 6. Análisis de resultados

**Aciertos (8):** «Instrumento para medir corriente eléctrica» devolvió `amperimetro_1` (h=0.6178); «Componente que emite luz» devolvió `led_1` (h=0.5703); «Componente que almacena energía eléctrica» devolvió `condensador_1` (h=0.5938); «Instrumento para observar objetos pequeños» devolvió `microscopio_1` (h=0.6096); «Instrumento para medir temperatura» devolvió `termometro_1` (h=0.6397); «Dispositivo para introducir texto en una computadora» devolvió `teclado_1` (h=0.5690); «Dispositivo utilizado para capturar fotografías» devolvió `camara_1` (h=0.6214); «Dispositivo para escuchar música» devolvió `auriculares_1` (h=0.6061).

**Fracasos (2):**

- «Instrumento para medir voltaje»: primer resultado `termometro_1` (h=0.5918, v=0.2574, t=0.9261). Ni el objeto esperado ni una alternativa funcional definida ocupan la primera posición. El objeto esperado no alcanza los tres primeros resultados.
- «Dispositivo para medir varias magnitudes eléctricas»: primer resultado `calculadora_1` (h=0.5826, v=0.2439, t=0.9214). Ni el objeto esperado ni una alternativa funcional definida ocupan la primera posición. El objeto esperado no alcanza los tres primeros resultados.

**Confusiones habituales:** los «instrumentos de medición» en español tienden a votar al termómetro en la modalidad visual (torre de texto de CLIP entrenada solo en inglés) y la modalidad textual tiende a agrupar descripciones que comparten el patrón «Instrumento para medir …». El caso «Componente que emite luz» es el motivo por el que existe la modalidad textual: el LED gana por la afinidad con su descripción real («Diodo que emite luz al pasar corriente»).

**Limitaciones:** catálogo pequeño (1 imagen por objeto), similitudes textuales concentradas en una banda estrecha (0.8–1.0, poca discriminación entre descripciones parecidas) y consultas en español contra un modelo con torre de texto solo en inglés.

## 7. Conclusiones

Con la configuración por defecto (50/50), el sistema acierta en primera posición en 8/10 pruebas y devuelve una alternativa funcional en otras 0. Las 2 restantes muestran la limitación conocida del catálogo pequeño y del español frente a CLIP base-patch32; el detalle de puntuaciones permite reproducir y auditar cada resultado. La búsqueda híbrida no se modificó durante estas pruebas.
