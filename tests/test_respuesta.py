# -*- coding: utf-8 -*-
"""
Pruebas de la generación de respuestas basadas en los resultados.

La respuesta debe construirse EXCLUSIVAMENTE con las descripciones de
LanceDB de los tres resultados de la búsqueda híbrida, sin modelos
generativos ni invención de características.
"""

from src.search.respuesta_generador import generar_respuesta


def _resultado(nombre, categoria="instrumentos", descripcion="Descripción de prueba."):
    return {
        "id": f"{nombre}_test",
        "nombre": nombre,
        "categoria": categoria,
        "descripcion": descripcion,
        "ruta_imagen": f"data/imagenes/{categoria}/{nombre}_test.jpg",
        "similitud_visual": 0.2,
        "similitud_textual": 0.8,
        "puntuacion": 0.5,
        "distancia": 1.4,
    }


class TestsCasosBasicos:

    def test_sin_resultados_devuelve_none(self):
        assert generar_respuesta("una consulta", []) is None
        assert generar_respuesta("una consulta", None) is None

    def test_consulta_vacia_devuelve_none(self):
        assert generar_respuesta("", [_resultado("microscopio")]) is None
        assert generar_respuesta("   ", [_resultado("microscopio")]) is None

    def test_sin_descripciones_informacion_insuficiente(self):
        resultados = [
            _resultado("microscopio", descripcion=""),
            _resultado("balanza", descripcion=None),
        ]
        assert generar_respuesta("un microscopio", resultados) is None

    def test_respuesta_es_texto_y_empieza_bien(self):
        resultados = [_resultado("microscopio"), _resultado("balanza"), _resultado("regla")]
        respuesta = generar_respuesta("instrumentos del laboratorio", resultados)
        assert isinstance(respuesta, str)
        assert respuesta.startswith("Los resultados encontrados para")
        assert "instrumentos del laboratorio" in respuesta


class TestsContenidoExtractivo:
    """La respuesta solo contiene descripciones reales: nada inventado."""

    def test_incluye_nombres_y_descripciones_literalmente(self):
        resultados = [
            _resultado("multimetro", descripcion="Instrumento para medir voltaje, corriente y resistencia."),
            _resultado("voltimetro", descripcion="Instrumento para medir la diferencia de potencial eléctrico."),
            _resultado("amperimetro", descripcion="Instrumento para medir la intensidad de corriente eléctrica."),
        ]
        respuesta = generar_respuesta("instrumentos para mediciones eléctricas", resultados)
        for nombre in ("multimetro", "voltimetro", "amperimetro"):
            assert nombre in respuesta
        assert "instrumento para medir voltaje, corriente y resistencia" in respuesta
        assert "instrumento para medir la diferencia de potencial eléctrico" in respuesta
        assert "instrumento para medir la intensidad de corriente eléctrica" in respuesta

    def test_no_inventa_palabras_ajenas_a_las_descripciones(self):
        resultados = [_resultado("camara", descripcion="Dispositivo para capturar fotografías.")]
        respuesta = generar_respuesta("una camara fotografica", resultados)
        assert "camara" in respuesta
        assert "capturar fotografías" in respuesta
        # invents? phrases absent in source:
        assert "cámara con flash" not in respuesta
        assert "usb" not in respuesta.lower()

    def test_descripcion_repetida_se_menciona_una_sola_vez_y_se_avisa(self):
        resultados = [
            _resultado("objeto_a", descripcion="Dispositivo para ver pequeños detalles."),
            _resultado("objeto_b", descripcion="Dispositivo para ver pequeños detalles."),
            _resultado("objeto_c", descripcion="Otra cosa distinta por completo."),
        ]
        respuesta = generar_respuesta("consulta de prueba", resultados)
        assert respuesta.count("dispositivo para ver pequeños detalles") == 1
        assert "repiten descripciones ya mencionadas" in respuesta

    def test_nombre_repetido_al_inicio_se_quita(self):
        """nombre 'diodo' + 'Diodo que emite luz…' no repite el nombre."""
        resultados = [
            _resultado("diodo", categoria="componentes",
                       descripcion="Diodo que emite luz al pasar corriente"),
        ]
        respuesta = generar_respuesta("Componente que emite luz", resultados)
        assert "que emite luz al pasar corriente" in respuesta
        # el nombre aparece exactamente una vez (fuera del paréntesis)
        assert respuesta.count("diodo") == 1


class TestsFormato:

    def test_dos_menciones_unidas_con_y(self):
        resultados = [
            _resultado("objeto_a", descripcion="A."),
            _resultado("objeto_b", descripcion="B."),
        ]
        respuesta = generar_respuesta("consulta", resultados)
        assert "objeto_a (a) y objeto_b (b)." in respuesta

    def test_termina_en_punto_unico(self):
        resultados = [_resultado("microscopio", descripcion="Instrumento para observar.")]
        respuesta = generar_respuesta("consulta", resultados)
        assert respuesta.endswith("observar).")

    def test_respuesta_breve(self):
        resultados = [
            _resultado("microscopio", descripcion="Instrumento para observar objetos muy pequeños"),
            _resultado("termometro", descripcion="Instrumento para medir la temperatura"),
            _resultado("regla", descripcion="Instrumento para medir longitudes"),
        ]
        respuesta = generar_respuesta("instrumentos de medida", resultados)
        assert len(respuesta) < 600

    def test_consulta_demasiado_larga_se_recorta(self):
        consulta = "palabra " * 50
        resultados = [_resultado("microscopio", descripcion="Instrumento para observar.")]
        respuesta = generar_respuesta(consulta, resultados)
        assert "…" in respuesta


class TestsIntegracionConBusqueda:
    """Con resultados reales de la búsqueda híbrida (LanceDB)."""

    def test_respuesta_con_busqueda_real(self):
        from src.config import BASE_DIR
        from src.search.image_search import buscar_imagenes_hibrida

        resultados = buscar_imagenes_hibrida("un microscopio", k=3)
        respuesta = generar_respuesta("un microscopio", resultados)
        assert respuesta is not None
        assert "un microscopio" in respuesta
        for fila in resultados:
            assert fila["nombre"] in respuesta
            # descripción literal salvo la minúscula inicial dentro del paréntesis
            assert fila["descripcion"].rstrip(". ")[1:] in respuesta

    def test_respuesta_actualizable_por_consulta(self):
        from src.search.image_search import buscar_imagenes_hibrida

        r1 = generar_respuesta("un microscopio", buscar_imagenes_hibrida("un microscopio", k=3))
        r2 = generar_respuesta("una camara fotografica", buscar_imagenes_hibrida("una camara fotografica", k=3))
        assert "un microscopio" in r1
        assert "una camara fotografica" in r2
        assert r1 != r2
