import pytest
import config
from retrievers import quitar_tildes, preprocesar, crear_vectorial, STOPWORDS


@pytest.mark.parametrize("texto, esperado", [
    ("régimen", "regimen"),
    ("Prohibición Comercial", "Prohibicion Comercial"),
    ("sin tildes", "sin tildes"),
])
def test_quitar_tildes(texto, esperado):
    assert quitar_tildes(texto) == esperado


@pytest.mark.parametrize("texto, esperado", [
    ("¿Puedo tener mascota?", ["puedo", "tener", "mascota"]),
    ("¿Qué dice el reglamento sobre las UTM?", ["dice", "reglamento", "sobre", "utm"]),
    ("Régimen de Multas (UTM).", ["regimen", "multa", "utm"]),
])
def test_preprocesar(texto, esperado):
    assert preprocesar(texto) == esperado


def test_preprocesar_devuelve_lista_sin_stopwords():
    tokens = preprocesar("El uso de la piscina y del salón")
    assert isinstance(tokens, list)  # un str haría que BM25 indexe letra por letra
    assert not STOPWORDS & set(tokens)


def test_preprocesar_unifica_consulta_y_corpus():
    # "Multas" en el reglamento y "multa" en la pregunta deben producir el mismo token.
    assert preprocesar("Multas")[0] == preprocesar("multa")[0]


def test_bm25_usa_preprocesar_y_k_de_config(bm25):
    # Protege contra el error silencioso de un nombre de parámetro mal escrito (preprocess_fn).
    assert bm25.preprocess_func is preprocesar
    assert bm25.k == config.TOP_K


@pytest.mark.parametrize("consulta, chunk_esperado", [
    ("¿Qué dice el reglamento sobre las UTM?", "convivencia_privada-004"),
    ("¿Puedo tener mascota?", "convivencia_privada-001"),
    ("Departamentos 101 y 104", "asados-004"),
])
def test_bm25_recupera_terminos_exactos_en_primer_lugar(bm25, consulta, chunk_esperado):
    assert bm25.invoke(consulta)[0].metadata["chunk_id"] == chunk_esperado


def test_crear_vectorial_sin_credenciales_falla(sin_credenciales):
    with pytest.raises(ValueError):
        crear_vectorial()
