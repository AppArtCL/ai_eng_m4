"""Tests de integración contra OpenAI y Pinecone reales.

Requieren credenciales en .env y el índice poblado (python setup_index.py && python ingest.py).
Se saltan automáticamente si faltan credenciales. Para excluirlos: pytest -m "not integration".
"""
import pytest
import config
from pinecone import Pinecone
from evaluate import evaluar
from rag_system import RAGSystem

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def rag():
    return RAGSystem()


def test_indice_tiene_dimension_y_vectores_esperados(chunks):
    stats = Pinecone(api_key=config.PINECONE_API_KEY).Index(config.PINECONE_INDEX_NAME).describe_index_stats()
    assert stats["dimension"] == config.DIM
    assert stats["namespaces"][config.PINECONE_NAMESPACE]["vector_count"] == len(chunks)


def test_retrieve_real_devuelve_top_k_con_metadata(rag):
    resultado = rag.retrieve("¿Puedo tener mascota?")

    assert len(resultado) == config.TOP_K
    assert len({d.metadata["chunk_id"] for d in resultado}) == config.TOP_K
    assert all({"doc_id", "seccion", "chunk_id"} <= set(d.metadata) for d in resultado)
    assert resultado[0].metadata["chunk_id"] == "convivencia_privada-001"


def test_resultados_deterministas(rag):
    consulta = "¿Se pueden hacer mudanzas los domingos?"
    primera = [d.metadata["chunk_id"] for d in rag.retrieve(consulta)]
    segunda = [d.metadata["chunk_id"] for d in rag.retrieve(consulta)]
    assert primera == segunda


def test_golden_set_recall_completo(rag, golden_set):
    reporte = evaluar(rag, golden_set)
    assert reporte["recall_promedio"] == 1.0
    assert reporte["seccion_promedio"] == 1.0
