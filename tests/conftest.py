"""Fixtures compartidas por todos los tests.

Los tests offline no usan red: trabajan con los chunks locales, BM25 y un retriever vectorial falso.
Los tests marcados con @pytest.mark.integration se saltan si faltan credenciales.
"""
import pytest
import config
import evaluate
from ingest import cargar_chunks
from retrievers import crear_bm25
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

HAY_CREDENCIALES = all([config.OPENAI_API_KEY, config.PINECONE_API_KEY, config.PINECONE_INDEX_NAME])


def pytest_collection_modifyitems(config, items):
    if HAY_CREDENCIALES:
        return
    saltar = pytest.mark.skip(reason="faltan OPENAI_API_KEY / PINECONE_API_KEY / PINECONE_INDEX_NAME")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(saltar)


# Retriever vectorial falso: devuelve siempre los chunks indicados, en ese orden, sin llamar a Pinecone.
class VectorialFalso(BaseRetriever):
    documentos: list[Document]

    def _get_relevant_documents(self, query, *, run_manager=None):
        return self.documentos


@pytest.fixture(scope="session")
def chunks() -> list[Document]:
    return cargar_chunks()


@pytest.fixture(scope="session")
def chunks_por_id(chunks) -> dict[str, Document]:
    return {c.metadata["chunk_id"]: c for c in chunks}


@pytest.fixture(scope="session")
def bm25(chunks):
    return crear_bm25(chunks)


@pytest.fixture(scope="session")
def golden_set() -> list[dict]:
    return evaluate.cargar_golden_set()


# Crea un VectorialFalso a partir de una lista de chunk_id.
@pytest.fixture
def vectorial_falso(chunks_por_id):
    def _crear(ids: list[str]) -> VectorialFalso:
        return VectorialFalso(documentos=[chunks_por_id[i] for i in ids])
    return _crear


# Simula un entorno sin credenciales: prueba que el código offline no depende de ellas.
@pytest.fixture
def sin_credenciales(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", None)
    monkeypatch.setattr(config, "PINECONE_API_KEY", None)
