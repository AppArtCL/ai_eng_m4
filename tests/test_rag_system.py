"""Tests offline de RAGSystem: BM25 real + retriever vectorial falso inyectado (sin credenciales)."""
import config
from rag_system import RAGSystem

CONSULTA = "¿Qué dice el reglamento sobre las UTM?"
IDS_VECTORIAL = ["asados-001", "convivencia_privada-004", "asados-002", "asados-003", "asados-004"]


def ids(documentos):
    return [d.metadata["chunk_id"] for d in documentos]


def test_retrieve_devuelve_top_k_sin_duplicados(sin_credenciales, bm25, vectorial_falso):
    rag = RAGSystem(bm25=bm25, vectorial=vectorial_falso(IDS_VECTORIAL))
    resultado = ids(rag.retrieve(CONSULTA))

    assert len(resultado) == config.TOP_K   # el ensemble devuelve la unión (hasta 10); retrieve recorta
    assert len(set(resultado)) == len(resultado)


def test_chunk_presente_en_ambas_listas_sube_al_primer_lugar(sin_credenciales, bm25, vectorial_falso):
    # convivencia_privada-004 es 1.º en BM25 y 2.º en el vectorial: RRF suma ambas contribuciones.
    rag = RAGSystem(bm25=bm25, vectorial=vectorial_falso(IDS_VECTORIAL))
    assert ids(rag.retrieve(CONSULTA))[0] == "convivencia_privada-004"


def test_fusion_usa_chunk_id_como_clave(sin_credenciales, bm25, vectorial_falso):
    rag = RAGSystem(bm25=bm25, vectorial=vectorial_falso(IDS_VECTORIAL))
    assert rag.retriever_hibrido.id_key == "chunk_id"


def test_pesos_desiguales_el_top_k_es_el_conjunto_del_retriever_dominante(sin_credenciales, bm25, vectorial_falso):
    # Hallazgo documentado en el README: con c=60 y listas de 5, el 5.º del retriever con peso 0.7
    # (0.7/65) supera al 1.º del de peso 0.3 (0.3/61). El liviano no puede meter chunks propios en el top-5;
    # solo puede REORDENAR los que comparte con el dominante (sumándoles su aporte).
    vectorial = vectorial_falso(IDS_VECTORIAL)
    rag_vectorial = RAGSystem(pesos=(0.3, 0.7), bm25=bm25, vectorial=vectorial)
    rag_bm25 = RAGSystem(pesos=(0.7, 0.3), bm25=bm25, vectorial=vectorial)

    assert set(ids(rag_vectorial.retrieve(CONSULTA))) == set(IDS_VECTORIAL)
    assert set(ids(rag_bm25.retrieve(CONSULTA))) == set(ids(bm25.invoke(CONSULTA)))


def test_pesos_desiguales_el_retriever_liviano_reordena_los_compartidos(sin_credenciales, bm25, vectorial_falso):
    # convivencia_privada-004: 2.º en el vectorial + 1.º en BM25 → supera al 1.º del vectorial (asados-001).
    rag = RAGSystem(pesos=(0.3, 0.7), bm25=bm25, vectorial=vectorial_falso(IDS_VECTORIAL))
    assert ids(rag.retrieve(CONSULTA))[0] == "convivencia_privada-004"
