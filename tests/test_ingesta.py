import re
from collections import Counter
import pytest
import tiktoken
import config
from ingest import es_titulo, parsear_documento, cargar_chunks

DOCUMENTOS = ["asados", "convivencia_privada", "espacios_comunes", "obras_mudanzas"]


@pytest.mark.parametrize("linea", [
    "Tenencia y Regulación de Animales Domésticos o Mascotas",
    "Condiciones y Reglas de Utilización del Salón Común (Multiuso)",
])
def test_es_titulo_reconoce_encabezados(linea):
    assert es_titulo(linea)


@pytest.mark.parametrize("linea", [
    "",                                                          # vacía
    "\t1.\tProhibición General: Queda formal y estrictamente prohibido",  # ítem numerado con tab
    "⚬\tSábados: Entre las 09:00 y las 14:00 horas.",            # viñeta
    "Entre estos se encuentran:",                                # termina en ':'
    "La piscina está reservada a los copropietarios.",           # oración (termina en '.')
    "x" * 120,                                                   # párrafo largo
])
def test_es_titulo_descarta_lo_que_no_es_encabezado(linea):
    assert not es_titulo(linea)


def test_parsear_documento_separa_secciones_con_metadata():
    secciones = parsear_documento(str(config.DATA_DIR / "asados.txt"))

    assert len(secciones) == 5
    assert secciones[0].metadata["seccion"] == "Introducción"
    for s in secciones:
        assert set(s.metadata) == {"doc_id", "source", "titulo", "seccion"}
        assert s.metadata["doc_id"] == "asados"
        assert s.metadata["source"] == "asados.txt"
        assert s.page_content.startswith(s.metadata["seccion"])  # el encabezado va también en el texto (ayuda a BM25)


def test_cargar_chunks_cubre_los_cuatro_documentos(chunks):
    assert len(chunks) == 22
    assert sorted({c.metadata["doc_id"] for c in chunks}) == DOCUMENTOS


def test_chunk_ids_unicos_y_con_formato(chunks):
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(re.fullmatch(r"[a-z_]+-\d{3}", i) for i in ids)


def test_chunk_ids_deterministas(chunks):
    # Requisito de la ingesta idempotente: re-ejecutar produce los mismos IDs (sobrescribe, no duplica).
    assert [c.metadata["chunk_id"] for c in cargar_chunks()] == [c.metadata["chunk_id"] for c in chunks]


def test_chunk_index_se_reinicia_por_documento(chunks):
    indices_por_doc = Counter()
    for c in chunks:
        assert c.metadata["chunk_index"] == indices_por_doc[c.metadata["doc_id"]]
        indices_por_doc[c.metadata["doc_id"]] += 1


def test_chunks_respetan_el_tamano_maximo_en_tokens(chunks):
    encoding = tiktoken.get_encoding(config.ENCODING_NAME)
    assert max(len(encoding.encode(c.page_content)) for c in chunks) <= config.CHUNK_SIZE


def test_sin_documentos_falla_con_mensaje_claro(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    with pytest.raises(ValueError, match="No se encontraron documentos"):
        cargar_chunks()
