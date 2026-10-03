import pytest
from evaluate import recall_at_k, precision_at_k, evaluar, formatear_reporte


@pytest.mark.parametrize("recuperados, recall, precision", [
    (["a", "a", "b", "c", "d"], 1.0, 0.4),
    (["b", "c", "d", "e", "f"], 0.0, 0.0),
    (["a", "a", "a", "a", "a"], 1.0, 1.0),
    (["a", "a", "a"], 1.0, 0.6),                 # menos de k resultados: precision divide por k, no por 3
    (["b", "c", "d", "e", "f", "a"], 0.0, 0.0),  # un acierto en 6.º lugar no cuenta para k=5
])
def test_metricas_calculadas_a_mano(recuperados, recall, precision):
    assert recall_at_k(recuperados, "a", k=5) == recall
    assert precision_at_k(recuperados, "a", k=5) == pytest.approx(precision)


def test_golden_set_tiene_cinco_preguntas_validas(golden_set, chunks):
    pares_existentes = {(c.metadata["doc_id"], c.metadata["seccion"]) for c in chunks}

    assert len(golden_set) == 5
    assert len({caso["id"] for caso in golden_set}) == 5
    for caso in golden_set:
        assert caso["pregunta"].strip()
        assert caso["tipo"] in {"lexica", "semantica"}
        # Un error de tipeo (p. ej. una tilde) haría que la métrica marque 0 aunque el retriever acierte.
        assert (caso["documento_id_esperado"], caso["seccion_esperada"]) in pares_existentes


def test_evaluar_con_sistema_falso(golden_set, chunks):
    class SiempreAsados:
        def retrieve(self, query):
            return chunks[:5]   # los 5 chunks de asados.txt

    reporte = evaluar(SiempreAsados(), golden_set)

    # Solo la pregunta #5 (asados) acierta, con sus 5 chunks del documento correcto.
    assert reporte["recall_promedio"] == pytest.approx(0.2)
    assert reporte["precision_promedio"] == pytest.approx(0.2)
    assert reporte["seccion_promedio"] == pytest.approx(0.2)
    assert "RECALL@5 PROMEDIO" in formatear_reporte(reporte)


def test_evaluar_golden_set_vacio_falla():
    with pytest.raises(ValueError, match="vacío"):
        evaluar(object(), [])
