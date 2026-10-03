import json
from typing import Protocol
import config
from langchain_core.documents import Document
from rag_system import RAGSystem

GOLDEN_SET = config.BASE_DIR / "golden_set.json"
SALIDA = config.BASE_DIR / "resultados" / "evaluacion.txt"

class Recuperador(Protocol):
    def retrieve(self, query: str) -> list[Document]: ...

def recall_at_k(recuperados: list[str], esperado: str, k: int = config.TOP_K) -> float:
    return 1.0 if esperado in recuperados[:k] else 0.0

def precision_at_k(recuperados: list[str], esperado: str, k: int = config.TOP_K) -> float:
    return sum(1 for r in recuperados[:k] if r == esperado) / k


def cargar_golden_set(ruta=GOLDEN_SET) -> list[dict]:
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def evaluar(sistema: Recuperador, golden_set: list[dict], k: int = config.TOP_K) -> dict:
    if not golden_set:
        raise ValueError("El golden set está vacío.")

    detalle = []
    for caso in golden_set:
        documentos = sistema.retrieve(caso["pregunta"])[:k]
        doc_ids = [d.metadata["doc_id"] for d in documentos]
        secciones = [d.metadata["seccion"] for d in documentos]
        esperado = caso["documento_id_esperado"]

        detalle.append({
            "id": caso.get("id"),
            "pregunta": caso["pregunta"],
            "tipo": caso.get("tipo", "-"),
            "esperado": esperado,
            "recuperados": doc_ids,
            "recall": recall_at_k(doc_ids, esperado, k),
            "precision": precision_at_k(doc_ids, esperado, k),
            "seccion_ok": caso["seccion_esperada"] in secciones if "seccion_esperada" in caso else None,
        })

    n = len(detalle)
    con_seccion = [r for r in detalle if r["seccion_ok"] is not None]
    return {
        "k": k,
        "detalle": detalle,
        "recall_promedio": sum(r["recall"] for r in detalle) / n,
        "precision_promedio": sum(r["precision"] for r in detalle) / n,
        "seccion_promedio": sum(r["seccion_ok"] for r in con_seccion) / len(con_seccion) if con_seccion else None,
    }


def formatear_reporte(reporte: dict) -> str:
    k = reporte["k"]
    lineas = ["=" * 80]
    for r in reporte["detalle"]:
        estado = "✅" if r["recall"] == 1.0 else "❌"
        lineas.append(f"{estado} #{r['id']} [{r['tipo']}] {r['pregunta']}")
        lineas.append(f"   Esperado: {r['esperado']} | Recuperados: {r['recuperados']}")
        seccion = "" if r["seccion_ok"] is None else f" | Sección correcta en top-{k}: {'sí' if r['seccion_ok'] else 'no'}"
        lineas.append(f"   Recall@{k}: {r['recall']:.0%} | Precision@{k}: {r['precision']:.0%}{seccion}")
        lineas.append("")

    lineas.append("=" * 80)
    lineas.append(f"📊 RECALL@{k} PROMEDIO:    {reporte['recall_promedio']:.1%}")
    lineas.append(f"📊 PRECISION@{k} PROMEDIO: {reporte['precision_promedio']:.1%}")
    if reporte["seccion_promedio"] is not None:
        lineas.append(f"📊 SECCIÓN CORRECTA EN TOP-{k}: {reporte['seccion_promedio']:.1%}")
    return "\n".join(lineas)


if __name__ == "__main__":
    reporte = evaluar(RAGSystem(), cargar_golden_set())
    texto = formatear_reporte(reporte)
    print(texto)

    SALIDA.parent.mkdir(exist_ok=True)
    SALIDA.write_text(texto + "\n", encoding="utf-8")
    print(f"\nReporte guardado en {SALIDA.relative_to(config.BASE_DIR)}")
