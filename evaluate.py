import json
from typing import Protocol
import config
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from ingest import cargar_chunks
from retrievers import crear_bm25, crear_vectorial
from rag_system import RAGSystem

GOLDEN_SET = config.BASE_DIR / "golden_set.json"
SALIDA = config.BASE_DIR / "resultados" / "evaluacion.txt"
PESOS_A_COMPARAR = [(0.3, 0.7), (0.5, 0.5), (0.7, 0.3)]   # [BM25, vectorial]

class Recuperador(Protocol):
    def retrieve(self, query: str) -> list[Document]: ...


# Adapta un retriever de LangChain (BM25 o vectorial solo) a la interfaz retrieve() que usa evaluar().
class RetrieverSolo:
    def __init__(self, retriever: BaseRetriever, k: int = config.TOP_K):
        self.retriever = retriever
        self.k = k

    def retrieve(self, query: str) -> list[Document]:
        return self.retriever.invoke(query)[:self.k]

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


# Evalúa BM25 solo, vectorial solo y el híbrido con cada combinación de pesos.
# Los retrievers se crean una sola vez y se inyectan en cada RAGSystem.
def comparar_configuraciones(golden_set: list[dict], k: int = config.TOP_K) -> dict[str, dict]:
    bm25 = crear_bm25(cargar_chunks(), k)
    vectorial = crear_vectorial(k)

    sistemas: dict[str, Recuperador] = {
        "BM25 solo": RetrieverSolo(bm25, k),
        "Vectorial solo": RetrieverSolo(vectorial, k),
    }
    for pesos in PESOS_A_COMPARAR:
        nombre = f"Híbrido {pesos[0]:.1f} / {pesos[1]:.1f}"
        sistemas[nombre] = RAGSystem(k=k, pesos=pesos, bm25=bm25, vectorial=vectorial)

    return {nombre: evaluar(sistema, golden_set, k) for nombre, sistema in sistemas.items()}


def formatear_comparacion(resultados: dict[str, dict]) -> str:
    primero = next(iter(resultados.values()))
    k = primero["k"]
    ids = [f"#{r['id']}" for r in primero["detalle"]]

    lineas = [
        "=" * 80,
        f"COMPARACIÓN DE CONFIGURACIONES (pesos = [BM25, vectorial])",
        "=" * 80,
        f"{'Configuración':<20} {'Recall@' + str(k):>9} {'Prec@' + str(k):>8} {'Sección':>8}   "
        + "  ".join(f"{i:>4}" for i in ids) + f"   ← Precision@{k} por pregunta",
    ]
    for nombre, rep in resultados.items():
        por_pregunta = "  ".join(f"{r['precision']:>4.0%}" for r in rep["detalle"])
        lineas.append(
            f"{nombre:<20} {rep['recall_promedio']:>9.0%} {rep['precision_promedio']:>8.0%} "
            f"{rep['seccion_promedio']:>8.0%}   {por_pregunta}"
        )
    return "\n".join(lineas)


if __name__ == "__main__":
    golden_set = cargar_golden_set()
    reporte = evaluar(RAGSystem(), golden_set)
    texto = formatear_reporte(reporte) + "\n\n" + formatear_comparacion(comparar_configuraciones(golden_set))
    print(texto)

    SALIDA.parent.mkdir(exist_ok=True)
    SALIDA.write_text(texto + "\n", encoding="utf-8")
    print(f"\nReporte guardado en {SALIDA.relative_to(config.BASE_DIR)}")
