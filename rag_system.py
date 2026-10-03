import logging
import config
from ingest import cargar_chunks
from retrievers import crear_bm25, crear_vectorial
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class RAGSystem:
    def __init__(self, k: int = config.TOP_K, pesos: tuple[float, float] = (0.5, 0.5)):
        self.k = k
        self.chunks = cargar_chunks()
        self.bm25 = crear_bm25(self.chunks, k)
        self.vectorial = crear_vectorial(k)
        self.retriever_hibrido = EnsembleRetriever(
            retrievers=[self.bm25, self.vectorial],
            weights=list(pesos),
            id_key="chunk_id",
        )

    # Devuelve los top-k chunks combinando resultados léxicos y semánticos.
    def retrieve(self, query: str) -> list[Document]:
        documentos = self.retriever_hibrido.invoke(query)[:self.k]
        return documentos
    

if __name__ == "__main__":
    rag = RAGSystem()
    consultas = [
        "¿Puedo tener mascota?",
        "¿Qué dice el reglamento sobre las UTM?",
        "Departamentos 101 y 104:",
        "¿Puedo hacer asados?"
    ]
    for consulta in consultas:
        print(f"\nConsulta: {consulta}")
        for posicion, doc in enumerate(rag.retrieve(consulta), start=1):
            print(f"  {posicion}. {doc.metadata['chunk_id']:<26} | {doc.metadata['seccion']}")
