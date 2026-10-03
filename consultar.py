"""Modo interactivo: consulta el sistema RAG híbrido (BM25 + Pinecone) desde la terminal.

Uso:
    python consultar.py      # requiere OPENAI_API_KEY y PINECONE_API_KEY en .env

Escribe una pregunta y presiona Enter. Para terminar: 'salir', 'exit', 'quit' o Ctrl+C.
También acepta preguntas desde un archivo o pipe:
    printf '¿Me pueden multar si fumo en el pasillo?\\nsalir\\n' | python consultar.py

Sesión de ejemplo (extracto, 2026-10-02; sesión completa en resultados/sesion_interactiva.txt):

    🧑 Tú: ¿Me pueden multar si fumo en el pasillo?

    📎 Top 5 fragmentos recuperados:

      1. [convivencia_privada | Normas de Convivencia, Emisiones de Humo y Ruido]  (convivencia_privada-003)
         La tranquilidad y el descanso de los vecinos configuran una obligación esencial contemplada en ...

      2. [convivencia_privada | Procedimiento Sancionatorio y Régimen de Multas]  (convivencia_privada-004)
         Cualquier transgresión a las obligaciones descritas en el Artículo Octavo o en los artículos ...

      3. [espacios_comunes | Preservación del Libre Tránsito y Prohibición de Ocupación]  (espacios_comunes-005)
      4. [asados | Regulación Exhaustiva sobre Asados y Tipo de Parrillas]  (asados-001)
      5. [convivencia_privada | Destino Exclusivo Habitacional y Prohibición Comercial]  (convivencia_privada-002)

    El top-2 combina la norma (fumar en áreas comunes) y su sanción (multas en UTM).
"""
import sys
import logging
import warnings

# Silencia el aviso de deprecación de langchain-community y los logs HTTP para no ensuciar la conversación.
warnings.filterwarnings("ignore", message=".*langchain-community.*")

from rag_system import RAGSystem

# En modo interactivo solo se muestran advertencias y errores (no los logs INFO de cada llamada HTTP).
logging.getLogger().setLevel(logging.WARNING)

COMANDOS_SALIDA = ("salir", "exit", "quit")
LARGO_FRAGMENTO = 200


# Quita la primera línea (el encabezado de sección, que ya se muestra aparte) y compacta espacios y tabulaciones.
def resumir(contenido: str, largo: int = LARGO_FRAGMENTO) -> str:
    cuerpo = " ".join(contenido.split("\n", 1)[-1].split())
    return cuerpo if len(cuerpo) <= largo else cuerpo[:largo] + "..."


def main() -> None:
    print("⏳ Cargando el sistema (chunks, BM25 y conexión a Pinecone)...")
    try:
        rag = RAGSystem()
    except ValueError as e:
        print(f"❌ {e}\n   Completa tus credenciales en .env (ver .env.example).")
        sys.exit(1)

    print("\n💬 Modo interactivo — consulta el sistema RAG híbrido (BM25 + Pinecone)")
    print(f"   (escribe {', '.join(repr(c) for c in COMANDOS_SALIDA)} o Ctrl+C para terminar)\n")

    while True:
        try:
            pregunta = input("🧑 Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        # Si la entrada viene de un archivo o pipe (no de un teclado), se repite la pregunta para que quede en la salida.
        if not sys.stdin.isatty():
            print(pregunta)

        if pregunta.lower() in COMANDOS_SALIDA:
            break
        if not pregunta:
            continue

        resultados = rag.retrieve(pregunta)
        if not resultados:
            print("⚠️ No se recuperó ningún fragmento.\n")
            continue

        print(f"\n📎 Top {len(resultados)} fragmentos recuperados:")
        for i, doc in enumerate(resultados, start=1):
            m = doc.metadata
            print(f"\n  {i}. [{m['doc_id']} | {m['seccion']}]  ({m['chunk_id']})")
            print(f"     {resumir(doc.page_content)}")
        print("-" * 80)

    print("\n👋 Listo, terminamos la sesión.")


if __name__ == "__main__":
    main()
