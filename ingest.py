from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import logging
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

loader = DirectoryLoader("data", glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
documentos_crudos = loader.load()

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP
)

chunks = splitter.split_documents(documentos_crudos)

for i, chunk in enumerate(chunks):
    nombre_archivo = os.path.basename(chunk.metadata["source"])
    chunk.metadata["source"] = nombre_archivo
    chunk.metadata["categoria"] = nombre_archivo.replace(".txt", "").replace("_", " ")
    chunk.metadata["chunk_id"] = i
    logging.info(f"Chunk {i} procesado desde el archivo {nombre_archivo}.")
    
logging.info(f"Total de chunks procesados: {len(chunks)}.")
print("Metadata de ejemplo:", chunks[0].metadata if chunks else {})