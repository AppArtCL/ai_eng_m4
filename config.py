import os
import re
import dotenv
import logging
from pydantic import SecretStr
from langchain_openai import OpenAIEmbeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

dotenv.load_dotenv()
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

if not OPENAI_API_KEY:
    logging.error("OPENAI_API_KEY no puede ser None o vacío.")
    raise ValueError("OPENAI_API_KEY no puede ser None o vacío.")
OPENAI_API_KEY_SECRET = SecretStr(OPENAI_API_KEY)

if not PINECONE_API_KEY:
    logging.error("PINECONE_API_KEY no puede ser None o vacío.")
    raise ValueError("PINECONE_API_KEY no puede ser None o vacío.")

if not PINECONE_INDEX_NAME:
    logging.error("PINECONE_INDEX_NAME no puede ser None o vacío.")
    raise ValueError("PINECONE_INDEX_NAME no puede ser None o vacío.")
if not re.fullmatch(r"[a-z0-9-]{1,45}", PINECONE_INDEX_NAME):
    logging.error(f"PINECONE_INDEX_NAME inválido: '{PINECONE_INDEX_NAME}'. Solo se permiten minúsculas, números y guiones (máx. 45 caracteres).")
    raise ValueError(f"PINECONE_INDEX_NAME inválido: '{PINECONE_INDEX_NAME}'. Solo se permiten minúsculas, números y guiones (máx. 45 caracteres).")

DIM = 1536
METRIC = "cosine"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K = 5
NAMESPACE = "consultas-convivencia"
EMBEDDING_MODEL = "text-embedding-3-small"

def get_embeddings():
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY_SECRET,
        dimensions=DIM,
    )