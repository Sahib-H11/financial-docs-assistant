import os
from dotenv import load_dotenv

load_dotenv()  # reads the .env file into environment variables

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")   # "openai" or "ollama"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "financial_docs"

CHUNK_SIZE = 1000     # characters per chunk
CHUNK_OVERLAP = 150   # characters shared by neighbouring chunks
TOP_K = 4             # how many chunks we send to the LLM