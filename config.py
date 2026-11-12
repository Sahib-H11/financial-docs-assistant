import os
from dotenv import load_dotenv

load_dotenv()  # reads the .env file into environment variables

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")   # "openai" or "ollama"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "financial_docs_small"  # small-to-big index (v1.2); old index: "financial_docs"
PAGES_DIR = "data/pages"  # full text of every page, used as context for the LLM

# Small chunks for SEARCH: the embedding model reads at most 128 tokens (~500 characters),
# so larger chunks would be cut off before they are embedded.
CHUNK_SIZE = 500      # characters per chunk
CHUNK_OVERLAP = 100   # characters shared by neighbouring chunks
# Whole pages for ANSWERING: the LLM gets the full page of each hit, including table headings.
PAGE_CONTEXT = True
MAX_PAGES = 4         # at most this many different pages are sent to the LLM
TOP_K = 8             # how many chunks we retrieve
# Hybrid search: combine meaning search (embeddings) with keyword search (BM25),
# so exact terms in tables ("Konzernumsatz", "net debt") are found reliably.
HYBRID = True