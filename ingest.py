from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

_embeddings = None  # loaded once, then reused (loading takes a few seconds)


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    return _embeddings


def get_vectorstore():
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=config.CHROMA_DIR,  # saved on disk, survives restarts
    )


def ingest_pdf(pdf_path: str) -> int:
    """Read a PDF, split it into chunks and store them. Returns the number of chunks."""
    filename = Path(pdf_path).name
    pages = PyPDFLoader(pdf_path).load()  # one Document per page

    for page in pages:
        page.metadata["source"] = filename
        page.metadata["page"] = page.metadata.get("page", 0) + 1  # PyPDF counts from 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = [c for c in splitter.split_documents(pages) if c.page_content.strip()]

    if not chunks:
        raise ValueError(f"No text found in {filename}. It may be a scanned PDF (needs OCR).")

    # Fixed IDs: uploading the same file again overwrites instead of duplicating
    ids = [f"{filename}-{i}" for i in range(len(chunks))]
    get_vectorstore().add_documents(chunks, ids=ids)
    return len(chunks)


if __name__ == "__main__":
    import sys

    for path in sys.argv[1:]:
        print(f"{path}: {ingest_pdf(path)} chunks stored")