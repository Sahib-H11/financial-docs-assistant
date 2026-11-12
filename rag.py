import json
import math
import re
from collections import Counter
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

import config
from ingest import get_vectorstore


def get_llm():
    """Return the chat model chosen in .env (cloud or local)."""
    if config.LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=config.OLLAMA_MODEL, temperature=0)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=config.OPENAI_MODEL)


PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an assistant for financial and tax documents. "
     "Answer ONLY with information from the context below. "
     "After each statement, cite the source as [file, p. X]. "
     "If the answer is not in the context, say that you cannot find it in the documents, and do not cite any sources. "
     "Answer in the same language as the question.\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])


def format_context(docs) -> str:
    """Put each chunk under a label with its file and page, so the LLM can cite it."""
    return "\n\n---\n\n".join(
        f"[{d.metadata['source']}, p. {d.metadata['page']}]\n{d.page_content}" for d in docs
    )


_page_cache = {}


def load_pages(filename: str) -> dict:
    """Full page texts of one PDF, saved by ingest.py (empty if not available)."""
    path = Path(config.PAGES_DIR) / f"{filename}.json"
    if not path.exists():
        return {}
    key = (filename, path.stat().st_mtime)  # re-read if the PDF was indexed again
    if key not in _page_cache:
        _page_cache[key] = json.loads(path.read_text(encoding="utf-8"))
    return _page_cache[key]


def expand_to_pages(docs) -> list:
    """Small-to-big: replace the retrieved small chunks by their full pages (best-ranked first)."""
    from langchain_core.documents import Document

    pages, seen = [], set()
    for d in docs:
        key = (d.metadata["source"], d.metadata["page"])
        if key in seen:
            continue
        seen.add(key)
        text = load_pages(key[0]).get(str(key[1]), d.page_content)
        pages.append(Document(page_content=text, metadata={"source": key[0], "page": key[1]}))
        if len(pages) == config.MAX_PAGES:
            break
    return pages


# ---------- Keyword search (BM25) ----------

STOPWORDS = set("""
a an the of in on at to for from by as is was were what which how much many did does do and or
its it s this that with per all q1 q2 q3 q4 quarter third first nine months year fiscal 2025 2026
der die das des dem den ein eine im in am an zum zur von vom und oder wie hoch war waren ist welche
welcher welches viel viele hatte hat für bei mit nach quartal geschäftsjahr dritten
""".split())


def tokenize(text: str) -> list[str]:
    """Lower-case words without filler words (and without the dates that appear on every page)."""
    return [w for w in re.findall(r"\w+", text.lower()) if w not in STOPWORDS and not w.isdigit()]


_bm25_cache = {}


def load_chunks(document: str | None):
    """All chunks (text + metadata) of one PDF, or of all PDFs, from the database."""
    where = {"source": document} if document else None
    data = get_vectorstore().get(where=where, include=["documents", "metadatas"])
    return data["documents"], data["metadatas"]


def bm25_search(question: str, document: str | None, k: int, k1: float = 1.5, b: float = 0.75):
    """Classic BM25 keyword ranking over the chunks. Returns the k best (text, metadata) pairs."""
    texts, metas = load_chunks(document)
    key = (document, len(texts))
    if key not in _bm25_cache:  # build the word statistics once per document set
        docs = [Counter(tokenize(t)) for t in texts]
        df = Counter(w for d in docs for w in d)
        avgdl = sum(sum(d.values()) for d in docs) / max(len(docs), 1)
        _bm25_cache[key] = (docs, df, avgdl)
    docs, df, avgdl = _bm25_cache[key]
    n = len(docs)
    scores = []
    for i, d in enumerate(docs):
        dl = sum(d.values())
        s = 0.0
        for w in set(tokenize(question)):
            if w in d:
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * d[w] * (k1 + 1) / (d[w] + k1 * (1 - b + b * dl / avgdl))
        scores.append(s)
    best = sorted(range(n), key=lambda i: scores[i], reverse=True)[:k]
    return [(texts[i], metas[i]) for i in best if scores[i] > 0]


def hybrid_search(question: str, document: str | None, k: int) -> list:
    """Reciprocal Rank Fusion: merge the meaning-search and keyword-search rankings."""
    from langchain_core.documents import Document

    search_kwargs = {"k": k}
    if document:
        search_kwargs["filter"] = {"source": document}
    dense = get_vectorstore().as_retriever(search_kwargs=search_kwargs).invoke(question)
    keyword = bm25_search(question, document, k)

    fused, docs_by_key = Counter(), {}
    for rank, d in enumerate(dense):
        key = (d.metadata["source"], d.metadata["page"], d.page_content[:80])
        fused[key] += 1 / (60 + rank)
        docs_by_key[key] = d
    for rank, (text, meta) in enumerate(keyword):
        key = (meta["source"], meta["page"], text[:80])
        fused[key] += 1 / (60 + rank)
        docs_by_key.setdefault(key, Document(page_content=text, metadata=meta))
    return [docs_by_key[key] for key, _ in fused.most_common(k)]


def answer_question(question: str, top_k: int | None = None, document: str | None = None) -> dict:
    k = top_k or config.TOP_K  # default from config.py; evaluate.py can try other values
    if config.HYBRID:
        docs = hybrid_search(question, document, k)  # 1. meaning + keyword search
    else:
        search_kwargs = {"k": k}
        if document:  # only search inside this one PDF, so reports don't compete
            search_kwargs["filter"] = {"source": document}
        docs = get_vectorstore().as_retriever(search_kwargs=search_kwargs).invoke(question)
    if config.PAGE_CONTEXT:
        docs = expand_to_pages(docs)  # 1b. give the LLM the whole pages

    if not docs:
        return {"answer": "No documents indexed yet. Please upload a PDF first.", "sources": []}

    chain = PROMPT | get_llm()  # 2. LangChain chain: prompt -> model
    result = chain.invoke({"context": format_context(docs), "question": question})  # 3. generate

    sources = [
        {"file": d.metadata["source"], "page": d.metadata["page"], "excerpt": d.page_content[:300]}
        for d in docs
    ]
    return {"answer": result.content, "sources": sources}


if __name__ == "__main__":
    import sys

    out = answer_question(" ".join(sys.argv[1:]))
    print(out["answer"], "\n")
    for s in out["sources"]:
        print(f"- {s['file']}, page {s['page']}")