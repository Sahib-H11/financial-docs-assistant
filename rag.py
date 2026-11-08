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


def answer_question(question: str) -> dict:
    retriever = get_vectorstore().as_retriever(search_kwargs={"k": config.TOP_K})
    docs = retriever.invoke(question)  # 1. retrieve

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