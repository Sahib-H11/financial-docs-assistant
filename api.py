import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from ingest import ingest_pdf, list_documents
from rag import answer_question

app = FastAPI(title="Financial Document Assistant")

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class Question(BaseModel):  # describes the JSON body of /ask
    question: str
    document: str | None = None  # optional: search only in this PDF


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/documents")
def documents():
    return {"documents": list_documents()}


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    path = UPLOAD_DIR / Path(file.filename).name
    with path.open("wb") as f:
        shutil.copyfileobj(file.file, f)  # save the uploaded file to disk

    try:
        n_chunks = ingest_pdf(str(path))
    except ValueError as e:  # e.g. scanned PDF without text
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:  # any other problem, e.g. a damaged or protected PDF
        raise HTTPException(status_code=500, detail=f"Could not process {file.filename}: {e}")

    return {"filename": file.filename, "chunks": n_chunks}


@app.post("/ask")
def ask(q: Question):
    if not q.question.strip():
        raise HTTPException(status_code=400, detail="Question is empty.")
    return answer_question(q.question, document=q.document)
