import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.answer import synthesise
from app.ingest import chunk_pages, extract_pages
from app.store import ChunkStore

app = FastAPI(
    title="docqa-api",
    description="Ask questions about a PDF and get answers with page-level citations.",
)
store = ChunkStore()


class Question(BaseModel):
    document_id: str
    question: str = Field(min_length=3)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents")
async def ingest(file: UploadFile = File(...)) -> dict:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "only PDF is supported")

    pages = extract_pages(await file.read())
    chunks = chunk_pages(pages)
    if not chunks:
        raise HTTPException(422, "no extractable text — is this a scanned PDF?")

    document_id = uuid.uuid4().hex[:12]
    store.add(document_id, chunks)

    return {
        "document_id": document_id,
        "filename": file.filename,
        "pages": len(pages),
        "chunks": len(chunks),
    }


@app.post("/ask")
def ask(payload: Question) -> dict:
    if not store.exists(payload.document_id):
        raise HTTPException(404, "unknown document_id — upload the PDF first")

    hits = store.search(payload.document_id, payload.question)
    if not hits:
        raise HTTPException(404, "no passages indexed for this document")

    return {
        "question": payload.question,
        "answer": synthesise(payload.question, hits),
        "sources": [
            {"page": h.page, "score": round(h.score, 3), "text": h.text}
            for h in hits
        ],
    }