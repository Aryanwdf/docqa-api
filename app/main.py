from fastapi import FastAPI, File, HTTPException, UploadFile

from app.ingest import chunk_pages, extract_pages

app = FastAPI(
    title="docqa-api",
    description="Ask questions about a PDF and get answers with page-level citations.",
)


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

    return {
        "filename": file.filename,
        "pages": len(pages),
        "chunks": len(chunks),
        "preview": [{"page": c.page, "text": c.text[:120]} for c in chunks[:3]],
    }