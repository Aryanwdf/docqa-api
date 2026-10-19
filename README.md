# docqa-api

Ask questions about a PDF and get answers with **page-level citations**.

Most document Q&A examples return an answer and leave you guessing where it
came from. This one tracks the source page through the whole pipeline, so every
answer can point back at the pages it was drawn from.

## Status

| | |
|---|---|
| ✅ | PDF ingestion, page-aware chunking, tests |
| 🔜 | Embeddings + retrieval (ChromaDB) |
| 🔜 | `POST /ask` — answers with page citations |
| 🔜 | SSE streaming, Dockerfile |

## Design notes

**Page numbers are carried, not reconstructed.** Each chunk stores the page it
came from at extraction time. Recovering that later from an embedding is
guesswork, so the information is kept from the start.

**Chunks overlap by default (900 / 150).** A sentence that straddles a boundary
stays retrievable from both sides. Without overlap, the passage that answers the
question is often the one split in half.

**Empty pages produce no chunks.** Scanned PDFs extract as empty text, so the
ingest endpoint returns `422` rather than silently indexing nothing.

## Running it

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Interactive API docs: <http://localhost:8000/docs>

## Example

```bash
curl -F "file=@lecture-notes.pdf" localhost:8000/documents
```

```json
{
  "filename": "lecture-notes.pdf",
  "pages": 154,
  "chunks": 407,
  "preview": [
    { "page": 1, "text": "Mathematik I für Wirtschaftsinformatik ..." }
  ]
}
```

## Tests

```bash
pytest -q
```

Covers page attribution, chunk overlap, empty-page handling, and invalid
chunking parameters.

## Stack

Python · FastAPI · pypdf · pytest