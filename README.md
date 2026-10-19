# docqa-api

Ask questions about a PDF and get answers with **page-level citations**.

Most document Q&A examples return an answer and leave you guessing where it
came from. This one carries the source page through the whole pipeline, so every
answer points back at the pages it was drawn from.

## Status

| | |
|---|---|
| ✅ | PDF ingestion, page-aware chunking |
| ✅ | Local embeddings + retrieval (ChromaDB) |
| ✅ | `POST /ask` — ranked passages with page numbers |
| ✅ | Optional LLM answer synthesis with `[n]` citations |
| 🔜 | SSE streaming, Dockerfile |

## How it works

```
PDF ──► extract per page ──► chunk (900/150, page kept)
                                      │
                                      ▼
                            ChromaDB  (local embeddings)
                                      │
            question ──► retrieve top-k ──► LLM ──► answer + [n] pages
```

## Design notes

**Page numbers are carried, not reconstructed.** Each chunk stores its source
page at extraction time. Recovering that later from an embedding is guesswork.

**Chunks overlap by default (900 / 150).** A sentence that straddles a boundary
stays retrievable from both sides. Without overlap, the passage that answers the
question is often the one split in half.

**The LLM step is optional.** Without `OPENAI_API_KEY` the service still returns
the retrieved passages and their pages. Retrieval has to be right before a model
is involved, so it is worth being able to inspect it on its own.

**The embedding function is injectable.** Tests pass a deterministic
bag-of-words embedding, so the suite runs in under a second and needs no model
download.

**Empty pages produce no chunks.** Scanned PDFs extract as empty text, so ingest
returns `422` rather than silently indexing nothing.

## Retrieval quality

Measured on a 154-page German lecture script (407 chunks), asking
*"Was ist eine Gruppe?"* — where the correct answer is a formal definition on
page 47.

| | Pages returned | Top score | Definition found |
|---|---|---|---|
| `all-MiniLM-L6-v2`, k=4 | 49, 49, 48, 48 | 0.446 | ✗ |
| `paraphrase-multilingual-MiniLM-L12-v2`, k=6 | 49, **47**, 16, **47**, 50, 50 | 0.541 | ✓ (rank 2 and 4) |

Chroma's default embedding model is English-trained, and on German technical
prose it returned the right *section* but not the definition itself. Switching to
a multilingual model and widening k fixed that.

Two caveats worth stating: both changes were made at once, so the gain cannot be
attributed to the model alone; and rank 3 in the second run is a false positive
from an unrelated chapter that matched on the shared words *Verknüpfung* and
*Assoziativität*. Precision at the top of the list is the next thing to improve.

Set `EMBEDDING_MODEL` to use a different model.

## Running it

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Interactive API docs: <http://localhost:8000/docs>

The first request downloads the embedding model (~450 MB) once.

## Example

```bash
curl -F "file=@lecture-notes.pdf" localhost:8000/documents
# {"document_id": "90ca2afb718d", "filename": "...", "pages": 154, "chunks": 407}

curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"document_id": "90ca2afb718d", "question": "Was ist eine Gruppe?"}'
```

```json
{
  "question": "Was ist eine Gruppe?",
  "answer": null,
  "sources": [
    { "page": 47, "score": 0.482, "text": "Definition 18: Gruppe ..." }
  ]
}
```

`answer` is `null` unless `OPENAI_API_KEY` is set.

## Tests

```bash
pytest -q
```

Nine tests covering page attribution, chunk overlap, empty-page handling,
invalid chunking parameters, retrieval ranking, and unknown documents.

## Stack

Python · FastAPI · ChromaDB · sentence-transformers · pypdf · pytest
