"""Vector store for document chunks, backed by ChromaDB.

Embeddings are computed locally, so the service needs no API key and anyone
who clones this repository gets the same retrieval behaviour.
"""

import os
from dataclasses import dataclass
from functools import lru_cache

import chromadb
from chromadb.utils import embedding_functions

from app.ingest import Chunk

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache(maxsize=1)
def default_embedding_function():
    """Multilingual by default, because the documents this was built for are German.

    Chroma's own default (all-MiniLM-L6-v2) is English-trained and measurably
    weaker on German technical text — see "Retrieval quality" in the README.
    Falls back to it when sentence-transformers is not installed, so the
    service still starts.
    """
    try:
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL),
            normalize_embeddings=True,
        )
    except ImportError:
        return embedding_functions.DefaultEmbeddingFunction()


@dataclass
class Hit:
    text: str
    page: int
    score: float


class ChunkStore:
    def __init__(self, path: str | None = None, embedding_function=None) -> None:
        # A path makes the index survive restarts; without one it stays in memory.
        self._client = (
            chromadb.PersistentClient(path=path) if path else chromadb.EphemeralClient()
        )
        # Injectable so tests can run without downloading a model.
        self._embed = embedding_function or default_embedding_function()

    def _collection(self, document_id: str):
        return self._client.get_or_create_collection(
            name=f"doc_{document_id}",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._embed,
        )

    def add(self, document_id: str, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self._collection(document_id).add(
            ids=[f"{document_id}:{c.index}" for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{"page": c.page} for c in chunks],
        )

    def search(self, document_id: str, question: str, k: int = 6) -> list[Hit]:
        collection = self._collection(document_id)
        if collection.count() == 0:
            return []

        result = collection.query(
            query_texts=[question],
            n_results=min(k, collection.count()),
        )
        return [
            Hit(text=doc, page=int(meta["page"]), score=1.0 - dist)
            for doc, meta, dist in zip(
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
            )
        ]

    def exists(self, document_id: str) -> bool:
        return f"doc_{document_id}" in [c.name for c in self._client.list_collections()]