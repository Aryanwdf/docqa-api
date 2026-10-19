import hashlib

import pytest
from chromadb.api.types import EmbeddingFunction

from app.ingest import Chunk
from app.store import ChunkStore


class WordHashEmbedding(EmbeddingFunction):
    """Deterministic bag-of-words embedding, so tests need no model download."""

    DIM = 64

    def __init__(self) -> None:
        pass

    def __call__(self, input):
        vectors = []
        for text in input:
            v = [0.0] * self.DIM
            for word in text.lower().split():
                v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.DIM] += 1.0
            vectors.append(v)
        return vectors

    @staticmethod
    def name() -> str:
        return "word-hash"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "WordHashEmbedding":
        return WordHashEmbedding()


@pytest.fixture
def store():
    return ChunkStore(embedding_function=WordHashEmbedding())


CHUNKS = [
    Chunk("photosynthesis converts light energy into chemical energy", 3, 0),
    Chunk("the mitochondrion is the site of cellular respiration and ATP", 7, 1),
    Chunk("cell walls in plants are made primarily of cellulose", 12, 2),
]


def test_search_returns_the_relevant_page(store):
    store.add("doc1", CHUNKS)
    hits = store.search("doc1", "mitochondrion respiration ATP", k=1)
    assert hits[0].page == 7


def test_hits_come_back_ranked(store):
    store.add("doc1", CHUNKS)
    scores = [h.score for h in store.search("doc1", "plants cellulose", k=3)]
    assert scores == sorted(scores, reverse=True)


def test_unknown_document_is_empty(store):
    assert store.search("never-indexed", "anything") == []


def test_exists_reflects_indexing(store):
    assert store.exists("doc2") is False
    store.add("doc2", CHUNKS)
    assert store.exists("doc2") is True


def test_adding_nothing_is_harmless(store):
    store.add("doc3", [])
    assert store.search("doc3", "anything") == []