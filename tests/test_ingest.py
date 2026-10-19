import pytest

from app.ingest import chunk_pages


def test_chunks_keep_their_page_number():
    chunks = chunk_pages(["a" * 2000, "b" * 500], size=900, overlap=150)
    assert {c.page for c in chunks} == {1, 2}
    assert all(c.text[0] == "a" for c in chunks if c.page == 1)
    assert all(c.text[0] == "b" for c in chunks if c.page == 2)


def test_consecutive_chunks_overlap():
    chunks = [c for c in chunk_pages(["x" * 2000], size=900, overlap=150) if c.page == 1]
    assert len(chunks) > 1
    assert chunks[0].text[-150:] == chunks[1].text[:150]


def test_empty_pages_produce_no_chunks():
    assert chunk_pages(["", ""]) == []


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_pages(["text"], size=100, overlap=100)