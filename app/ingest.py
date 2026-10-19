import io
from dataclasses import dataclass
from pypdf import PdfReader


@dataclass
class Chunk:
    text: str
    page: int
    index: int


def extract_pages(data: bytes) -> list[str]:
    """Return the text of each page, in order. Empty pages stay empty."""
    reader = PdfReader(io.BytesIO(data))
    return [(p.extract_text() or "").strip() for p in reader.pages]


def chunk_pages(pages: list[str], size: int = 900, overlap: int = 150) -> list[Chunk]:
    """Split each page into overlapping chunks, keeping the source page number.

    Overlap keeps sentences that straddle a chunk boundary retrievable from
    both sides, which matters once these chunks are embedded.
    """
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    chunks: list[Chunk] = []
    for page_no, text in enumerate(pages, start=1):
        start = 0
        while start < len(text):
            end = start + size
            chunks.append(Chunk(text[start:end], page_no, len(chunks)))
            if end >= len(text):
                break
            start = end - overlap
    return chunks