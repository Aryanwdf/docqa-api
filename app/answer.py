"""Turn retrieved passages into an answer that cites its pages.

The LLM step is optional: without OPENAI_API_KEY the service still returns the
retrieved passages and their pages, which is the part that has to be right
before any model is involved.
"""

import os

from app.store import Hit

SYSTEM_PROMPT = (
    "You answer questions strictly from the numbered sources provided. "
    "Cite the sources you use as [1], [2] and so on. "
    "If the sources do not contain the answer, say so plainly instead of guessing."
)


def build_prompt(question: str, hits: list[Hit]) -> str:
    sources = "\n\n".join(
        f"[{i}] (page {h.page})\n{h.text}" for i, h in enumerate(hits, start=1)
    )
    return f"Sources:\n\n{sources}\n\nQuestion: {question}"


def synthesise(question: str, hits: list[Hit]) -> str | None:
    """Return an answer, or None when no API key is configured."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not hits:
        return None

    from openai import OpenAI

    response = OpenAI(api_key=api_key).chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, hits)},
        ],
    )
    return response.choices[0].message.content