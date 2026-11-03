from __future__ import annotations

"""Shared text formatting for chunks.

The embedding input (ingest) and the reranker input (rag_search) must use the
same heading-augmented format, so both call with_heading().
"""


def with_heading(heading_path: str, text: str) -> str:
    """Prepend the heading path to the chunk text; return text alone if no heading."""
    return f"{heading_path}\n{text}" if heading_path else text


def reranker_passages(candidates: list, use_heading: bool) -> list[str]:
    """Build reranker passages from search candidates (heading_path + text objects)."""
    if use_heading:
        return [with_heading(c.heading_path, c.text) for c in candidates]
    return [c.text for c in candidates]
