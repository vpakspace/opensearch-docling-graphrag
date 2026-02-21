"""Markdown-aware text chunker.

Splitting strategy (in priority order):

1. Markdown headers (``##`` / ``###``) define top-level sections.
2. Within each section, paragraphs (double newline) are accumulated up
   to *chunk_size* characters.
3. When a single paragraph exceeds *chunk_size*, it is further split on
   sentence boundaries.
4. Tables (blocks where every non-empty line starts with ``|``) are kept
   as atomic units and never split across chunk boundaries.

Each :class:`~opensearch_graphrag.models.Chunk` receives:

* An ``id`` — first 8 hex characters of the MD5 hash of its text.
* A ``chunk_index`` stored in ``metadata`` (assigned after all chunks for
  the document are collected so indices are globally sequential).
"""

from __future__ import annotations

import hashlib
import re

from opensearch_graphrag.config import get_settings
from opensearch_graphrag.models import Chunk

# ── Public API ───────────────────────────────────────────────────


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Split *text* into semantically coherent :class:`~opensearch_graphrag.models.Chunk` objects.

    Args:
        text:         Input text (markdown or plain).
        chunk_size:   Maximum character count per chunk.  Defaults to
                      ``cfg.chunking.chunk_size`` from
                      :func:`~opensearch_graphrag.config.get_settings`.
        chunk_overlap: Characters of overlap between adjacent chunks when a
                      section is split mid-paragraph.  Defaults to
                      ``cfg.chunking.chunk_overlap``.

    Returns:
        List of :class:`~opensearch_graphrag.models.Chunk` objects with
        auto-generated IDs and sequential ``chunk_index`` values in
        ``metadata``.  Returns an empty list for blank input.

    Example::

        chunks = chunk_text("# Section\\n\\nSome content here.")
        for c in chunks:
            print(c.id, c.text[:40])
    """
    cfg = get_settings()
    if chunk_size is None:
        chunk_size = cfg.chunking.chunk_size
    if chunk_overlap is None:
        chunk_overlap = cfg.chunking.chunk_overlap

    if not text.strip():
        return []

    chunks: list[Chunk] = []
    for section_title, section_content in _split_by_headers(text):
        chunks.extend(_chunk_section(section_content, chunk_size, chunk_overlap, section_title))

    # Assign global sequential indices after all chunks are collected
    for i, chunk in enumerate(chunks):
        chunk.chunk_index = i
        chunk.metadata["chunk_index"] = i

    return chunks


# ── Internal helpers ─────────────────────────────────────────────


def _split_by_headers(text: str) -> list[tuple[str, str]]:
    """Split *text* on markdown ``##`` / ``###`` headers.

    Returns a list of ``(header_title, section_body)`` pairs.  Content
    before the first header is emitted with an empty title.
    """
    header_re = re.compile(r"^(#{2,3})\s+(.+)$", re.MULTILINE)
    sections: list[tuple[str, str]] = []
    current_title = ""
    current_lines: list[str] = []

    for line in text.split("\n"):
        match = header_re.match(line)
        if match:
            if current_lines:
                sections.append((current_title, "\n".join(current_lines)))
            current_title = match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_title, "\n".join(current_lines)))

    # Guarantee at least one entry so callers always have something to iterate
    if not sections:
        sections.append(("", text))

    return sections


def _chunk_section(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    section_title: str,
) -> list[Chunk]:
    """Convert a single header section into one or more :class:`~opensearch_graphrag.models.Chunk` objects.

    Tables (every non-empty line starts with ``|``) are emitted as a
    single atomic chunk without further splitting.
    """
    if not text.strip():
        return []

    # ── Table detection — keep entire block atomic ────────────────
    non_empty_lines = [ln for ln in text.split("\n") if ln.strip()]
    if non_empty_lines and all(ln.strip().startswith("|") for ln in non_empty_lines):
        return [_create_chunk(text, section_title)]

    # ── Paragraph accumulation ────────────────────────────────────
    paragraphs = text.split("\n\n")
    chunks: list[Chunk] = []
    current: str = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}".strip() if current else para
        else:
            if current:
                chunks.append(_create_chunk(current, section_title))
                overlap_prefix = current[-chunk_overlap:] if chunk_overlap > 0 else ""
                current = f"{overlap_prefix}\n\n{para}".strip() if overlap_prefix else para
            else:
                # Single paragraph exceeds chunk_size — split by sentences
                for sentence_chunk in _split_by_sentences(para, chunk_size, chunk_overlap):
                    chunks.append(_create_chunk(sentence_chunk, section_title))
                current = ""

    if current:
        chunks.append(_create_chunk(current, section_title))

    return chunks


def _split_by_sentences(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split *text* on sentence boundaries when a paragraph exceeds *chunk_size*.

    Uses a simple regex that splits after ``.``, ``!``, or ``?`` followed
    by whitespace so that punctuation stays with its sentence.
    """
    # Split while keeping the delimiter attached to the preceding part
    sentence_re = re.compile(r"([.!?]+\s+)")
    parts = sentence_re.split(text)

    # Recombine: parts alternate between sentence bodies and delimiters
    sentences: list[str] = []
    current = ""
    for i, part in enumerate(parts):
        current += part
        if i % 2 == 1:  # delimiter part — sentence is complete
            sentences.append(current)
            current = ""
    if current:
        sentences.append(current)

    result: list[str] = []
    current_chunk = ""

    for sent in sentences:
        if len(current_chunk) + len(sent) <= chunk_size:
            current_chunk += sent
        else:
            if current_chunk:
                result.append(current_chunk)
                overlap_prefix = current_chunk[-chunk_overlap:] if chunk_overlap > 0 else ""
                current_chunk = overlap_prefix + sent
            else:
                # Single sentence exceeds chunk_size — emit as-is
                result.append(sent)
                current_chunk = ""

    if current_chunk:
        result.append(current_chunk)

    return result


def _create_chunk(text: str, section_title: str) -> Chunk:
    """Create a :class:`~opensearch_graphrag.models.Chunk` with an auto-generated MD5-based ID.

    The chunk ``id`` is the first 8 hex characters of ``MD5(text.encode())``.
    ``section_title`` is stored in ``metadata["section_title"]`` when non-empty.
    """
    chunk_id = hashlib.md5(text.encode()).hexdigest()[:8]
    metadata: dict = {}
    if section_title:
        metadata["section_title"] = section_title
    return Chunk(
        id=chunk_id,
        text=text,
        metadata=metadata,
    )
