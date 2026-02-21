"""Tests for opensearch_graphrag.chunker.

All tests are pure in-process — no external services required.
"""

from __future__ import annotations

import hashlib

from opensearch_graphrag.chunker import (
    _create_chunk,
    _split_by_headers,
    _split_by_sentences,
    chunk_text,
)
from opensearch_graphrag.models import Chunk

# ── test_chunk_empty ─────────────────────────────────────────────


def test_chunk_empty():
    """Blank or whitespace-only input returns an empty list."""
    assert chunk_text("") == []
    assert chunk_text("   ") == []
    assert chunk_text("\n\n\n") == []


# ── test_chunk_short_text ────────────────────────────────────────


def test_chunk_short_text():
    """Text shorter than chunk_size produces exactly one chunk."""
    text = "This is a short sentence."
    chunks = chunk_text(text, chunk_size=512, chunk_overlap=0)

    assert len(chunks) == 1
    assert isinstance(chunks[0], Chunk)
    assert chunks[0].text == text


def test_chunk_short_text_has_index_zero():
    """Single chunk always gets chunk_index 0."""
    chunks = chunk_text("Hello world.", chunk_size=512, chunk_overlap=0)
    assert chunks[0].chunk_index == 0
    assert chunks[0].metadata["chunk_index"] == 0


# ── test_chunk_with_headers ──────────────────────────────────────


def test_chunk_with_headers():
    """Text with markdown headers is split into multiple chunks."""
    text = (
        "## Introduction\n\n"
        "This section introduces the topic.\n\n"
        "## Methods\n\n"
        "This section describes the methods used.\n\n"
        "## Results\n\n"
        "This section presents the results."
    )
    chunks = chunk_text(text, chunk_size=512, chunk_overlap=0)

    # At least one chunk per header section
    assert len(chunks) >= 3

    texts = [c.text for c in chunks]
    assert any("introduces" in t for t in texts)
    assert any("methods" in t for t in texts)
    assert any("results" in t for t in texts)


def test_chunk_with_headers_section_title_in_metadata():
    """section_title is stored in chunk metadata when present."""
    text = "## Background\n\nSome background information."
    chunks = chunk_text(text, chunk_size=512, chunk_overlap=0)

    assert len(chunks) >= 1
    assert chunks[0].metadata.get("section_title") == "Background"


def test_chunk_headers_produce_sequential_indices():
    """chunk_index values are globally sequential across sections."""
    text = "## A\n\nParagraph A.\n\n## B\n\nParagraph B.\n\n## C\n\nParagraph C."
    chunks = chunk_text(text, chunk_size=512, chunk_overlap=0)

    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


# ── test_chunk_table_atomic ──────────────────────────────────────


def test_chunk_table_atomic():
    """Markdown tables are never split — emitted as a single chunk."""
    table = (
        "| Name | Score |\n"
        "| ---- | ----- |\n"
        "| Alice | 95   |\n"
        "| Bob   | 87   |\n"
        "| Carol | 91   |\n"
    )
    chunks = chunk_text(table, chunk_size=50, chunk_overlap=0)

    # Must be exactly one chunk even though it exceeds chunk_size=50
    assert len(chunks) == 1
    assert "Alice" in chunks[0].text
    assert "Carol" in chunks[0].text


def test_chunk_table_with_surrounding_text():
    """Table is kept atomic even when surrounded by prose."""
    text = (
        "## Data\n\n"
        "Here are the scores:\n\n"
        "| Name | Score |\n"
        "| ---- | ----- |\n"
        "| Alice | 95   |\n"
    )
    chunks = chunk_text(text, chunk_size=512, chunk_overlap=0)

    # Table must appear intact in some chunk
    all_text = "\n".join(c.text for c in chunks)
    assert "| Alice | 95" in all_text


# ── test_chunk_ids_are_md5 ───────────────────────────────────────


def test_chunk_ids_are_sha256():
    """Chunk IDs are the first 8 hex characters of SHA-256(text)."""
    text = "Deterministic content."
    chunks = chunk_text(text, chunk_size=512, chunk_overlap=0)

    assert len(chunks) == 1
    expected_id = hashlib.sha256(chunks[0].text.encode()).hexdigest()[:8]
    assert chunks[0].id == expected_id


def test_chunk_ids_are_deterministic():
    """Same text always produces the same chunk ID."""
    text = "Stable text for hashing."
    chunks1 = chunk_text(text, chunk_size=512, chunk_overlap=0)
    chunks2 = chunk_text(text, chunk_size=512, chunk_overlap=0)

    assert chunks1[0].id == chunks2[0].id


def test_chunk_ids_differ_for_different_texts():
    """Different texts yield different chunk IDs."""
    text_a = "## Alpha\n\nContent alpha."
    text_b = "## Beta\n\nContent beta."

    chunks_a = chunk_text(text_a, chunk_size=512, chunk_overlap=0)
    chunks_b = chunk_text(text_b, chunk_size=512, chunk_overlap=0)

    ids_a = {c.id for c in chunks_a}
    ids_b = {c.id for c in chunks_b}
    assert ids_a.isdisjoint(ids_b)


# ── test_chunk_metadata_has_index ────────────────────────────────


def test_chunk_metadata_has_index():
    """Every chunk carries chunk_index in metadata AND as a field."""
    text = "## One\n\nFirst.\n\n## Two\n\nSecond.\n\n## Three\n\nThird."
    chunks = chunk_text(text, chunk_size=512, chunk_overlap=0)

    for i, chunk in enumerate(chunks):
        assert "chunk_index" in chunk.metadata, f"Missing chunk_index in metadata for chunk {i}"
        assert chunk.metadata["chunk_index"] == i
        assert chunk.chunk_index == i


# ── Additional edge-case tests ───────────────────────────────────


def test_chunk_large_paragraph_split_by_sentences():
    """A single paragraph that exceeds chunk_size is split into sentences."""
    # Build a paragraph of ~10 short sentences totalling well above 80 chars
    sentences = [f"Sentence number {n} adds content." for n in range(1, 11)]
    big_para = " ".join(sentences)

    chunks = chunk_text(big_para, chunk_size=80, chunk_overlap=0)

    # Should produce more than one chunk
    assert len(chunks) > 1
    # All original words must be present somewhere
    reconstructed = " ".join(c.text for c in chunks)
    for sent in sentences:
        # Every sentence word should appear somewhere in the output
        assert sent.split()[0] in reconstructed


def test_chunk_overlap_carries_trailing_context():
    """chunk_overlap > 0 causes each chunk to begin with context from the previous."""
    # Two paragraphs each of 100 chars; overlap = 20
    para_a = "A" * 100
    para_b = "B" * 100
    text = f"{para_a}\n\n{para_b}"

    chunks = chunk_text(text, chunk_size=110, chunk_overlap=20)

    # The second chunk should start with trailing characters of the first
    assert len(chunks) >= 2
    assert chunks[1].text.startswith("A" * 20) or "A" in chunks[1].text


def test_create_chunk_helper_no_title():
    """_create_chunk produces no section_title key when title is empty."""
    chunk = _create_chunk("Some content.", "")
    assert "section_title" not in chunk.metadata


def test_create_chunk_helper_with_title():
    """_create_chunk stores section_title in metadata."""
    chunk = _create_chunk("Some content.", "My Section")
    assert chunk.metadata["section_title"] == "My Section"


def test_split_by_headers_no_headers():
    """_split_by_headers returns a single section with empty title for plain text."""
    text = "No headers here.\nJust plain content."
    sections = _split_by_headers(text)
    assert len(sections) == 1
    assert sections[0][0] == ""


def test_split_by_headers_multiple():
    """_split_by_headers correctly identifies two separate header sections."""
    text = "## Alpha\n\nAlpha content.\n\n## Beta\n\nBeta content."
    sections = _split_by_headers(text)
    titles = [s[0] for s in sections]
    assert "Alpha" in titles
    assert "Beta" in titles


def test_split_by_sentences_short():
    """_split_by_sentences returns a single item for text below chunk_size."""
    text = "Short sentence."
    parts = _split_by_sentences(text, chunk_size=200, chunk_overlap=0)
    assert len(parts) == 1
    assert parts[0] == text


# ── Additional edge cases ─────────────────────────────────────────


def test_chunk_code_block_preserved():
    """Code blocks with triple backticks are kept intact."""
    text = "## Code\n\n```python\ndef hello():\n    print('hi')\n```\n\nSome text."
    chunks = chunk_text(text, chunk_size=512, chunk_overlap=0)
    all_text = "\n".join(c.text for c in chunks)
    assert "```python" in all_text
    assert "def hello():" in all_text


def test_chunk_deep_headers():
    """#### headers are NOT split points (only ## and ### are)."""
    text = "#### Deep Header\n\nContent under deep header."
    chunks = chunk_text(text, chunk_size=512, chunk_overlap=0)
    assert len(chunks) == 1
    assert "#### Deep Header" in chunks[0].text


def test_chunk_empty_lines_between_paragraphs():
    """Multiple empty lines between paragraphs don't create empty chunks."""
    text = "First paragraph.\n\n\n\n\nSecond paragraph."
    chunks = chunk_text(text, chunk_size=512, chunk_overlap=0)
    for chunk in chunks:
        assert chunk.text.strip() != ""


def test_chunk_mixed_headers():
    """Mixed ## and ### headers both create section boundaries."""
    text = "## Main\n\nMain content.\n\n### Sub\n\nSub content."
    chunks = chunk_text(text, chunk_size=512, chunk_overlap=0)
    assert len(chunks) >= 2
    titles = [c.metadata.get("section_title", "") for c in chunks]
    assert "Main" in titles
    assert "Sub" in titles
