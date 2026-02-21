"""Tests for opensearch_graphrag.loader.

All tests run without a real Docling installation — only .txt and .md
files are exercised (handled by the plain-text fast path), plus error
conditions that do not reach the converter at all.
"""

from __future__ import annotations

import pytest

from opensearch_graphrag.loader import DoclingLoader, DocumentResult, load_file

# ── Helpers ──────────────────────────────────────────────────────


def _write(tmp_path, filename: str, content: str):
    """Write *content* to *tmp_path/filename* and return the Path."""
    p = tmp_path / filename
    p.write_text(content, encoding="utf-8")
    return p


# ── test_load_text_file ──────────────────────────────────────────


def test_load_text_file(tmp_path):
    """.txt files are loaded via the plain-text fast path, no Docling needed."""
    content = "OpenSearch is a distributed search engine."
    path = _write(tmp_path, "doc.txt", content)

    loader = DoclingLoader()
    result = loader.load(path)

    assert isinstance(result, DocumentResult)
    assert result.markdown == content
    assert result.tables == []
    assert result.images == []
    assert result.metadata["format"] == ".txt"
    assert result.metadata["pages"] == 1


# ── test_load_md_file ────────────────────────────────────────────


def test_load_md_file(tmp_path):
    """.md files are loaded via the plain-text fast path."""
    content = "## Section\n\nSome markdown content.\n"
    path = _write(tmp_path, "notes.md", content)

    loader = DoclingLoader()
    result = loader.load(path)

    assert result.markdown == content
    assert result.metadata["format"] == ".md"
    assert result.metadata["pages"] == 1


# ── test_load_nonexistent ────────────────────────────────────────


def test_load_nonexistent(tmp_path):
    """Loading a missing file raises FileNotFoundError."""
    loader = DoclingLoader()
    with pytest.raises(FileNotFoundError, match="File not found"):
        loader.load(tmp_path / "ghost.txt")


# ── test_load_unsupported ────────────────────────────────────────


def test_load_unsupported(tmp_path):
    """Loading an unsupported extension raises ValueError."""
    path = tmp_path / "archive.zip"
    path.write_bytes(b"PK\x03\x04")  # minimal fake zip header

    loader = DoclingLoader()
    with pytest.raises(ValueError, match="Unsupported format"):
        loader.load(path)


# ── test_load_bytes_txt ──────────────────────────────────────────


def test_load_bytes_txt():
    """load_bytes decodes plain .txt bytes without writing to disk."""
    content = "Hello from bytes."
    data = content.encode("utf-8")

    loader = DoclingLoader()
    result = loader.load_bytes(data, "readme.txt")

    assert result.markdown == content
    assert result.metadata["format"] == ".txt"
    assert result.metadata["pages"] == 1


def test_load_bytes_txt_unicode():
    """load_bytes handles UTF-8 multibyte characters correctly."""
    content = "Привет мир — hello world"
    loader = DoclingLoader()
    result = loader.load_bytes(content.encode("utf-8"), "msg.txt")
    assert result.markdown == content


def test_load_bytes_unsupported_raises():
    """load_bytes raises ValueError for unsupported extension."""
    loader = DoclingLoader()
    with pytest.raises(ValueError, match="Unsupported format"):
        loader.load_bytes(b"data", "file.csv")


# ── test_load_file_convenience ───────────────────────────────────


def test_load_file_convenience(tmp_path):
    """load_file() returns the markdown string for plain text files."""
    content = "Just plain text content for convenience API test."
    path = _write(tmp_path, "plain.txt", content)

    result = load_file(str(path))

    assert isinstance(result, str)
    assert result == content


def test_load_file_md(tmp_path):
    """load_file() works correctly for .md files."""
    content = "# Title\n\nParagraph one.\n"
    path = _write(tmp_path, "doc.md", content)

    result = load_file(str(path))
    assert result == content


def test_load_file_nonexistent(tmp_path):
    """load_file() propagates FileNotFoundError from DoclingLoader."""
    with pytest.raises(FileNotFoundError):
        load_file(str(tmp_path / "missing.txt"))
