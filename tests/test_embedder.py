"""Tests for opensearch_graphrag.embedder — Ollama client is fully mocked."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from opensearch_graphrag.embedder import embed_chunks, embed_text
from opensearch_graphrag.models import Chunk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EMBED_DIM = 768


def _fake_vector(seed: int = 0) -> list[float]:
    """Return a deterministic 768-dimensional unit-like vector."""
    return [float(seed + i) / (_EMBED_DIM * 100) for i in range(_EMBED_DIM)]


def _mock_response(vectors: list[list[float]]) -> MagicMock:
    """Build a mock httpx Response for POST /api/embed."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"embeddings": vectors}
    mock_resp.raise_for_status.return_value = None
    return mock_resp


def _make_chunk(idx: int, text: str = "") -> Chunk:
    return Chunk(
        id=f"c{idx}",
        text=text or f"Chunk number {idx}.",
        chunk_index=idx,
    )


# ---------------------------------------------------------------------------
# test_embed_text
# ---------------------------------------------------------------------------


def test_embed_text(tmp_path: Any) -> None:
    """embed_text returns a 768-dim list of floats for a single text."""
    vector = _fake_vector(0)
    mock_resp = _mock_response([vector])

    with patch("opensearch_graphrag.config.get_ollama_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = mock_resp

        result = embed_text("OpenSearch is a search engine.")

    assert isinstance(result, list)
    assert len(result) == _EMBED_DIM
    assert all(isinstance(v, float) for v in result)
    assert result == vector


def test_embed_text_uses_correct_model() -> None:
    """embed_text sends the model name from settings in the request body."""
    vector = _fake_vector(1)
    mock_resp = _mock_response([vector])

    with patch("opensearch_graphrag.config.get_ollama_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = mock_resp

        embed_text("Some text about graph databases.")

    # Verify that post was called once and the JSON body contains the model name
    mock_client.post.assert_called_once()
    call_args = mock_client.post.call_args

    # call_args.kwargs["json"] or positional — normalise both
    payload: dict = call_args.kwargs.get("json") or call_args.args[1]
    assert payload["model"] == "nomic-embed-text-v2-moe"
    assert payload["input"] == "Some text about graph databases."


# ---------------------------------------------------------------------------
# test_embed_chunks
# ---------------------------------------------------------------------------


def test_embed_chunks() -> None:
    """embed_chunks populates the embedding field on every returned chunk."""
    chunks = [_make_chunk(i) for i in range(3)]
    vectors = [_fake_vector(i) for i in range(3)]
    mock_resp = _mock_response(vectors)

    with patch("opensearch_graphrag.config.get_ollama_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = mock_resp

        result = embed_chunks(chunks)

    assert len(result) == 3
    for i, chunk in enumerate(result):
        assert isinstance(chunk, Chunk)
        assert len(chunk.embedding) == _EMBED_DIM
        assert chunk.embedding == vectors[i], f"Chunk {i} got wrong embedding"
        # Original fields must be preserved
        assert chunk.id == f"c{i}"
        assert chunk.text == f"Chunk number {i}."


def test_embed_chunks_sends_all_texts_in_one_request() -> None:
    """embed_chunks sends all texts as a single batched request to Ollama."""
    chunks = [_make_chunk(i) for i in range(4)]
    vectors = [_fake_vector(i) for i in range(4)]
    mock_resp = _mock_response(vectors)

    with patch("opensearch_graphrag.config.get_ollama_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = mock_resp

        embed_chunks(chunks)

    # Exactly one POST should be issued
    mock_client.post.assert_called_once()
    call_args = mock_client.post.call_args
    payload: dict = call_args.kwargs.get("json") or call_args.args[1]

    assert isinstance(payload["input"], list)
    assert len(payload["input"]) == 4
    assert payload["input"] == [c.text for c in chunks]


def test_embed_empty_chunks() -> None:
    """embed_chunks with an empty list returns an empty list without HTTP calls."""
    with patch("opensearch_graphrag.config.get_ollama_client") as mock_get_client:
        result = embed_chunks([])

    assert result == []
    mock_get_client.assert_not_called()


def test_embed_chunks_original_objects_unchanged() -> None:
    """embed_chunks does not mutate the original Chunk instances."""
    chunks = [_make_chunk(0)]
    original_embedding = list(chunks[0].embedding)  # [] by default
    vectors = [_fake_vector(0)]
    mock_resp = _mock_response(vectors)

    with patch("opensearch_graphrag.config.get_ollama_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = mock_resp

        result = embed_chunks(chunks)

    # Original chunk is unchanged (model_copy returns a new object)
    assert chunks[0].embedding == original_embedding
    # Returned chunk has the embedding
    assert result[0].embedding == vectors[0]
    # They are different objects
    assert result[0] is not chunks[0]


def test_embed_chunks_raises_on_count_mismatch() -> None:
    """embed_chunks raises ValueError when Ollama returns wrong number of vectors."""
    chunks = [_make_chunk(i) for i in range(3)]
    # Ollama returns only 2 vectors for 3 chunks
    mock_resp = _mock_response([_fake_vector(0), _fake_vector(1)])

    with patch("opensearch_graphrag.config.get_ollama_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = mock_resp

        with pytest.raises(ValueError, match="Expected 3 embeddings"):
            embed_chunks(chunks)


# ---------------------------------------------------------------------------
# Error handling & dimension validation tests
# ---------------------------------------------------------------------------


def test_embed_text_connect_error() -> None:
    """embed_text raises on connection error."""
    with patch("opensearch_graphrag.config.get_ollama_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.side_effect = httpx.ConnectError("Connection refused")

        with pytest.raises(httpx.ConnectError):
            embed_text("test")


def test_embed_text_read_timeout() -> None:
    """embed_text raises on read timeout."""
    with patch("opensearch_graphrag.config.get_ollama_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.side_effect = httpx.ReadTimeout("Read timed out")

        with pytest.raises(httpx.ReadTimeout):
            embed_text("test")


def test_embed_text_http_500() -> None:
    """embed_text raises on HTTP 500 from Ollama."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Server Error", request=MagicMock(), response=MagicMock(status_code=500),
    )

    with patch("opensearch_graphrag.config.get_ollama_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = mock_resp

        with pytest.raises(httpx.HTTPStatusError):
            embed_text("test")


def test_embed_text_dimension_mismatch() -> None:
    """embed_text raises EmbeddingError when dimension doesn't match config."""
    from opensearch_graphrag.exceptions import EmbeddingError

    # Return a 512-dim vector instead of expected 768
    wrong_dim_vector = [0.1] * 512
    mock_resp = _mock_response([wrong_dim_vector])

    with patch("opensearch_graphrag.config.get_ollama_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = mock_resp

        with pytest.raises(EmbeddingError, match="Expected embedding dimension 768"):
            embed_text("test")
