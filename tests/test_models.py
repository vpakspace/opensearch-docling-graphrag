"""Tests for models module."""

from opensearch_graphrag.models import Chunk, Entity, QAResult, Relationship, SearchResult


def test_chunk_defaults():
    c = Chunk(id="c1", text="hello")
    assert c.id == "c1"
    assert c.text == "hello"
    assert c.embedding == []
    assert c.metadata == {}
    assert c.source == ""
    assert c.chunk_index == 0


def test_chunk_with_embedding():
    c = Chunk(id="c1", text="hello", embedding=[0.1, 0.2, 0.3])
    assert len(c.embedding) == 3


def test_entity():
    e = Entity(name="IBM", type="Organization", source_chunk_id="c1")
    assert e.name == "IBM"
    assert e.type == "Organization"
    assert e.source_chunk_id == "c1"


def test_relationship():
    r = Relationship(source="IBM", target="Granite", type="DEVELOPS")
    assert r.source == "IBM"
    assert r.target == "Granite"
    assert r.type == "DEVELOPS"


def test_relationship_default_type():
    r = Relationship(source="A", target="B")
    assert r.type == "RELATED_TO"


def test_search_result():
    sr = SearchResult(chunk_id="c1", text="hello", score=0.95)
    assert sr.chunk_id == "c1"
    assert sr.score == 0.95
    assert sr.source == ""


def test_qa_result():
    sr = SearchResult(chunk_id="c1", text="hello", score=0.9)
    qa = QAResult(answer="The answer", confidence=0.85, sources=[sr], mode="hybrid")
    assert qa.answer == "The answer"
    assert qa.confidence == 0.85
    assert len(qa.sources) == 1
    assert qa.mode == "hybrid"


def test_qa_result_defaults():
    qa = QAResult(answer="Yes")
    assert qa.confidence == 0.0
    assert qa.sources == []
    assert qa.mode == "hybrid"
    assert qa.grounded is True
    assert qa.grounding_score == 1.0
    assert qa.warning == ""


def test_qa_result_grounding_fields():
    qa = QAResult(
        answer="X",
        grounded=False,
        grounding_score=0.15,
        warning="Low grounding",
    )
    assert qa.grounded is False
    assert qa.grounding_score == 0.15
    assert qa.warning == "Low grounding"
