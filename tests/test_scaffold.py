"""Smoke tests: verify all packages are importable."""


def test_import_opensearch_graphrag():
    import opensearch_graphrag  # noqa: F401


def test_import_api():
    import api  # noqa: F401


def test_import_ui():
    import ui  # noqa: F401


def test_project_structure():
    from pathlib import Path

    root = Path(__file__).parent.parent
    assert (root / "docker-compose.yml").exists()
    assert (root / "requirements.txt").exists()
    assert (root / "pyproject.toml").exists()
    assert (root / ".env.example").exists()
    assert (root / "scripts" / "pull_models.sh").exists()
    assert (root / "data" / "sample_graphrag.txt").exists()
