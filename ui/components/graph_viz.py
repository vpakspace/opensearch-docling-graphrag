"""PyVis graph visualization for Streamlit."""

from __future__ import annotations

import tempfile
from pathlib import Path


def render_graph(entities: list[dict], relationships: list[dict]) -> str | None:
    """Render a knowledge graph with PyVis and return HTML string.

    Args:
        entities: list of {"name": str, "type": str}
        relationships: list of {"source": str, "target": str, "type": str}

    Returns:
        HTML string or None if no data.
    """
    if not entities:
        return None

    from pyvis.network import Network

    net = Network(height="600px", width="100%", directed=True, notebook=False)
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=200)

    type_colors = {
        "Person": "#FF6B6B",
        "Organization": "#4ECDC4",
        "Location": "#45B7D1",
        "Date": "#96CEB4",
        "Other": "#DDA0DD",
    }

    added_nodes: set[str] = set()
    for entity in entities:
        name = entity.get("name", "")
        etype = entity.get("type", "Other")
        if name and name not in added_nodes:
            color = type_colors.get(etype, "#DDA0DD")
            net.add_node(name, label=name, title=f"{name} ({etype})", color=color, size=20)
            added_nodes.add(name)

    for rel in relationships:
        src = rel.get("source", "")
        tgt = rel.get("target", "")
        rtype = rel.get("type", "RELATED_TO")
        if src in added_nodes and tgt in added_nodes:
            net.add_edge(src, tgt, title=rtype, label=rtype, arrows="to")

    fd, tmp_path = tempfile.mkstemp(suffix=".html")
    path = Path(tmp_path)
    try:
        import os
        os.close(fd)
        net.save_graph(str(path))
        return path.read_text(encoding="utf-8")
    finally:
        path.unlink(missing_ok=True)
