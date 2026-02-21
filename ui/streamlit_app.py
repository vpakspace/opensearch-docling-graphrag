"""Streamlit UI for OpenSearch Docling GraphRAG — 6 tabs."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st  # noqa: E402

from ui.i18n import get_translator  # noqa: E402

st.set_page_config(
    page_title="OpenSearch Docling GraphRAG",
    page_icon="🔍",
    layout="wide",
)


# ── Sidebar ──────────────────────────────────────────────────────
lang = st.sidebar.selectbox("Language / Язык", ["en", "ru"], index=0)
t = get_translator(lang)

st.sidebar.markdown(f"### {t('app_title')}")
st.sidebar.markdown(t("app_subtitle"))

# ── API Client ───────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8508/api/v1")


def _api_get(path: str) -> dict | list | None:
    import httpx

    try:
        resp = httpx.get(f"{API_BASE}{path}", timeout=30.0)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"{t('error')}: {e}")
        return None


def _api_post(path: str, data: dict) -> dict | list | None:
    import httpx

    try:
        resp = httpx.post(f"{API_BASE}{path}", json=data, timeout=120.0)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"{t('error')}: {e}")
        return None


# ── Direct Python fallback ───────────────────────────────────────
@st.cache_resource
def _get_service():
    """Create PipelineService directly if API is not available."""
    try:
        from opensearch_graphrag.config import get_settings
        from opensearch_graphrag.opensearch_store import OpenSearchStore
        from opensearch_graphrag.service import PipelineService

        cfg = get_settings()
        store = OpenSearchStore(settings=cfg)

        driver = None
        try:
            from neo4j import GraphDatabase

            driver = GraphDatabase.driver(
                cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password),
            )
        except Exception:
            pass

        return PipelineService(store=store, neo4j_driver=driver, settings=cfg)
    except Exception:
        return None


# ── Tabs ─────────────────────────────────────────────────────────
tabs = st.tabs([
    t("tab_home"),
    t("tab_upload"),
    t("tab_search"),
    t("tab_graph"),
    t("tab_batch"),
    t("tab_settings"),
])


# ── Tab 1: Home ──────────────────────────────────────────────────
with tabs[0]:
    st.header(t("tab_home"))

    health = _api_get("/health")
    if health:
        col1, col2, col3 = st.columns(3)
        with col1:
            status = "✅" if health.get("opensearch") else "❌"
            st.metric(t("opensearch_status"), status)
        with col2:
            status = "✅" if health.get("neo4j") else "❌"
            st.metric(t("neo4j_status"), status)
        with col3:
            status = "✅" if health.get("ollama") else "❌"
            st.metric(t("ollama_status"), status)

    stats = _api_get("/graph/stats")
    if stats:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(t("documents_count"), stats.get("documents", 0))
        c2.metric(t("chunks_count"), stats.get("chunks", 0))
        c3.metric(t("entities_count"), stats.get("entities", 0))
        c4.metric(t("relationships_count"), stats.get("relationships", 0))


# ── Tab 2: Upload ────────────────────────────────────────────────
with tabs[1]:
    st.header(t("tab_upload"))
    st.markdown(t("upload_help"))

    uploaded = st.file_uploader(
        t("upload_file"),
        type=["pdf", "docx", "pptx", "html", "txt", "md"],
    )
    col1, col2 = st.columns(2)
    skip_ner = col1.checkbox(t("skip_ner"), value=False)
    use_gpu = col2.checkbox(t("use_gpu"), value=False)

    if uploaded and st.button(t("ingest_button")):
        with st.spinner(t("ingesting")):
            import tempfile

            fd, tmp_path = tempfile.mkstemp(suffix=os.path.splitext(uploaded.name)[1])
            os.close(fd)
            with open(tmp_path, "wb") as f:
                f.write(uploaded.read())

            try:
                from opensearch_graphrag.chunker import chunk_text
                from opensearch_graphrag.config import get_settings
                from opensearch_graphrag.embedder import embed_chunks
                from opensearch_graphrag.loader import load_file
                from opensearch_graphrag.opensearch_store import OpenSearchStore

                cfg = get_settings()
                text = load_file(tmp_path, use_gpu=use_gpu)
                chunks = chunk_text(text)
                for i, chunk in enumerate(chunks):
                    chunk.source = uploaded.name
                    chunk.chunk_index = i
                chunks = embed_chunks(chunks)

                store = OpenSearchStore(settings=cfg)
                store.init_index()
                stored = store.add_chunks(chunks)

                if not skip_ner:
                    from neo4j import GraphDatabase

                    from opensearch_graphrag.entity_extractor import extract_entities
                    from opensearch_graphrag.graph_builder import GraphBuilder

                    driver = GraphDatabase.driver(
                        cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password),
                    )
                    try:
                        builder = GraphBuilder(driver)
                        doc_id = uploaded.name.replace(".", "_")
                        builder.add_document(doc_id, uploaded.name)
                        for chunk in chunks:
                            builder.add_chunk(chunk.id, chunk.text, doc_id, chunk.chunk_index)
                            entities = extract_entities(chunk.text, chunk_id=chunk.id, settings=cfg)
                            for entity in entities:
                                builder.add_entity(entity)
                                builder.link_entity_to_chunk(entity.name, entity.type, chunk.id)
                    finally:
                        driver.close()

                st.success(f"{t('ingest_success')} ({stored} chunks)")
            except Exception as e:
                st.error(f"{t('ingest_error')}: {e}")
            finally:
                os.unlink(tmp_path)


# ── Tab 3: Search & Q&A ─────────────────────────────────────────
with tabs[2]:
    st.header(t("tab_search"))

    query = st.text_input(t("search_query"), placeholder="What is OpenSearch?")
    mode = st.selectbox(t("search_mode"), ["hybrid", "bm25", "vector", "graph", "enhanced", "cognitive"], index=0)

    if query and st.button(t("search_button")):
        with st.spinner(t("loading")):
            result = _api_post("/query", {"text": query, "mode": mode})

        if result:
            st.subheader(t("answer"))
            st.markdown(result.get("answer", ""))

            confidence = result.get("confidence", 0.0)
            st.progress(min(confidence, 1.0), text=f"{t('confidence')}: {confidence:.0%}")

            warning = result.get("warning", "")
            if warning:
                st.warning(warning)

            sources = result.get("sources", [])
            if sources:
                st.subheader(t("sources"))
                for i, src in enumerate(sources):
                    with st.expander(f"[{i + 1}] {src.get('source', 'unknown')} (score: {src.get('score', 0):.3f})"):
                        st.text(src.get("text", ""))


# ── Tab 4: Graph Explorer ───────────────────────────────────────
with tabs[3]:
    st.header(t("graph_title"))

    stats = _api_get("/graph/stats")
    if stats and (stats.get("entities", 0) > 0 or stats.get("relationships", 0) > 0):
        svc = _get_service()
        if svc and svc._driver:
            try:
                with svc._driver.session() as session:
                    # Get entities
                    result = session.run(
                        "MATCH (e:Entity) RETURN e.name AS name, e.type AS type LIMIT 100"
                    )
                    entities = [dict(r) for r in result]

                    # Get relationships
                    result = session.run(
                        "MATCH (s:Entity)-[r]->(t:Entity) "
                        "RETURN s.name AS source, t.name AS target, type(r) AS type LIMIT 200"
                    )
                    rels = [dict(r) for r in result]

                if entities:
                    # Type filter
                    all_types = sorted({e.get("type", "Other") for e in entities})
                    selected_types = st.multiselect(t("entity_filter"), all_types, default=all_types)
                    filtered = [e for e in entities if e.get("type", "Other") in selected_types]

                    from ui.components.graph_viz import render_graph

                    html = render_graph(filtered, rels)
                    if html:
                        import streamlit.components.v1 as components

                        components.html(html, height=650)
                else:
                    st.info(t("graph_empty"))
            except Exception as e:
                st.error(f"{t('error')}: {e}")
        else:
            st.info(t("graph_empty"))
    else:
        st.info(t("graph_empty"))


# ── Tab 5: Batch Process ────────────────────────────────────────
with tabs[4]:
    st.header(t("tab_batch"))

    batch_dir = st.text_input(t("batch_dir"), placeholder="/path/to/documents/")

    if batch_dir and st.button(t("batch_button")):
        if not os.path.isdir(batch_dir):
            st.error(f"{t('error')}: Directory not found")
        else:
            files = [
                os.path.join(batch_dir, f)
                for f in sorted(os.listdir(batch_dir))
                if os.path.isfile(os.path.join(batch_dir, f)) and not f.startswith(".")
            ]
            if not files:
                st.warning(t("no_results"))
            else:
                progress = st.progress(0, text=t("batch_progress"))
                for i, fpath in enumerate(files):
                    try:
                        from opensearch_graphrag.chunker import chunk_text
                        from opensearch_graphrag.config import get_settings
                        from opensearch_graphrag.embedder import embed_chunks
                        from opensearch_graphrag.loader import load_file
                        from opensearch_graphrag.opensearch_store import OpenSearchStore

                        cfg = get_settings()
                        text = load_file(fpath)
                        chunks = chunk_text(text)
                        for j, chunk in enumerate(chunks):
                            chunk.source = os.path.basename(fpath)
                            chunk.chunk_index = j
                        chunks = embed_chunks(chunks)
                        store = OpenSearchStore(settings=cfg)
                        store.init_index()
                        store.add_chunks(chunks)
                        st.write(f"✅ {os.path.basename(fpath)} ({len(chunks)} chunks)")
                    except Exception as e:
                        st.write(f"❌ {os.path.basename(fpath)}: {e}")

                    progress.progress((i + 1) / len(files))

                st.success(t("success"))


# ── Tab 6: Settings ─────────────────────────────────────────────
with tabs[5]:
    st.header(t("settings_title"))

    try:
        from opensearch_graphrag.config import get_settings

        cfg = get_settings()

        col1, col2 = st.columns(2)
        with col1:
            st.text_input(t("ollama_url"), value=cfg.ollama.base_url, disabled=True)
            st.text_input(t("llm_model"), value=cfg.ollama.llm_model, disabled=True)
            st.text_input(t("embed_model"), value=cfg.ollama.embed_model, disabled=True)
        with col2:
            st.text_input(t("opensearch_url"), value=cfg.opensearch.url, disabled=True)
            st.text_input(t("neo4j_uri"), value=cfg.neo4j.uri, disabled=True)
            st.number_input(t("chunk_size"), value=cfg.chunking.chunk_size, disabled=True)

        st.divider()

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button(t("clear_index")):
                try:
                    from opensearch_graphrag.opensearch_store import OpenSearchStore

                    store = OpenSearchStore(settings=cfg)
                    store.delete_all()
                    st.success(t("cleared"))
                except Exception as e:
                    st.error(f"{t('error')}: {e}")

        with col_b:
            if st.button(t("clear_graph")):
                try:
                    from neo4j import GraphDatabase

                    from opensearch_graphrag.graph_builder import GraphBuilder

                    driver = GraphDatabase.driver(
                        cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password),
                    )
                    try:
                        builder = GraphBuilder(driver)
                        builder.clear()
                        st.success(t("cleared"))
                    finally:
                        driver.close()
                except Exception as e:
                    st.error(f"{t('error')}: {e}")

    except Exception as e:
        st.error(f"{t('error')}: {e}")
