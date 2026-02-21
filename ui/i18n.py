"""Internationalization support (EN/RU)."""

from __future__ import annotations

TRANSLATIONS: dict[str, dict[str, str]] = {
    "app_title": {"en": "OpenSearch Docling GraphRAG", "ru": "OpenSearch Docling GraphRAG"},
    "app_subtitle": {
        "en": "Fully Local RAG: OpenSearch + Neo4j + Ollama + Docling",
        "ru": "Полностью локальный RAG: OpenSearch + Neo4j + Ollama + Docling",
    },
    "language": {"en": "Language", "ru": "Язык"},
    # Tabs
    "tab_home": {"en": "Home", "ru": "Главная"},
    "tab_upload": {"en": "Upload", "ru": "Загрузка"},
    "tab_search": {"en": "Search & Q&A", "ru": "Поиск и Q&A"},
    "tab_graph": {"en": "Graph Explorer", "ru": "Граф"},
    "tab_batch": {"en": "Batch Process", "ru": "Пакетная обработка"},
    "tab_settings": {"en": "Settings", "ru": "Настройки"},
    # Home
    "status": {"en": "Status", "ru": "Статус"},
    "connected": {"en": "Connected", "ru": "Подключено"},
    "disconnected": {"en": "Disconnected", "ru": "Отключено"},
    "opensearch_status": {"en": "OpenSearch", "ru": "OpenSearch"},
    "neo4j_status": {"en": "Neo4j", "ru": "Neo4j"},
    "ollama_status": {"en": "Ollama", "ru": "Ollama"},
    "documents_count": {"en": "Documents", "ru": "Документы"},
    "chunks_count": {"en": "Chunks", "ru": "Фрагменты"},
    "entities_count": {"en": "Entities", "ru": "Сущности"},
    "relationships_count": {"en": "Relationships", "ru": "Связи"},
    # Upload
    "upload_file": {"en": "Upload a document", "ru": "Загрузить документ"},
    "upload_help": {
        "en": "Supported: PDF, DOCX, PPTX, HTML, TXT, MD",
        "ru": "Поддерживаются: PDF, DOCX, PPTX, HTML, TXT, MD",
    },
    "skip_ner": {"en": "Skip NER (no graph)", "ru": "Пропустить NER (без графа)"},
    "use_gpu": {"en": "Use GPU", "ru": "Использовать GPU"},
    "ingest_button": {"en": "Ingest", "ru": "Загрузить"},
    "ingesting": {"en": "Ingesting...", "ru": "Загрузка..."},
    "ingest_success": {"en": "Ingestion complete!", "ru": "Загрузка завершена!"},
    "ingest_error": {"en": "Ingestion failed", "ru": "Ошибка загрузки"},
    # Search
    "search_query": {"en": "Enter your question", "ru": "Введите вопрос"},
    "search_mode": {"en": "Search mode", "ru": "Режим поиска"},
    "search_button": {"en": "Search", "ru": "Искать"},
    "answer": {"en": "Answer", "ru": "Ответ"},
    "confidence": {"en": "Confidence", "ru": "Уверенность"},
    "sources": {"en": "Sources", "ru": "Источники"},
    "no_results": {"en": "No results found", "ru": "Результаты не найдены"},
    # Graph
    "graph_title": {"en": "Knowledge Graph", "ru": "Граф знаний"},
    "graph_empty": {"en": "No graph data available", "ru": "Данные графа недоступны"},
    "entity_filter": {"en": "Filter by entity type", "ru": "Фильтр по типу сущности"},
    # Batch
    "batch_dir": {"en": "Directory path", "ru": "Путь к директории"},
    "batch_button": {"en": "Process all", "ru": "Обработать все"},
    "batch_progress": {"en": "Processing...", "ru": "Обработка..."},
    # Settings
    "settings_title": {"en": "Configuration", "ru": "Конфигурация"},
    "ollama_url": {"en": "Ollama URL", "ru": "Ollama URL"},
    "llm_model": {"en": "LLM Model", "ru": "Модель LLM"},
    "embed_model": {"en": "Embedding Model", "ru": "Модель эмбеддингов"},
    "opensearch_url": {"en": "OpenSearch URL", "ru": "OpenSearch URL"},
    "neo4j_uri": {"en": "Neo4j URI", "ru": "Neo4j URI"},
    "chunk_size": {"en": "Chunk size", "ru": "Размер фрагмента"},
    "clear_index": {"en": "Clear index", "ru": "Очистить индекс"},
    "clear_graph": {"en": "Clear graph", "ru": "Очистить граф"},
    "clear_confirm": {"en": "Are you sure?", "ru": "Вы уверены?"},
    "cleared": {"en": "Cleared!", "ru": "Очищено!"},
    # Search modes
    "mode_enhanced": {"en": "Enhanced (Cog-RAG)", "ru": "Расширенный (Cog-RAG)"},
    "mode_cognitive": {"en": "Cognitive (2-stage)", "ru": "Когнитивный (2-стадийный)"},
    # Hallucination warning
    "hallucination_warning": {
        "en": "Low grounding: answer may not be fully supported by context",
        "ru": "Низкая обоснованность: ответ может быть недостаточно подтверждён контекстом",
    },
    "grounding_score": {"en": "Grounding", "ru": "Обоснованность"},
    # Entity types
    "entity_person": {"en": "Person", "ru": "Персона"},
    "entity_organization": {"en": "Organization", "ru": "Организация"},
    "entity_location": {"en": "Location", "ru": "Местоположение"},
    "entity_date": {"en": "Date", "ru": "Дата"},
    "entity_other": {"en": "Other", "ru": "Другое"},
    # Common
    "error": {"en": "Error", "ru": "Ошибка"},
    "loading": {"en": "Loading...", "ru": "Загрузка..."},
    "success": {"en": "Success", "ru": "Успех"},
}


def get_translator(lang: str = "en"):
    """Return a translator function for the given language."""

    def t(key: str) -> str:
        entry = TRANSLATIONS.get(key, {})
        return entry.get(lang, entry.get("en", key))

    return t
