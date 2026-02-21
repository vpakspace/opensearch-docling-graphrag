"""Document loader based on IBM Docling.

Reuses patterns from agentic-graph-rag (rag-core/loader.py):
  - DoclingLoader class with lazy initialization
  - GPU optional via AcceleratorDevice.AUTO
  - load_bytes for upload handlers (writes temp file, delegates to load)
  - Plain text (.txt, .md) handled directly, no Docling needed

Supports: .pdf, .docx, .pptx, .xlsx, .html, .md, .txt
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docling.document_converter import DocumentConverter

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".md", ".txt"}
_PLAIN_TEXT_EXTENSIONS = {".txt", ".md"}


@dataclass
class DocumentResult:
    """Result of document processing.

    Attributes:
        markdown: Full document text as markdown.
        tables:   List of extracted table dicts with keys
                  ``caption``, ``markdown``, ``csv``, ``page``.
        images:   List of image metadata dicts with keys
                  ``caption``, ``page``.
        metadata: Document-level metadata (format, pages, counts).
    """

    markdown: str
    tables: list[dict] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class DoclingLoader:
    """Document loader with lazy Docling initialization and optional GPU support.

    Docling models (~1-2 GB) are downloaded and loaded on the first call to
    :meth:`load` or :meth:`load_bytes` for non-plain-text files.  Subsequent
    calls reuse the cached converter.

    Args:
        use_gpu: When ``True`` and Docling's accelerator extras are installed,
                 PDF processing uses ``AcceleratorDevice.AUTO`` (CUDA / MPS).

    Example::

        loader = DoclingLoader(use_gpu=False)
        result = loader.load("report.pdf")
        print(result.markdown[:200])
    """

    def __init__(self, use_gpu: bool = False) -> None:
        self._converter: DocumentConverter | None = None
        self._use_gpu = use_gpu

    # ── Converter lifecycle ──────────────────────────────────────

    def _get_converter(self) -> DocumentConverter:
        """Lazy-initialize the Docling DocumentConverter."""
        if self._converter is None:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption

            pipeline_options = PdfPipelineOptions()
            pipeline_options.generate_picture_images = True

            if self._use_gpu:
                try:
                    from docling.datamodel.accelerator_options import (
                        AcceleratorDevice,
                        AcceleratorOptions,
                    )

                    pipeline_options.accelerator_options = AcceleratorOptions(
                        device=AcceleratorDevice.AUTO
                    )
                    logger.info("GPU acceleration enabled (AcceleratorDevice.AUTO)")
                except ImportError:
                    logger.warning(
                        "GPU acceleration imports unavailable — falling back to CPU"
                    )

            self._converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
        return self._converter

    # ── Public API ───────────────────────────────────────────────

    def load(self, file_path: str | Path) -> DocumentResult:
        """Load a document from disk and return structured content.

        Args:
            file_path: Path to a supported document file.

        Returns:
            :class:`DocumentResult` with ``markdown``, ``tables``,
            ``images``, and ``metadata``.

        Raises:
            FileNotFoundError: When ``file_path`` does not exist.
            ValueError: When the file extension is not in
                        :data:`SUPPORTED_EXTENSIONS`.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format: '{path.suffix}'. "
                f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        # Plain text — skip heavy Docling pipeline entirely
        if ext in _PLAIN_TEXT_EXTENSIONS:
            text = path.read_text(encoding="utf-8")
            logger.debug("Loaded plain text from %s (%d chars)", path.name, len(text))
            return DocumentResult(
                markdown=text,
                metadata={"format": ext, "pages": 1},
            )

        # All other formats — delegate to Docling
        converter = self._get_converter()
        result = converter.convert(str(path))
        doc = result.document

        tables = self._extract_tables(doc)
        images = self._extract_images(doc)
        markdown = doc.export_to_markdown()

        pages = getattr(doc, "num_pages", None)
        if callable(pages):
            pages = pages()

        metadata = {
            "format": ext,
            "pages": pages,
            "tables_count": len(tables),
            "images_count": len(images),
        }

        logger.info(
            "Loaded %d chars from %s (%d tables, %d images)",
            len(markdown),
            path.name,
            len(tables),
            len(images),
        )
        return DocumentResult(
            markdown=markdown,
            tables=tables,
            images=images,
            metadata=metadata,
        )

    def load_bytes(self, data: bytes, filename: str) -> DocumentResult:
        """Load a document from raw bytes (for web upload handlers).

        The bytes are written to a temporary file on disk so that Docling can
        process them via its normal file-based pipeline.  The temp file is
        deleted immediately after loading regardless of success or failure.

        Args:
            data:     Raw file bytes.
            filename: Original filename — used to determine the file extension.

        Returns:
            :class:`DocumentResult` (same as :meth:`load`).

        Raises:
            ValueError: When the extension derived from ``filename`` is not
                        supported.
        """
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format: '{ext}'. "
                f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        # Plain text — decode in memory, no temp file needed
        if ext in _PLAIN_TEXT_EXTENSIONS:
            text = data.decode("utf-8", errors="replace")
            logger.debug("Loaded plain text bytes for %s (%d chars)", filename, len(text))
            return DocumentResult(
                markdown=text,
                metadata={"format": ext, "pages": 1},
            )

        fd, tmp_path = tempfile.mkstemp(suffix=ext)
        path = Path(tmp_path)
        try:
            os.close(fd)
            path.write_bytes(data)
            return self.load(path)
        finally:
            path.unlink(missing_ok=True)

    # ── Internal extraction helpers ──────────────────────────────

    @staticmethod
    def _extract_tables(doc: object) -> list[dict]:
        """Extract tables from a Docling document object."""
        tables: list[dict] = []
        for item, _level in doc.iterate_items():  # type: ignore[attr-defined]
            if hasattr(item, "export_to_dataframe"):
                try:
                    df = item.export_to_dataframe()
                    page_num = None
                    if hasattr(item, "prov") and item.prov:
                        page_num = getattr(item.prov[0], "page_no", None)
                    tables.append(
                        {
                            "caption": getattr(item, "caption", "") or "",
                            "markdown": df.to_markdown(index=False),
                            "csv": df.to_csv(index=False),
                            "page": page_num,
                        }
                    )
                except Exception as exc:
                    logger.debug("Table extraction skipped: %s", exc)
        return tables

    @staticmethod
    def _extract_images(doc: object) -> list[dict]:
        """Extract image metadata from a Docling document object."""
        images: list[dict] = []
        for item, _level in doc.iterate_items():  # type: ignore[attr-defined]
            if hasattr(item, "get_image"):
                try:
                    img = item.get_image(doc)
                    if img:
                        page_num = None
                        if hasattr(item, "prov") and item.prov:
                            page_num = getattr(item.prov[0], "page_no", None)
                        images.append(
                            {
                                "caption": getattr(item, "caption", "") or "",
                                "page": page_num,
                            }
                        )
                except Exception as exc:
                    logger.debug("Image extraction skipped: %s", exc)
        return images


# ── Convenience function ─────────────────────────────────────────


def load_file(file_path: str, use_gpu: bool = False) -> str:
    """Load a document and return its markdown text.

    Convenience wrapper around :class:`DoclingLoader` for pipelines that
    only need the plain text content.

    Args:
        file_path: Path to a supported document file.
        use_gpu:   Enable GPU acceleration for PDF processing.

    Returns:
        Markdown-formatted text content of the document.

    Example::

        text = load_file("report.pdf", use_gpu=True)
        print(text[:500])
    """
    loader = DoclingLoader(use_gpu=use_gpu)
    return loader.load(file_path).markdown
