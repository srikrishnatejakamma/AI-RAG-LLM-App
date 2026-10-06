from __future__ import annotations

import asyncio
import io
import logging
import threading
import re
import os
from functools import lru_cache
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator

from docx import Document as DocxDocument
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

from .domain import ChunkData, DocumentData
from .store import MemoryStore
from .text_pipeline import chunk_text, embed_local


log = logging.getLogger("python-rag-backend")
_docling_conversion_lock = threading.Lock()


@dataclass(frozen=True)
class DocumentElement:
    """A format-neutral piece of extracted content with its source position."""

    text: str
    kind: str = "paragraph"
    page_number: int | None = None
    heading_level: int | None = None
    source_location: str | None = None
    group_id: str | None = None


def _heading_level(style_name: str) -> int | None:
    """Read heading depth from stable Word heading style identifiers."""
    match = re.fullmatch(r"heading\s*(\d+)", style_name.strip(), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _is_list_item(paragraph: Paragraph) -> bool:
    properties = paragraph._p.pPr
    style_id = paragraph.style.style_id.casefold() if paragraph.style else ""
    return bool((properties is not None and properties.numPr is not None) or style_id.startswith("list"))


def _iter_docx_blocks(container: Any) -> Iterator[Paragraph | Table]:
    """Yield paragraphs and tables in their original order from a DOCX container."""
    document_element = getattr(container, "element", None)
    if document_element is not None and getattr(document_element, "body", None) is not None:
        element = document_element.body
    else:
        element = getattr(container, "_element", None)
    if element is None:
        return
    for child in element.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, container)
        elif isinstance(child, CT_Tbl):
            yield Table(child, container)


def _open_pdf(data: bytes) -> PdfReader:
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:
        raise ValueError("PDF could not be opened or is damaged") from exc
    if reader.is_encrypted:
        try:
            decrypted = reader.decrypt("")
        except Exception as exc:
            raise ValueError("Encrypted PDF cannot be read without its password") from exc
        if not decrypted:
            raise ValueError("Encrypted PDF cannot be read without its password")
    return reader


def parse_pdf_elements(data: bytes) -> list[DocumentElement]:
    """Extract page-aware PDF text using pypdf's layout-preserving mode."""
    reader = _open_pdf(data)
    elements: list[DocumentElement] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text(extraction_mode="layout") or ""
        except (KeyError, TypeError):
            text = page.extract_text() or ""
        if not text.strip():
            try:
                has_images = bool(page.images)
            except (AttributeError, KeyError, TypeError):
                has_images = False
            if has_images:
                raise ValueError(
                    f"PDF page {page_number} contains image content without extractable text; OCR is required"
                )
        for block_index, block in enumerate(_text_blocks(text), start=1):
            elements.append(
                DocumentElement(
                    text=block,
                    kind="paragraph",
                    page_number=page_number,
                    source_location=f"page {page_number}, block {block_index}",
                )
            )
    return elements


def parse_pdf_pages(data: bytes) -> list[tuple[int, str]]:
    """Compatibility helper returning one extracted text value per PDF page."""
    reader = _open_pdf(data)
    pages: list[tuple[int, str]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text(extraction_mode="layout") or ""
        except (KeyError, TypeError):
            text = page.extract_text() or ""
        pages.append((page_number, text))
    return pages


def parse_docx_elements(data: bytes) -> list[DocumentElement]:
    """Extract ordered DOCX body blocks, tables, lists, headings, and page-free locations."""
    document = DocxDocument(io.BytesIO(data))
    elements: list[DocumentElement] = []
    table_number = 0

    # Headers and footers are document-level content. Include each once instead
    # of duplicating it on every rendered page.
    for section_index, section in enumerate(document.sections, start=1):
        for kind, container in (("header", section.header),):
            for block_index, block in enumerate(_iter_docx_blocks(container), start=1):
                text = _docx_block_text(block)
                if text:
                    elements.append(DocumentElement(text, kind, source_location=f"section {section_index} {kind} {block_index}"))

    for block_index, block in enumerate(_iter_docx_blocks(document), start=1):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if not text:
                continue
            level = _heading_level(block.style.style_id if block.style else "")
            kind = "heading" if level is not None else "list_item" if _is_list_item(block) else "paragraph"
            elements.append(
                DocumentElement(
                    text=text,
                    kind=kind,
                    heading_level=level,
                    source_location=f"body block {block_index}",
                )
            )
            continue

        table_number += 1
        table_id = f"table-{table_number}"
        for row_index, row in enumerate(block.rows, start=1):
            cells = [_clean_cell_text(cell.text) for cell in row.cells]
            # Empty cells are retained so the resulting row keeps its column positions.
            if any(cells):
                elements.append(
                    DocumentElement(
                        text=" | ".join(cells),
                        kind="table_row",
                        source_location=f"table {table_number}, row {row_index}",
                        group_id=table_id,
                    )
                )

    for section_index, section in enumerate(document.sections, start=1):
        for block_index, block in enumerate(_iter_docx_blocks(section.footer), start=1):
            text = _docx_block_text(block)
            if text:
                elements.append(DocumentElement(text, "footer", source_location=f"section {section_index} footer {block_index}"))
    return elements


def parse_docx_text(data: bytes) -> str:
    """Compatibility helper returning structured DOCX extraction as plain text."""
    return "\n".join(element.text for element in parse_docx_elements(data))


def _docx_block_text(block: Paragraph | Table) -> str:
    if isinstance(block, Paragraph):
        return block.text.strip()
    rows = []
    for row in block.rows:
        cells = [_clean_cell_text(cell.text) for cell in row.cells]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _clean_cell_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _text_blocks(text: str) -> list[str]:
    # Preserve paragraph boundaries while joining visual line wraps within a block.
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return [re.sub(r"[ \t]+", " ", block).strip() for block in re.split(r"\n\s*\n", normalized) if block.strip()]


def parse_txt_elements(data: bytes, encoding: str | None = None) -> list[DocumentElement]:
    text = parse_txt_text(data, encoding=encoding)
    return [
        DocumentElement(text=block, kind="paragraph", source_location=f"text block {index}")
        for index, block in enumerate(_text_blocks(text), start=1)
    ]


def parse_txt_text(data: bytes, encoding: str | None = None) -> str:
    requested_encoding = (encoding or os.getenv("RAG_TEXT_ENCODING", "auto")).strip()
    if requested_encoding.casefold() != "auto":
        try:
            return data.decode(requested_encoding)
        except (LookupError, UnicodeDecodeError) as exc:
            raise ValueError(f"Text could not be decoded with RAG_TEXT_ENCODING={requested_encoding}") from exc
    # Check UTF-32 before UTF-16 because their BOMs share a prefix.
    for encoding, markers in (
        ("utf-32", (b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")),
        ("utf-16", (b"\xff\xfe", b"\xfe\xff")),
        ("utf-8-sig", (b"\xef\xbb\xbf",)),
    ):
        if data.startswith(markers):
            return data.decode(encoding)
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        if b"\x00" in data:
            for encoding in ("utf-16", "utf-32"):
                try:
                    return data.decode(encoding)
                except UnicodeDecodeError:
                    continue
            raise ValueError("Text document encoding could not be identified")
        try:
            from charset_normalizer import from_bytes
        except ImportError as exc:
            raise ValueError(
                "Text encoding is ambiguous; install charset-normalizer or re-save the file as UTF-8"
            ) from exc
        match = from_bytes(data).best()
        if (
            match is None
            or match.chaos > 0.2
            or match.coherence < 0.2
            or (match.encoding == "ascii" and any(byte > 127 for byte in data))
        ):
            raise ValueError(
                "Text encoding is uncertain; re-save the file as UTF-8 or configure its encoding"
            )
        # CharsetMatch.output() normalizes to UTF-8 by default. `str(match)`
        # decodes using the detected source encoding and avoids re-decoding
        # normalized UTF-8 bytes as (for example) Windows-1252.
        return str(match)


@lru_cache(maxsize=8)
def _docling_converter(
    ocr_enabled: bool,
    ocr_engine: str,
    ocr_languages: tuple[str, ...],
):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions(do_ocr=ocr_enabled, do_table_structure=True)
    if ocr_enabled:
        if ocr_engine == "rapidocr":
            from docling.datamodel.pipeline_options import RapidOcrOptions

            pipeline_options.ocr_options = RapidOcrOptions(lang=list(ocr_languages))
        elif ocr_engine == "tesseract":
            from docling.datamodel.pipeline_options import TesseractCliOcrOptions

            pipeline_options.ocr_options = TesseractCliOcrOptions(lang=list(ocr_languages))
        else:
            raise ValueError("RAG_DOCUMENT_OCR_ENGINE must be rapidocr or tesseract")
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.DOCX],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        },
    )


def _docling_available() -> bool:
    try:
        import docling  # noqa: F401
    except ImportError:
        return False
    return True


def _parse_with_docling(
    data: bytes,
    extension: str,
    ocr_enabled: bool,
    ocr_engine: str,
    ocr_languages: tuple[str, ...],
    max_pages: int,
) -> list[DocumentElement]:
    from docling.datamodel.base_models import DocumentStream

    converter = _docling_converter(ocr_enabled, ocr_engine, ocr_languages)
    source = DocumentStream(name=f"upload.{extension}", stream=io.BytesIO(data))
    # Docling's converter and model pipelines keep mutable caches. Serializing
    # conversion avoids races when multiple upload tasks share this process.
    with _docling_conversion_lock:
        result = converter.convert(
            source,
            max_num_pages=max_pages,
            max_file_size=len(data),
        )
    document = getattr(result, "document", None)
    if document is None:
        raise ValueError("Document analysis did not produce readable content")

    elements: list[DocumentElement] = []
    table_number = 0
    for item, _ in document.iterate_items(traverse_pictures=True):
        label_value = getattr(getattr(item, "label", None), "value", "")
        label = str(label_value or type(item).__name__).casefold()
        page_numbers = {
            int(provenance.page_no)
            for provenance in getattr(item, "prov", [])
            if getattr(provenance, "page_no", None) is not None
        }
        page_number = min(page_numbers) if page_numbers else None
        if label == "table":
            table_number += 1
            text = item.export_to_markdown(doc=document)
            kind = "table"
            group_id = f"table-{table_number}"
        elif label in {"picture", "chart"}:
            caption = getattr(item, "caption_text", None)
            text = caption(document) if callable(caption) else ""
            kind = label
            group_id = None
        else:
            text = str(getattr(item, "text", "") or "").strip()
            kind = {
                "title": "heading",
                "section_header": "heading",
                "list_item": "list_item",
                "page_header": "header",
                "page_footer": "footer",
            }.get(label, label)
            group_id = None
        if not text:
            continue
        location = f"page {page_number}" if page_number else None
        elements.append(
            DocumentElement(
                text=text,
                kind=kind,
                page_number=page_number,
                heading_level=getattr(item, "level", None) if kind == "heading" else None,
                source_location=location,
                group_id=group_id,
            )
        )
    return elements


def _coerce_elements(items: Iterable[DocumentElement | tuple[int | None, str]]) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    for item in items:
        if isinstance(item, DocumentElement):
            elements.append(item)
        else:
            page_number, text = item
            elements.append(DocumentElement(text=text, page_number=page_number))
    return elements


def _make_chunk_text(body: str, section_path: tuple[str, ...]) -> str:
    if not section_path:
        return body
    breadcrumb = " > ".join(section_path)
    return f"{breadcrumb}\n{body}" if body else breadcrumb


def _element_groups(elements: list[DocumentElement]) -> list[tuple[list[DocumentElement], tuple[str, ...]]]:
    groups: list[tuple[list[DocumentElement], tuple[str, ...]]] = []
    headings: list[str] = []
    current: list[DocumentElement] = []
    current_key: tuple[str, int | None, tuple[str, ...], str | None] | None = None

    def flush() -> None:
        nonlocal current, current_key
        if current:
            groups.append((current, current_key[2] if current_key else ()))
        current = []
        current_key = None

    for element in elements:
        text = element.text.strip()
        if not text:
            continue
        if element.kind == "heading":
            flush()
            level = max(1, element.heading_level or 1)
            headings[:] = headings[: level - 1]
            headings.append(text)
            # Keep a heading searchable, but it does not consume paragraph metadata.
            groups.append(([element], tuple(headings)))
            continue
        section_path = tuple(headings)
        key = (element.kind, element.page_number, section_path, element.group_id)
        if current_key is not None and key != current_key:
            flush()
        current_key = key
        current.append(element)
    flush()
    return groups


def build_chunks(
    pages: Iterable[DocumentElement | tuple[int | None, str]],
    chunk_size: int,
    overlap: int,
    embed_many: Callable[[list[str]], list[list[float]]] | None = None,
) -> list[ChunkData]:
    elements = _coerce_elements(pages)
    pending: list[tuple[str, int | None, tuple[str, ...], tuple[str, ...], int, int, tuple[str, ...]]] = []
    source_index = 0
    for group, section_path in _element_groups(elements):
        body = "\n".join(element.text.strip() for element in group if element.text.strip())
        if not body:
            continue
        group_types = tuple(dict.fromkeys(element.kind for element in group))
        source_locations = tuple(dict.fromkeys(element.source_location for element in group if element.source_location))
        page_numbers = {element.page_number for element in group}
        page_number = next(iter(page_numbers)) if len(page_numbers) == 1 else None
        start = source_index + 1
        source_index += len(group)
        end = source_index

        # Re-split long structures, then repeat their section breadcrumb so every
        # resulting chunk remains interpretable when retrieved on its own.
        parts = chunk_text(body, chunk_size, overlap)
        chunk_section_path = () if group_types == ("heading",) else section_path
        for part in parts:
            pending.append((
                _make_chunk_text(part, chunk_section_path), page_number, section_path,
                group_types, start, end, source_locations,
            ))

    if not pending:
        return []
    texts = [item[0] for item in pending]
    vectors = embed_many(texts) if embed_many else [embed_local(text) for text in texts]
    if len(vectors) != len(pending):
        raise RuntimeError("Embedding provider returned an incomplete batch")
    return [
        ChunkData(
            text=text,
            index=index,
            page_number=page_number,
            vector=vectors[index - 1],
            section_path=section_path,
            element_types=element_types,
            source_start=source_start,
            source_end=source_end,
            source_locations=source_locations,
        )
        for index, (text, page_number, section_path, element_types, source_start, source_end, source_locations)
        in enumerate(pending, start=1)
    ]


def extract_and_build_chunks(
    extension: str,
    payload: bytes,
    chunk_size: int,
    overlap: int,
    embed_many: Callable[[list[str]], list[list[float]]] | None = None,
    parser_provider: str = "native",
    ocr_enabled: bool = True,
    ocr_engine: str = "rapidocr",
    ocr_languages: tuple[str, ...] = ("iso:en",),
    max_pages: int = 300,
    text_encoding: str = "auto",
) -> list[ChunkData]:
    normalized_extension = extension.lower().lstrip(".")
    provider = parser_provider.strip().lower()
    if provider not in {"auto", "native", "docling"}:
        raise ValueError("RAG_DOCUMENT_PARSER must be auto, native, or docling")
    docling_format = normalized_extension in {"pdf", "docx"}
    use_docling = docling_format and (provider == "docling" or (provider == "auto" and _docling_available()))
    if use_docling:
        try:
            elements = _parse_with_docling(
                payload,
                normalized_extension,
                ocr_enabled,
                ocr_engine,
                ocr_languages,
                max_pages,
            )
        except ImportError as exc:
            raise ValueError(
                "Docling parser or its configured OCR engine is missing; install python-backend requirements"
            ) from exc
    elif provider == "docling" and docling_format:
        raise ValueError("Docling parser is not installed; install python-backend requirements")
    elif normalized_extension == "pdf":
        elements = parse_pdf_elements(payload)
    elif normalized_extension == "docx":
        elements = parse_docx_elements(payload)
    elif normalized_extension in {"txt", "text"}:
        elements = parse_txt_elements(payload, encoding=text_encoding)
    else:
        raise ValueError(f"Unsupported document format: {extension}")
    chunks = build_chunks(elements, chunk_size, overlap, embed_many)
    if not chunks and normalized_extension == "pdf":
        raise ValueError("No extractable text was found in this PDF; scanned or image-only pages require OCR")
    return chunks


async def ingest_document(
    store: MemoryStore,
    collections: dict[str, Any],
    chunk_size: int,
    overlap: int,
    collection_id: str,
    document_id: str,
    extension: str,
    payload: bytes,
    actor: str,
    request_id: str,
    embed_many: Callable[[list[str]], list[list[float]]] | None = None,
    file_search: Any | None = None,
    parser_provider: str = "native",
    ocr_enabled: bool = True,
    ocr_engine: str = "rapidocr",
    ocr_languages: tuple[str, ...] = ("iso:en",),
    max_pages: int = 300,
    text_encoding: str = "auto",
) -> None:
    await asyncio.sleep(0)
    collection = collections.get(collection_id)
    if not collection:
        return
    document: DocumentData | None = collection.documents.get(document_id)
    if not document:
        return
    hosted_file_id: str | None = None
    try:
        if file_search is not None:
            if document.openai_file_id:
                await asyncio.to_thread(file_search.delete_file, document.openai_file_id)
                document.openai_file_id = None
            hosted_file_id = await asyncio.to_thread(
                file_search.upload_and_index, collection_id, document_id, document.name, payload
            )
            if collections.get(collection_id) is not collection or collection.documents.get(document_id) is not document:
                await asyncio.to_thread(file_search.delete_file, hosted_file_id)
                return
            document.openai_file_id = hosted_file_id
            document.status = "READY"
            document.error = None
            await store.record_audit(actor, "DOCUMENT_INGESTION_COMPLETED", "document", document_id, "READY", request_id)
            return
        chunks = await asyncio.to_thread(
            extract_and_build_chunks,
            extension,
            payload,
            chunk_size,
            overlap,
            embed_many,
            parser_provider,
            ocr_enabled,
            ocr_engine,
            ocr_languages,
            max_pages,
            text_encoding,
        )
        if not chunks:
            raise ValueError("No readable text was found in this document")
        if len(chunks) > 1800:
            raise ValueError("Document is too large after extraction; split it into smaller files")
        document.chunks = chunks
        document.status = "READY"
        document.error = None
        await store.record_audit(actor, "DOCUMENT_INGESTION_COMPLETED", "document", document_id, "READY", request_id)
    except Exception as exc:
        log.warning("Ingestion failed for document %s", document_id, exc_info=exc)
        if hosted_file_id and file_search is not None:
            try:
                await asyncio.to_thread(file_search.delete_file, hosted_file_id)
            except Exception:
                log.warning("Could not remove orphaned hosted file for document %s", document_id)
        document.status = "FAILED"
        document.error = str(exc) if isinstance(exc, ValueError) else "Document processing failed. Check provider settings and retry."
        await store.record_audit(actor, "DOCUMENT_INGESTION_FAILED", "document", document_id, "FAILED", request_id)
