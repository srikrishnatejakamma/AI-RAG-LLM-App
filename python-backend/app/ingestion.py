from __future__ import annotations

import asyncio
import io
import logging
import re
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


def parse_pdf_elements(data: bytes) -> list[DocumentElement]:
    """Extract page-aware PDF text using pypdf's layout-preserving mode."""
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        try:
            if not reader.decrypt(""):
                raise ValueError("Encrypted PDF cannot be read without its password")
        except Exception as exc:
            if isinstance(exc, ValueError):
                raise
            raise ValueError("Encrypted PDF cannot be read without its password") from exc

    elements: list[DocumentElement] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text(extraction_mode="layout") or ""
        except (KeyError, TypeError):
            text = page.extract_text() or ""
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
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted and not reader.decrypt(""):
        raise ValueError("Encrypted PDF cannot be read without its password")
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


def parse_txt_elements(data: bytes) -> list[DocumentElement]:
    text = parse_txt_text(data)
    return [
        DocumentElement(text=block, kind="paragraph", source_location=f"text block {index}")
        for index, block in enumerate(_text_blocks(text), start=1)
    ]


def parse_txt_text(data: bytes) -> str:
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
        # Preserve support for common legacy single-byte text encodings.
        return data.decode("latin1")


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

    for element_index, element in enumerate(elements):
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
        for part in parts:
            pending.append((
                _make_chunk_text(part, section_path), page_number, section_path,
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
) -> list[ChunkData]:
    normalized_extension = extension.lower().lstrip(".")
    if normalized_extension == "pdf":
        elements = parse_pdf_elements(payload)
    elif normalized_extension == "docx":
        elements = parse_docx_elements(payload)
    elif normalized_extension in {"txt", "text"}:
        elements = parse_txt_elements(payload)
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
            extract_and_build_chunks, extension, payload, chunk_size, overlap, embed_many
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
