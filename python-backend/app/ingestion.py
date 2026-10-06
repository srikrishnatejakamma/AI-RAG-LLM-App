from __future__ import annotations

import asyncio
import io
import logging
from typing import Any, Callable, Iterable

from docx import Document as DocxDocument
from pypdf import PdfReader

from .domain import ChunkData, DocumentData
from .store import MemoryStore
from .text_pipeline import chunk_text, embed_local


log = logging.getLogger("python-rag-backend")


def parse_pdf_pages(data: bytes) -> list[tuple[int, str]]:
    reader = PdfReader(io.BytesIO(data))
    pages: list[tuple[int, str]] = []
    for idx, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text(extraction_mode="layout") or ""
        except (KeyError, TypeError):
            text = page.extract_text() or ""
        pages.append((idx, text))
    return pages


def parse_docx_text(data: bytes) -> str:
    doc = DocxDocument(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs)


def parse_txt_text(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        if b"\x00" in data:
            try:
                return data.decode("utf-16")
            except UnicodeDecodeError:
                pass
        return data.decode("latin1")


def build_chunks(
    pages: Iterable[tuple[int | None, str]],
    chunk_size: int,
    overlap: int,
    embed_many: Callable[[list[str]], list[list[float]]] | None = None,
) -> list[ChunkData]:
    pending: list[tuple[int | None, str]] = []
    for page_no, page_text in pages:
        if not page_text or not page_text.strip():
            continue
        for part in chunk_text(page_text, chunk_size, overlap):
            pending.append((page_no, part))
    if not pending:
        return []
    texts = [part for _, part in pending]
    vectors = embed_many(texts) if embed_many else [embed_local(text) for text in texts]
    if len(vectors) != len(pending):
        raise RuntimeError("Embedding provider returned an incomplete batch")
    return [
        ChunkData(text=text, index=index, page_number=page_no, vector=vectors[index - 1])
        for index, (page_no, text) in enumerate(pending, start=1)
    ]


def extract_and_build_chunks(
    extension: str,
    payload: bytes,
    chunk_size: int,
    overlap: int,
    embed_many: Callable[[list[str]], list[list[float]]] | None = None,
) -> list[ChunkData]:
    if extension == "pdf":
        pages = parse_pdf_pages(payload)
    elif extension == "docx":
        pages = [(None, parse_docx_text(payload))]
    else:
        pages = [(None, parse_txt_text(payload))]
    return build_chunks(pages, chunk_size, overlap, embed_many)


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
