from __future__ import annotations

import asyncio
import logging
import secrets
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, Response

from .ai import (
    EmbeddingService,
    OpenAIAgentOrchestrator,
    contextualize_question,
    correct_query_spelling,
    extractive_answer,
    is_collection_overview,
)
from .config import Settings
from .domain import CollectionData, DocumentData, SessionData
from .file_search import OpenAIFileSearch
from .ingestion import ingest_document
from .models import AuditPage, ChatRequest, ChatResponse, CollectionView, CreateCollectionRequest, CsrfView, DocumentView, HealthView, SessionView
from .retrieval import HybridRetriever
from .store import MemoryStore
from .text_pipeline import checksum


log = logging.getLogger("python-rag-backend")
logging.basicConfig(level=logging.INFO)


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def require_uuid(value: str) -> None:
    try:
        uuid.UUID(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"error": "Identifier must be a UUID"}) from exc


class AppContext:
    def __init__(self) -> None:
        self.settings = Settings()
        self.store = MemoryStore()
        self.sessions: dict[str, SessionData] = {}
        self.embedding_service = EmbeddingService(self.settings)
        self.retriever = HybridRetriever(dense_weight=0.35 if self.settings.embedding_provider == "local" else 1.0)
        self.agent = OpenAIAgentOrchestrator(self.settings)
        self.file_search = OpenAIFileSearch(self.settings)
        self.ingestion_slots = asyncio.BoundedSemaphore(6)


ctx = AppContext()
API_TAGS = [
    {"name": "System", "description": "Health checks and CSRF bootstrap."},
    {"name": "Authentication", "description": "Session login and logout."},
    {"name": "Collections", "description": "Create and manage isolated document sets."},
    {"name": "Documents", "description": "Upload, inspect, and remove source documents."},
    {"name": "Chat", "description": "Ask questions grounded in a collection's indexed sources."},
    {"name": "Audit", "description": "Admin-only cursor-paginated audit events."},
]
app = FastAPI(
    title="Gather RAG API",
    summary="Upload documents and ask collection-scoped, source-grounded questions.",
    description=(
        "Runtime document workspace API. Upload PDF, DOCX, and TXT files into collections, "
        "then ask questions against indexed passages. State-changing requests require the "
        "X-XSRF-TOKEN header; authenticated requests use the JSESSIONID session cookie."
    ),
    version="1.0.0",
    openapi_url="/v3/api-docs",
    docs_url="/swagger-ui.html",
    redoc_url="/redoc",
    openapi_tags=API_TAGS,
)


def custom_openapi() -> dict[str, Any]:
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        summary=app.summary,
        description=app.description,
        routes=app.routes,
        tags=API_TAGS,
    )
    login_body = schema.get("components", {}).get("schemas", {}).get("Body_login_login_post", {})
    login_properties = login_body.get("properties", {})
    if "csrf_form_token" in login_properties:
        login_properties["_csrf"] = login_properties.pop("csrf_form_token")
    components = schema.setdefault("components", {})
    components["securitySchemes"] = {
        "sessionCookie": {"type": "apiKey", "in": "cookie", "name": "JSESSIONID", "description": "Session cookie set by POST /login."},
        "csrfToken": {"type": "apiKey", "in": "header", "name": "X-XSRF-TOKEN", "description": "Token returned by GET /api/csrf and mirrored in the XSRF-TOKEN cookie."},
    }
    for path, path_item in schema.get("paths", {}).items():
        for method, operation in path_item.items():
            if not isinstance(operation, dict) or method not in {"get", "post", "put", "patch", "delete"}:
                continue
            protected = path.startswith("/api/") and path not in {"/api/health", "/api/csrf"}
            state_changing = method in {"post", "put", "patch", "delete"}
            if path == "/login":
                operation["security"] = [{"csrfToken": []}]
            elif path == "/logout":
                operation["security"] = [{"csrfToken": []}]
            elif protected and state_changing:
                operation["security"] = [{"sessionCookie": [], "csrfToken": []}]
            elif protected:
                operation["security"] = [{"sessionCookie": []}]
    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi


async def ingest_with_slot(*args: Any, **kwargs: Any) -> None:
    try:
        await ingest_document(*args, **kwargs)
    finally:
        ctx.ingestion_slots.release()


def json_error(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": message})


def session_from_request(request: Request) -> SessionData | None:
    sid = request.cookies.get("JSESSIONID", "")
    return ctx.sessions.get(sid) if sid else None


def require_auth(request: Request) -> tuple[str, list[str], SessionData]:
    session = session_from_request(request)
    if not session:
        raise HTTPException(status_code=401, detail={"error": "Sign in is required"})
    return session.username, session.roles, session


def require_role(roles: list[str], *required: str) -> None:
    if not any(f"ROLE_{role}" in roles for role in required):
        raise HTTPException(status_code=403, detail={"error": "Your role cannot perform this action"})


def validate_csrf(request: Request, session: SessionData | None, header_token: str | None, form_token: str | None = None) -> None:
    if request.method in {"GET", "HEAD", "OPTIONS"}:
        return
    cookie_token = request.cookies.get("XSRF-TOKEN")
    provided = header_token or form_token
    if not cookie_token or not provided or provided != cookie_token:
        raise HTTPException(status_code=403, detail={"error": "CSRF token is invalid"})
    if session and session.csrf_token != cookie_token:
        raise HTTPException(status_code=403, detail={"error": "CSRF token is invalid"})


def ensure_collection(collection_id: str) -> CollectionData:
    collection = ctx.store.collections.get(collection_id)
    if not collection:
        raise HTTPException(status_code=404, detail={"error": "Collection not found"})
    return collection


def collection_view(collection: CollectionData) -> dict[str, Any]:
    return {
        "id": collection.id,
        "name": collection.name,
        "createdAt": collection.created_at.isoformat(),
        "documentCount": len(collection.documents),
    }


def document_summary(document: DocumentData) -> dict[str, Any]:
    return {
        "id": document.id,
        "name": document.name,
        "status": document.status,
        "chunkCount": document.chunk_count,
        "uploadedAt": document.uploaded_at.isoformat(),
        "error": document.error,
    }


def build_citation(match: tuple[DocumentData, any, float]) -> dict[str, Any]:
    document, chunk, _ = match
    excerpt = chunk.text if len(chunk.text) <= 220 else chunk.text[:217] + "…"
    return {
        "documentId": document.id,
        "documentName": document.name,
        "chunk": chunk.index,
        "page": chunk.page_number,
        "excerpt": excerpt,
        "sourceLocation": "; ".join(chunk.source_locations) or None,
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail
    if isinstance(detail, dict) and "error" in detail:
        return JSONResponse(status_code=exc.status_code, content=detail)
    return JSONResponse(status_code=exc.status_code, content={"error": str(detail)})


@app.exception_handler(Exception)
async def generic_exception_handler(_: Request, __: Exception) -> JSONResponse:
    return json_error(500, "The request could not be processed")


@app.get("/api/health", tags=["System"], response_model=HealthView)
async def health() -> dict[str, str]:
    result = {
        "status": "ok",
        "storage": "memory",
        "embeddingProvider": ctx.settings.embedding_provider_name(),
    }
    result.update(ctx.settings.answer_provider_health())
    result.update(ctx.settings.retrieval_provider_health())
    result.update(ctx.settings.auth_health())
    return result


@app.get("/api/csrf", tags=["System"], response_model=CsrfView)
async def csrf(request: Request) -> JSONResponse:
    session = session_from_request(request)
    token = session.csrf_token if session else secrets.token_urlsafe(32)
    response = JSONResponse({"headerName": "X-XSRF-TOKEN", "parameterName": "_csrf", "token": token})
    response.set_cookie("XSRF-TOKEN", token, httponly=False, samesite="strict", path="/")
    return response


@app.post("/login", tags=["Authentication"], status_code=204, responses={401: {"description": "Invalid credentials"}})
async def login(
    request: Request,
    username: str = Form(default=""),
    password: str = Form(default=""),
    csrf_form_token: str = Form(default="", alias="_csrf"),
    x_xsrf_token: str | None = Header(default=None),
) -> Response:
    validate_csrf(request, None, x_xsrf_token, csrf_form_token)
    account = ctx.settings.users.get(username)
    if not account or account[0] != password:
        await ctx.store.record_audit(username or "unknown", "AUTHENTICATION_FAILED", "session", "", "DENIED", "")
        return Response(status_code=401)
    sid = secrets.token_urlsafe(32)
    csrf_token = secrets.token_urlsafe(32)
    ctx.sessions[sid] = SessionData(username=username, roles=account[1], csrf_token=csrf_token, created_at=utc_now())
    await ctx.store.record_audit(username, "AUTHENTICATION_SUCCEEDED", "session", "", "SUCCESS", "")
    response = Response(status_code=204)
    response.set_cookie("JSESSIONID", sid, httponly=True, samesite="strict", path="/")
    response.set_cookie("XSRF-TOKEN", csrf_token, httponly=False, samesite="strict", path="/")
    return response


@app.post("/logout", tags=["Authentication"], status_code=204)
async def logout(request: Request, x_xsrf_token: str | None = Header(default=None)) -> Response:
    session = session_from_request(request)
    validate_csrf(request, session, x_xsrf_token)
    if session:
        await ctx.store.record_audit(session.username, "LOGOUT", "session", "", "SUCCESS", "")
    sid = request.cookies.get("JSESSIONID", "")
    if sid in ctx.sessions:
        del ctx.sessions[sid]
    response = Response(status_code=204)
    response.delete_cookie("JSESSIONID", path="/")
    response.delete_cookie("XSRF-TOKEN", path="/")
    return response


@app.get("/api/me", tags=["Authentication"], response_model=SessionView)
async def me(request: Request) -> dict[str, Any]:
    username, roles, _ = require_auth(request)
    return {"username": username, "roles": roles}


@app.get("/api/collections", tags=["Collections"], response_model=list[CollectionView])
async def list_collections(request: Request) -> list[dict[str, Any]]:
    require_auth(request)
    return [collection_view(c) for c in sorted(ctx.store.collections.values(), key=lambda it: it.created_at)]


@app.post("/api/collections", tags=["Collections"], response_model=CollectionView)
async def create_collection(
    request: Request,
    payload: CreateCollectionRequest,
    x_xsrf_token: str | None = Header(default=None),
) -> dict[str, Any]:
    username, roles, session = require_auth(request)
    require_role(roles, "ADMIN", "EDITOR")
    validate_csrf(request, session, x_xsrf_token)
    name = (payload.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail={"error": "Collection name is required"})
    if len(name) > 80:
        raise HTTPException(status_code=400, detail={"error": "Collection name must be 80 characters or fewer"})
    collection = CollectionData(id=str(uuid.uuid4()), name=name, created_at=utc_now())
    ctx.store.collections[collection.id] = collection
    await ctx.store.record_audit(username, "COLLECTION_CREATED", "collection", collection.id, "SUCCESS", "")
    return collection_view(collection)


@app.delete("/api/collections/{collection_id}", tags=["Collections"], status_code=204)
async def delete_collection(request: Request, collection_id: str, x_xsrf_token: str | None = Header(default=None)) -> Response:
    username, roles, session = require_auth(request)
    require_role(roles, "ADMIN")
    validate_csrf(request, session, x_xsrf_token)
    require_uuid(collection_id)
    if collection_id not in ctx.store.collections:
        raise HTTPException(status_code=404, detail={"error": "Collection not found"})
    collection = ctx.store.collections[collection_id]
    if ctx.settings.retrieval_provider == "openai-file-search":
        try:
            await asyncio.to_thread(ctx.file_search.delete_collection_files, [d.openai_file_id for d in collection.documents.values() if d.openai_file_id])
        except Exception as exc:
            log.warning("Hosted document cleanup failed: %s", exc)
            raise HTTPException(status_code=502, detail={"error": "Could not remove the collection's hosted files"}) from exc
    del ctx.store.collections[collection_id]
    await ctx.store.record_audit(username, "COLLECTION_DELETED", "collection", collection_id, "SUCCESS", "")
    return Response(status_code=204)


@app.get("/api/collections/{collection_id}/documents", tags=["Documents"], response_model=list[DocumentView])
async def list_documents(request: Request, collection_id: str) -> list[dict[str, Any]]:
    require_auth(request)
    require_uuid(collection_id)
    collection = ensure_collection(collection_id)
    docs = sorted(collection.documents.values(), key=lambda d: d.uploaded_at)
    return [document_summary(d) for d in docs]


@app.post(
    "/api/collections/{collection_id}/documents",
    tags=["Documents"],
    response_model=DocumentView,
    responses={
        202: {"model": DocumentView, "description": "Document ingestion was queued."},
        503: {"description": "The bounded ingestion queue is full; retry shortly."},
    },
)
async def upload_document(
    request: Request,
    collection_id: str,
    file: UploadFile = File(...),
    x_xsrf_token: str | None = Header(default=None),
) -> JSONResponse:
    username, roles, session = require_auth(request)
    require_role(roles, "ADMIN", "EDITOR")
    validate_csrf(request, session, x_xsrf_token)
    require_uuid(collection_id)
    collection = ensure_collection(collection_id)
    if ctx.settings.retrieval_provider == "openai-file-search" and not ctx.file_search.available:
        raise HTTPException(status_code=503, detail={"error": "Hosted File Search requires OPENAI_API_KEY and OPENAI_VECTOR_STORE_ID"})

    payload = await file.read(ctx.settings.max_upload_bytes + 1)
    if not payload:
        raise HTTPException(status_code=400, detail={"error": "Choose a non-empty document"})
    if len(payload) > ctx.settings.max_upload_bytes:
        raise HTTPException(status_code=400, detail={"error": "File exceeds the configured upload limit"})
    original_name = (file.filename or "document").replace("\\", "/").split("/")[-1]
    extension = original_name.rsplit(".", 1)[-1].lower() if "." in original_name else ""
    if extension not in {"pdf", "docx", "txt"}:
        raise HTTPException(status_code=400, detail={"error": "Supported formats are PDF, DOCX, and TXT"})
    if extension == "pdf" and not payload.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail={"error": "The file contents do not match the .pdf extension"})
    if extension == "docx" and not payload.startswith(b"PK"):
        raise HTTPException(status_code=400, detail={"error": "The file contents do not match the .docx extension"})

    digest = checksum(payload)
    existing = next((doc for doc in collection.documents.values() if doc.checksum == digest), None)
    if existing and existing.status != "FAILED":
        await ctx.store.record_audit(username, "DOCUMENT_UPLOAD_DEDUPLICATED", "document", existing.id, existing.status, "")
        return JSONResponse(status_code=200, content=document_summary(existing))
    if ctx.ingestion_slots.locked():
        raise HTTPException(status_code=503, detail={"error": "The ingestion queue is full; retry shortly"})
    await ctx.ingestion_slots.acquire()
    try:
        if existing and existing.status == "FAILED":
            existing.status = "PROCESSING"
            existing.error = None
            pending = existing
        else:
            pending = DocumentData(
                id=str(uuid.uuid4()),
                name=original_name,
                checksum=digest,
                uploaded_at=utc_now(),
                status="PROCESSING",
                error=None,
            )
            collection.documents[pending.id] = pending
        asyncio.create_task(
            ingest_with_slot(
                ctx.store,
                ctx.store.collections,
                ctx.settings.chunk_size,
                ctx.settings.chunk_overlap,
                collection_id,
                pending.id,
                extension,
                payload,
                username,
                "",
                embed_many=ctx.embedding_service.embed_many,
                file_search=ctx.file_search if ctx.settings.retrieval_provider == "openai-file-search" else None,
            )
        )
    except Exception:
        ctx.ingestion_slots.release()
        raise
    await ctx.store.record_audit(username, "DOCUMENT_INGESTION_QUEUED", "document", pending.id, "PROCESSING", "")
    return JSONResponse(status_code=202, content=document_summary(pending))


@app.delete("/api/collections/{collection_id}/documents/{document_id}", tags=["Documents"], status_code=204)
async def delete_document(
    request: Request,
    collection_id: str,
    document_id: str,
    x_xsrf_token: str | None = Header(default=None),
) -> Response:
    username, roles, session = require_auth(request)
    require_role(roles, "ADMIN", "EDITOR")
    validate_csrf(request, session, x_xsrf_token)
    require_uuid(collection_id)
    require_uuid(document_id)
    collection = ensure_collection(collection_id)
    if document_id not in collection.documents:
        raise HTTPException(status_code=404, detail={"error": "Document not found"})
    document = collection.documents[document_id]
    if ctx.settings.retrieval_provider == "openai-file-search" and document.openai_file_id:
        try:
            await asyncio.to_thread(ctx.file_search.delete_file, document.openai_file_id)
        except Exception as exc:
            log.warning("Hosted document cleanup failed: %s", exc)
            raise HTTPException(status_code=502, detail={"error": "Could not remove the hosted document"}) from exc
    del collection.documents[document_id]
    await ctx.store.record_audit(username, "DOCUMENT_DELETED", "document", document_id, "SUCCESS", "")
    return Response(status_code=204)


@app.post("/api/collections/{collection_id}/chat", tags=["Chat"], response_model=ChatResponse)
async def chat(
    request: Request,
    collection_id: str,
    payload: ChatRequest,
    x_xsrf_token: str | None = Header(default=None),
) -> dict[str, Any]:
    username, roles, session = require_auth(request)
    if not roles:
        raise HTTPException(status_code=401, detail={"error": "Sign in is required"})
    validate_csrf(request, session, x_xsrf_token)
    require_uuid(collection_id)
    collection = ensure_collection(collection_id)
    question = (payload.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail={"error": "Question is required"})
    if len(question) > 2000:
        raise HTTPException(status_code=400, detail={"error": "Question must be 2,000 characters or fewer"})

    history = payload.history[-8:]
    prior_user_messages = [item.text for item in history if item.role == "user"]
    prior_assistant_messages = [item.text for item in history if item.role == "assistant"]
    previous_answer = next((item.text for item in reversed(history) if item.role == "assistant"), "")
    ready_chunks = tuple(
        (document, chunk)
        for document in collection.documents.values()
        if document.status == "READY"
        for chunk in document.chunks
    )
    collection_text = [chunk.text for _, chunk in ready_chunks]
    collection_context = [
        f"[{document.name}, chunk {chunk.index}, page {chunk.page_number}]\n{chunk.text}"
        for document, chunk in ready_chunks
    ]
    standalone_question = correct_query_spelling(question, collection_text)
    contextual_question = contextualize_question(
        standalone_question,
        prior_user_messages,
        collection_text,
        previous_answers=prior_assistant_messages,
    )
    continuation = contextual_question != standalone_question
    contextual_question = correct_query_spelling(contextual_question, collection_text)
    broad_summary = is_collection_overview(contextual_question, collection_context)
    if ctx.settings.retrieval_provider == "openai-file-search":
        if not ctx.file_search.available:
            raise HTTPException(status_code=503, detail={"error": "Hosted File Search is not configured"})
        try:
            answer, file_results = await asyncio.to_thread(ctx.file_search.answer, contextual_question, collection_id, 20 if broad_summary else ctx.settings.retrieval_top_k)
        except Exception as exc:
            log.warning("Hosted file search failed: %s", exc)
            raise HTTPException(status_code=502, detail={"error": "Hosted document search is temporarily unavailable"}) from exc
        by_file = {doc.openai_file_id: doc for doc in collection.documents.values() if doc.openai_file_id}
        citations: list[dict[str, Any]] = []
        for result in file_results[: ctx.settings.retrieval_top_k]:
            doc = by_file.get(result.get("file_id"))
            if not doc or doc.status != "READY":
                continue
            excerpt = " ".join(str(part.get("text", "")) for part in result.get("content", []) if part.get("type") == "text").strip()
            if not excerpt:
                excerpt = doc.name
            citations.append({"documentId": doc.id, "documentName": doc.name, "chunk": len(citations) + 1, "page": None, "excerpt": excerpt if len(excerpt) <= 220 else excerpt[:217] + "…"})
        if not citations or not answer:
            message = "Documents are still being processed. Try again in a moment." if any(d.status == "PROCESSING" for d in collection.documents.values()) else "I couldn't find enough information in this collection to answer that."
            await ctx.store.record_audit(username, "QUESTION_ANSWERED", "collection", collection_id, "NO_EVIDENCE", "")
            return {"answer": answer if citations and answer else message, "sources": citations, "grounded": bool(citations and answer)}
        await ctx.store.record_audit(username, "QUESTION_ANSWERED", "collection", collection_id, "GROUNDED", "")
        return {"answer": answer, "sources": citations, "grounded": True}

    query_vec = await asyncio.to_thread(ctx.embedding_service.embed, contextual_question)
    matches = await asyncio.to_thread(
        ctx.retriever.retrieve,
        collection.id,
        ready_chunks,
        query_vec,
        contextual_question,
        max(ctx.settings.retrieval_top_k, 20) if broad_summary else ctx.settings.retrieval_top_k,
        ctx.settings.minimum_score,
    )

    if not matches:
        processing = any(doc.status == "PROCESSING" for doc in collection.documents.values())
        message = "Documents are still being processed. Try again in a moment." if processing else "I couldn't find enough information in this collection to answer that."
        await ctx.store.record_audit(username, "QUESTION_ANSWERED", "collection", collection_id, "PROCESSING" if processing else "NO_EVIDENCE", "")
        return {"answer": message, "sources": [], "grounded": False}

    context: list[str] = []
    for doc, chunk, _ in matches:
        page_info = f", page {chunk.page_number}" if chunk.page_number is not None else ""
        context.append(f"[{doc.name}, chunk {chunk.index}{page_info}]\n{chunk.text}")

    fallback_context = context
    if broad_summary:
        fallback_context = collection_context

    answer = ""
    if ctx.agent.available():
        try:
            answer = ctx.agent.answer(contextual_question, context)
        except Exception as exc:
            log.warning("OpenAI agent failed, using fallback: %s", exc)
            answer = extractive_answer(
                contextual_question,
                fallback_context,
                previous_answer if continuation else None,
                overview=broad_summary,
                section_context=collection_context,
            )
    else:
        answer = extractive_answer(
            contextual_question,
            fallback_context,
            previous_answer if continuation else None,
            overview=broad_summary,
            section_context=collection_context,
        )

    citations = [build_citation(m) for m in matches]
    await ctx.store.record_audit(username, "QUESTION_ANSWERED", "collection", collection_id, "GROUNDED", "")
    return {"answer": answer, "sources": citations, "grounded": True}


@app.get("/api/admin/audit", tags=["Audit"], response_model=AuditPage)
async def admin_audit(request: Request, limit: int = 100, beforeEventId: int = (2**63 - 1)) -> dict[str, Any]:
    _, roles, _ = require_auth(request)
    require_role(roles, "ADMIN")
    safe_limit = max(1, min(limit, 500))
    events = [event for event in ctx.store.audit_events if event.event_id < beforeEventId][:safe_limit]
    next_cursor = events[-1].event_id if len(events) == safe_limit else 0
    return {
        "items": [
            {
                "eventId": e.event_id,
                "occurredAt": e.occurred_at.isoformat(),
                "actor": e.actor,
                "action": e.action,
                "resourceType": e.resource_type,
                "resourceId": e.resource_id,
                "outcome": e.outcome,
                "requestId": e.request_id,
            }
            for e in events
        ],
        "nextCursor": next_cursor,
    }
