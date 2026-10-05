from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CreateCollectionRequest(BaseModel):
    name: str


class ChatRequest(BaseModel):
    question: str


class CollectionView(BaseModel):
    id: str
    name: str
    created_at: datetime = Field(alias="createdAt")
    document_count: int = Field(alias="documentCount", ge=0)

    model_config = ConfigDict(populate_by_name=True)


class DocumentView(BaseModel):
    id: str
    name: str
    status: Literal["PROCESSING", "READY", "FAILED"]
    chunk_count: int = Field(alias="chunkCount", ge=0)
    uploaded_at: datetime = Field(alias="uploadedAt")
    error: str | None = None

    model_config = ConfigDict(populate_by_name=True)


class Citation(BaseModel):
    document_id: str = Field(alias="documentId")
    document_name: str = Field(alias="documentName")
    chunk: int = Field(ge=1)
    page: int | None = Field(default=None, ge=1)
    excerpt: str

    model_config = ConfigDict(populate_by_name=True)


class ChatResponse(BaseModel):
    answer: str
    sources: list[Citation]
    grounded: bool


class SessionView(BaseModel):
    username: str
    roles: list[str]


class HealthView(BaseModel):
    status: Literal["ok", "degraded"]
    storage: str
    embeddingProvider: str
    answerProvider: str
    answerProviderStatus: str
    answerProviderMessage: str
    retrievalProvider: str
    retrievalProviderStatus: str
    retrievalProviderMessage: str
    authOneTimeAdminCredentials: str
    authAdminUsername: str


class CsrfView(BaseModel):
    headerName: str
    parameterName: str
    token: str


class AuditEventView(BaseModel):
    event_id: int = Field(alias="eventId", ge=1)
    occurred_at: datetime = Field(alias="occurredAt")
    actor: str
    action: str
    resource_type: str = Field(alias="resourceType")
    resource_id: str = Field(alias="resourceId")
    outcome: str
    request_id: str = Field(alias="requestId")

    model_config = ConfigDict(populate_by_name=True)


class AuditPage(BaseModel):
    items: list[AuditEventView]
    next_cursor: int = Field(alias="nextCursor", ge=0)

    model_config = ConfigDict(populate_by_name=True)


class ReadSourceToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_number: int = Field(..., ge=1, description="1-based source passage number to inspect")


class FindSourcesToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keyword: str = Field(..., min_length=1, description="Keyword to search in the retrieved source passages")
