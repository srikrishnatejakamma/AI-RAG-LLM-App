from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ChunkData:
    text: str
    index: int
    page_number: int | None
    vector: list[float]
    section_path: tuple[str, ...] = ()
    element_types: tuple[str, ...] = ()
    source_start: int | None = None
    source_end: int | None = None
    source_locations: tuple[str, ...] = ()


@dataclass
class DocumentData:
    id: str
    name: str
    checksum: str
    uploaded_at: datetime
    status: str
    error: str | None
    chunks: list[ChunkData] = field(default_factory=list)
    openai_file_id: str | None = None

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)


@dataclass
class CollectionData:
    id: str
    name: str
    created_at: datetime
    documents: dict[str, DocumentData] = field(default_factory=dict)


@dataclass
class AuditEvent:
    event_id: int
    occurred_at: datetime
    actor: str
    action: str
    resource_type: str
    resource_id: str
    outcome: str
    request_id: str


@dataclass
class SessionData:
    username: str
    roles: list[str]
    csrf_token: str
    created_at: datetime
