from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from .domain import AuditEvent, CollectionData


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


class MemoryStore:
    def __init__(self) -> None:
        self.collections: dict[str, CollectionData] = {}
        self.audit_events: list[AuditEvent] = []
        self.audit_seq = 0
        self.lock = asyncio.Lock()

    async def record_audit(
        self,
        actor: str,
        action: str,
        resource_type: str,
        resource_id: str,
        outcome: str,
        request_id: str = "",
    ) -> AuditEvent:
        async with self.lock:
            self.audit_seq += 1
            event = AuditEvent(
                event_id=self.audit_seq,
                occurred_at=utc_now(),
                actor=actor[:120],
                action=action[:120],
                resource_type=resource_type[:120],
                resource_id=(resource_id or "")[:120],
                outcome=outcome[:120],
                request_id=(request_id or "")[:120],
            )
            self.audit_events.insert(0, event)
            self.audit_events = self.audit_events[:10_000]
            return event
