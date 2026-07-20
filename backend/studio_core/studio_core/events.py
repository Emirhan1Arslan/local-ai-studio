"""WS olay veriyolu şemaları (ADR-0003): job.*, download.*, system.*, log.

UI, SDK ve MCP istemcileri aynı olayları aynı formatta dinler. Şema sürümlüdür;
kırıcı değişiklik = yeni sürüm alanı, sessiz değişiklik yok.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

EVENT_SCHEMA_VERSION = 1

EventTopic = Literal[
    "job.queued",
    "job.started",
    "job.progress",
    "job.done",
    "job.failed",
    "job.canceled",
    "job.paused",
    "download.queued",
    "download.progress",
    "download.done",
    "download.failed",
    "download.paused",
    "download.canceled",
    "workflow.started",
    "workflow.node_started",
    "workflow.node_done",
    "workflow.node_skipped",
    "workflow.node_failed",
    "workflow.done",
    "workflow.failed",
    "model.added",
    "model.removed",
    "model.updated",
    "system.telemetry",
    "system.status",
    "log",
]


class Event(BaseModel):
    """Veriyolundaki tek zarf tipi. `data` içeriği topic'e göre değişir."""

    v: int = EVENT_SCHEMA_VERSION
    topic: EventTopic
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    data: dict[str, Any] = Field(default_factory=dict)


def make_event(topic: EventTopic, **data: Any) -> Event:
    return Event(topic=topic, data=data)
