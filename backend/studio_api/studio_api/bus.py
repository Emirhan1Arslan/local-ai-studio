"""Süreç içi olay veriyolu: publish() → tüm bağlı WS istemcilerine fan-out.

Faz 3'te scheduler/worker olayları da buradan akacak. Yavaş istemci, dolu kuyrukta
en eski olayı düşürür (backpressure: UI canlı kalır, tarih WS'in işi değildir).
"""

from __future__ import annotations

import asyncio

from studio_core.events import Event, EventTopic, make_event


class EventBus:
    def __init__(self, queue_size: int = 1000) -> None:
        self._subscribers: set[asyncio.Queue[Event]] = set()
        self._queue_size = queue_size

    def subscribe(self) -> asyncio.Queue[Event]:
        q: asyncio.Queue[Event] = asyncio.Queue(maxsize=self._queue_size)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[Event]) -> None:
        self._subscribers.discard(q)

    def publish(self, event: Event) -> None:
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                except asyncio.QueueEmpty:
                    pass

    def emit(self, topic: EventTopic, **data) -> None:
        self.publish(make_event(topic, **data))


bus = EventBus()
