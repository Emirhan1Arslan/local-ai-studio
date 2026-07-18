"""Adapter kayıt defteri: iş türü → EngineAdapter (ADR-0009, engine-agnostic ilke)."""

from __future__ import annotations

from collections.abc import Callable

from studio_inference.adapters.base import EngineAdapter
from studio_inference.adapters.dummy import DummyAdapter

_REGISTRY: dict[str, Callable[[], EngineAdapter]] = {
    "dummy": DummyAdapter,
}


def get_adapter(kind: str) -> EngineAdapter | None:
    factory = _REGISTRY.get(kind)
    return factory() if factory else None


def available_kinds() -> list[str]:
    return sorted(_REGISTRY)


def register(kind: str, factory: Callable[[], EngineAdapter]) -> None:
    _REGISTRY[kind] = factory
