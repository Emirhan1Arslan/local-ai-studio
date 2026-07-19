"""Adapter kayıt defteri: iş türü → EngineAdapter (ADR-0009, engine-agnostic ilke).

Kayıtlar noktalı yol olarak tutulur ve İHTİYAÇ ANINDA import edilir: torch/diffusers
yalnızca worker sürecinde yüklenir, API süreci hafif kalır (ADR-0013 slim çekirdek).
"""

from __future__ import annotations

import importlib

from studio_inference.adapters.base import EngineAdapter

_REGISTRY: dict[str, str] = {
    "dummy": "studio_inference.adapters.dummy:DummyAdapter",
    "image": "studio_inference.adapters.sdxl:SdxlAdapter",
    "imgtool": "studio_inference.adapters.imgtool:ImgToolAdapter",
    "llm": "studio_inference.adapters.llm:LlamaAdapter",
    "audio": "studio_inference.adapters.audio:AudioAdapter",
}


def get_adapter(kind: str) -> EngineAdapter | None:
    spec = _REGISTRY.get(kind)
    if spec is None:
        return None
    module_name, _, class_name = spec.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)()


def available_kinds() -> list[str]:
    return sorted(_REGISTRY)


def register(kind: str, spec: str) -> None:
    """spec: 'paket.modul:SinifAdi' — plugin'ler (Faz 13) bu kancayı kullanacak."""
    _REGISTRY[kind] = spec
