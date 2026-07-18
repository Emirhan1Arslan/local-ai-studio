"""Backend adapter sözleşmesi (Faz 3, plan §5): load / infer(stream) / unload / vram_estimate.

Tüm üretim motorları (Diffusers, llama.cpp, ONNX, TensorRT, harici ComfyUI…) bu arayüzle
bağlanır. `infer` olay sözlüğü akıtır; son olay `{"event": "done", "result": {...}}` olmalıdır.
Worker süreci bu olayları JSON-lines olarak stdout'a geçirir.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable

InferEvent = dict[str, Any]


@runtime_checkable
class EngineAdapter(Protocol):
    def load(self, model_path: str, **kwargs: Any) -> None:
        """Modeli belleğe/VRAM'e yükler."""
        ...

    def infer(self, params: dict[str, Any]) -> Iterator[InferEvent]:
        """Üretimi çalıştırır; progress olayları ve en sonda done olayı üretir."""
        ...

    def unload(self) -> None:
        """Modeli boşaltır; VRAM'in gerçekten geri döndüğünü garanti eder."""
        ...

    def vram_estimate(self, params: dict[str, Any]) -> int:
        """Tahmini VRAM ihtiyacı (MB). Scheduler kapasite kontrolünde kullanır."""
        ...
