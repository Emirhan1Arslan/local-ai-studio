"""Sahte adapter: Faz 3 iskeletini gerçek model olmadan uçtan uca test etmek için.

params:
- steps / step_ms: adım sayısı ve süresi (progress akışı üretir)
- fail: true → kasıtlı hata (failed yolu)
- hang: true → sonsuz bekleme (kill/iptal testleri)
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any


class DummyAdapter:
    def load(self, model_path: str, **kwargs: Any) -> None:
        pass

    def infer(self, params: dict[str, Any]) -> Iterator[dict[str, Any]]:
        if params.get("fail"):
            raise RuntimeError("kasıtlı test hatası")
        if params.get("hang"):
            while True:  # yalnızca dışarıdan kill ile ölür
                time.sleep(1)
        steps = int(params.get("steps", 5))
        step_ms = int(params.get("step_ms", 50))
        for i in range(steps):
            time.sleep(step_ms / 1000)
            yield {
                "event": "progress",
                "value": (i + 1) / steps,
                "message": f"adım {i + 1}/{steps}",
            }
        result = {"steps": steps}
        if params.get("result_path"):  # asset kaydı testleri için sahte çıktı yolu
            result["path"] = params["result_path"]
        yield {"event": "done", "result": result}

    def unload(self) -> None:
        pass

    def vram_estimate(self, params: dict[str, Any]) -> int:
        return 0
