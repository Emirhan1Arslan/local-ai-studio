"""Telemetri servisi (Faz 3): saniyelik GPU/VRAM/CPU/RAM akışı → system.telemetry olayı.

Torch'suz: NVIDIA için nvidia-smi sorgulanır (util + bellek); GPU yoksa yalnız CPU/RAM.
Scheduler, VRAM kapasite kontrolünde son anlık görüntüyü kullanabilir.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from collections.abc import Callable

import psutil
from studio_core.logging import get_logger

log = get_logger("studio_inference.telemetry")

EmitFn = Callable[..., None]


def _gpu_snapshot() -> list[dict]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "util_percent": int(float(parts[2])),
                    "vram_used_mb": int(float(parts[3])),
                    "vram_total_mb": int(float(parts[4])),
                }
            )
    return gpus


class TelemetryService:
    def __init__(self, emit: EmitFn, interval: float = 2.0) -> None:
        self._emit = emit
        self._interval = interval
        self._task: asyncio.Task | None = None
        self.last: dict | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self) -> None:
        while True:
            snapshot = await asyncio.to_thread(self._snapshot)
            self.last = snapshot
            self._emit("system.telemetry", **snapshot)
            await asyncio.sleep(self._interval)

    def _snapshot(self) -> dict:
        vm = psutil.virtual_memory()
        return {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_used_mb": (vm.total - vm.available) // (1024 * 1024),
            "ram_total_mb": vm.total // (1024 * 1024),
            "gpus": _gpu_snapshot(),
        }

    def gpu_free_mb(self, index: int = 0) -> int | None:
        if not self.last or not self.last.get("gpus"):
            return None
        for gpu in self.last["gpus"]:
            if gpu["index"] == index:
                return gpu["vram_total_mb"] - gpu["vram_used_mb"]
        return None


telemetry: TelemetryService | None = None
