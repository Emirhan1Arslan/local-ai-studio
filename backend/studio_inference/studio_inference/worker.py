"""Worker süreci (Faz 3): iş türü başına ayrı süreç, JSON-lines IPC (ADR-0004).

stdin  ← {"cmd": "run", "job_id": "...", "kind": "...", "params": {...}}
stdout → {"event": "started"} · {"event": "progress", ...} · {"event": "done", "result": {...}}
         {"event": "error", "error": "..."}

Worker DB'ye DOKUNMAZ; tüm durum güncellemeleri scheduler'dadır. Böylece worker'ı
kill -9 ile öldürmek veri bütünlüğünü bozmaz, VRAM süreçle birlikte kesin geri döner.
"""

from __future__ import annotations

import json
import sys

from studio_inference.adapters import get_adapter


def _emit(**event) -> None:
    print(json.dumps(event), flush=True)


def main() -> int:
    line = sys.stdin.readline()
    try:
        cmd = json.loads(line)
    except json.JSONDecodeError:
        _emit(event="error", error="geçersiz komut JSON'ı")
        return 2
    if cmd.get("cmd") != "run":
        _emit(event="error", error=f"bilinmeyen komut: {cmd.get('cmd')!r}")
        return 2

    kind = cmd.get("kind", "")
    adapter = get_adapter(kind)
    if adapter is None:
        _emit(event="error", error=f"adapter bulunamadı: {kind!r}")
        return 2

    _emit(event="started")
    try:
        for event in adapter.infer(cmd.get("params", {})):
            _emit(**event)
    except Exception as exc:  # hata worker'da kalır, sürece değil işe yazılır
        _emit(event="error", error=str(exc))
        return 1
    finally:
        adapter.unload()
    return 0


if __name__ == "__main__":
    sys.exit(main())
