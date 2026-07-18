"""Worker süreci (Faz 3/4): iş türü başına KALICI süreç, JSON-lines IPC (ADR-0004).

Döngü: stdin'den komut oku → çalıştır → done/error bas → sıradaki komutu bekle.
Adapter örneği (ve yüklediği model) komutlar arasında bellekte kalır → model cache.
İptal/VRAM baskısı = scheduler süreci öldürür; sonraki iş için yenisi doğar.

stdin  ← {"cmd": "run", "job_id": "...", "kind": "...", "params": {...}} | {"cmd": "exit"}
stdout → {"event": "started"} · {"event": "progress", ...} ·
         {"event": "done", "result": {...}} | {"event": "error", "error": "..."}

Worker DB'ye DOKUNMAZ; tüm durum güncellemeleri scheduler'dadır.
"""

from __future__ import annotations

import json
import sys

from studio_inference.adapters import get_adapter


def _emit(**event) -> None:
    print(json.dumps(event), flush=True)


def main() -> int:
    adapters: dict = {}
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except json.JSONDecodeError:
            _emit(event="error", error="geçersiz komut JSON'ı")
            continue

        op = cmd.get("cmd")
        if op == "exit":
            break
        if op != "run":
            _emit(event="error", error=f"bilinmeyen komut: {op!r}")
            continue

        kind = cmd.get("kind", "")
        adapter = adapters.get(kind)
        if adapter is None:
            adapter = get_adapter(kind)
            if adapter is None:
                _emit(event="error", error=f"adapter bulunamadı: {kind!r}")
                continue
            adapters[kind] = adapter

        _emit(event="started")
        try:
            for event in adapter.infer(cmd.get("params", {})):
                _emit(**event)
        except Exception as exc:  # hata işe yazılır, kalıcı süreç ayakta kalır
            _emit(event="error", error=str(exc))

    for adapter in adapters.values():
        try:
            adapter.unload()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
