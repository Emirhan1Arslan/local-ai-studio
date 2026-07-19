"""Video düzenleme adapter'ı (Faz 8 adım 4): FFmpeg tabanlı kes/birleştir/hız/ses/format.

DİKKAT: Bu modül worker env'inde (envs/image ile paylaşılır — scheduler._ENV_ALIAS)
çalışır. 3.11+ sözdizimi KULLANMA. Ağır bağımlılık yok — yalnızca subprocess ile
FFmpeg çağırır (bkz. adapters/_ffmpeg.py, Faz 7'de kurulan build).

params:
- op: trim | concat | speed | add_audio | convert (zorunlu)

trim:
- video_path (zorunlu) · start (sn, vars. 0) · end (sn, opsiyonel — yoksa sona kadar)

concat:
- video_paths: [yol, ...] (zorunlu, en az 2) — hepsi aynı codec/çözünürlükte olmalı

speed:
- video_path (zorunlu) · factor: 0.25-4.0 (vars. 1.0'dan farklı olmalı; >1 hızlandırır)

add_audio:
- video_path · audio_path (ikisi de zorunlu) · mode: replace (vars.) | mix

convert:
- video_path (zorunlu) · format: mp4 (vars.) | mov | avi | gif

Tüm op'lar: output_path (zorunlu, format ile aynı uzantı)
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from typing import Any

from studio_inference.adapters._ffmpeg import ffmpeg_exe, has_audio_stream


def _run_ffmpeg(args: list[str]) -> None:
    cmd = [ffmpeg_exe(), "-y", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg başarısız: {proc.stderr[-2000:]}")


class VideoToolAdapter:
    def load(self, model_path: str, **kwargs: Any) -> None:
        pass  # model gerektirmez — saf FFmpeg subprocess çağrıları

    def infer(self, params: dict) -> Iterator[dict]:
        op = params.get("op", "")
        output_path = params.get("output_path", "")
        if not output_path:
            raise RuntimeError("output_path gerekli")

        yield {"event": "progress", "value": 0.1, "message": f"{op} çalışıyor…"}

        if op == "trim":
            video_path = str(params.get("video_path", ""))
            if not video_path:
                raise RuntimeError("video_path gerekli")
            start = float(params.get("start", 0) or 0)
            args = ["-ss", str(start), "-i", video_path]
            end = params.get("end")
            if end is not None:
                duration = max(0.0, float(end) - start)
                args += ["-t", str(duration)]
            args += ["-c", "copy", output_path]
            _run_ffmpeg(args)

        elif op == "concat":
            paths = list(params.get("video_paths") or [])
            if len(paths) < 2:
                raise RuntimeError("concat için en az 2 video_paths gerekli")
            import os
            import tempfile

            with tempfile.NamedTemporaryFile(
                "w", suffix=".txt", delete=False, encoding="utf-8"
            ) as f:
                for p in paths:
                    escaped = str(p).replace("'", "'\\''")
                    f.write(f"file '{escaped}'\n")
                list_path = f.name
            try:
                _run_ffmpeg(
                    ["-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", output_path]
                )
            finally:
                os.unlink(list_path)

        elif op == "speed":
            video_path = str(params.get("video_path", ""))
            if not video_path:
                raise RuntimeError("video_path gerekli")
            factor = float(params.get("factor", 1.0))
            if factor <= 0 or factor == 1.0:
                raise RuntimeError("factor 0'dan büyük ve 1.0'dan farklı olmalı")
            # setpts video hızını ayarlar (ters oran); atempo ses hızını (0.5-2.0
            # aralığı dışı için zincirleme gerekir — şimdilik tek atempo ile sınırlı)
            if not (0.5 <= factor <= 2.0):
                raise RuntimeError("bu ilk sürümde factor 0.5-2.0 aralığıyla sınırlı")
            if has_audio_stream(video_path):
                _run_ffmpeg(
                    [
                        "-i",
                        video_path,
                        "-filter_complex",
                        f"[0:v]setpts={1 / factor}*PTS[v];[0:a]atempo={factor}[a]",
                        "-map",
                        "[v]",
                        "-map",
                        "[a]",
                        output_path,
                    ]
                )
            else:
                # Ses izi yok (ör. LTX-Video çıktısı) — yalnız video hızlandırılır.
                _run_ffmpeg(
                    ["-i", video_path, "-filter:v", f"setpts={1 / factor}*PTS", output_path]
                )

        elif op == "add_audio":
            video_path = str(params.get("video_path", ""))
            audio_path = str(params.get("audio_path", ""))
            if not video_path or not audio_path:
                raise RuntimeError("video_path ve audio_path gerekli")
            mode = str(params.get("mode", "replace") or "replace")
            if mode not in ("replace", "mix"):
                raise RuntimeError(f"bilinmeyen mode: {mode!r} (replace|mix)")
            if mode == "replace":
                _run_ffmpeg(
                    [
                        "-i",
                        video_path,
                        "-i",
                        audio_path,
                        "-map",
                        "0:v",
                        "-map",
                        "1:a",
                        "-c:v",
                        "copy",
                        "-shortest",
                        output_path,
                    ]
                )
            elif has_audio_stream(video_path):
                _run_ffmpeg(
                    [
                        "-i",
                        video_path,
                        "-i",
                        audio_path,
                        "-filter_complex",
                        "[0:a][1:a]amix=inputs=2:duration=shortest[a]",
                        "-map",
                        "0:v",
                        "-map",
                        "[a]",
                        "-c:v",
                        "copy",
                        output_path,
                    ]
                )
            else:
                # Kaynak videoda (ör. LTX-Video çıktısı) ses izi yok — karışacak
                # ikinci bir iz olmadığından replace ile aynı sonuca düşer.
                _run_ffmpeg(
                    [
                        "-i",
                        video_path,
                        "-i",
                        audio_path,
                        "-map",
                        "0:v",
                        "-map",
                        "1:a",
                        "-c:v",
                        "copy",
                        "-shortest",
                        output_path,
                    ]
                )

        elif op == "convert":
            video_path = str(params.get("video_path", ""))
            if not video_path:
                raise RuntimeError("video_path gerekli")
            fmt = str(params.get("format", "mp4") or "mp4")
            if fmt not in ("mp4", "mov", "avi", "gif"):
                raise RuntimeError(f"bilinmeyen format: {fmt!r} (mp4|mov|avi|gif)")
            if fmt == "gif":
                # Basit ama kaliteli GIF: palet üretip iki geçişte uygula.
                _run_ffmpeg(
                    [
                        "-i",
                        video_path,
                        "-vf",
                        "fps=12,scale=480:-1:flags=lanczos,split[s0][s1];"
                        "[s0]palettegen[p];[s1][p]paletteuse",
                        output_path,
                    ]
                )
            else:
                _run_ffmpeg(["-i", video_path, output_path])

        else:
            raise RuntimeError(f"bilinmeyen op: {op!r} (trim|concat|speed|add_audio|convert)")

        yield {"event": "done", "result": {"path": output_path, "op": op}}

    def unload(self) -> None:
        pass

    def vram_estimate(self, params: dict) -> int:
        return 0  # yalnızca CPU/FFmpeg subprocess — GPU/VRAM kullanmaz
