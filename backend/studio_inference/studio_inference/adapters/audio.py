"""Ses adapter'ı (Faz 7): STT (faster-whisper) + TTS (Kokoro veya XTTS-v2).

DİKKAT: Bu modül worker env'inde (envs/audio) çalışır. 3.11+ sözdizimi KULLANMA.
faster_whisper/kokoro/TTS/torch importları metod içindedir; API süreci bu
modülü hiç yüklemez.

Windows/RTX 5090 notu: ctranslate2 (faster-whisper'ın çalışma zamanı) ve
kokoro (torch tabanlı) hazır wheel'lerle kuruldu, CUDA'yı llama.cpp'nin
aksine EK BİR PATH düzeltmesi gerekmeden otomatik buldu (kendi CUDA çalışma
zamanı DLL'lerini taşıyorlar) — canlıda doğrulandı.

XTTS-v2 (coqui-tts) için torchcodec, FFmpeg'in paylaşımlı DLL'lerini (avcodec/
avformat/avutil...) çalışma zamanında arıyor; sistemde kurulu olmadığından
`~/LocalAIStudio/tools/ffmpeg/bin` altına elle indirildi (bkz. docs/worker-envs.md)
— `_ensure_ffmpeg_on_path()` bu dizini llm.py'deki torch-DLL çözümüyle aynı
desenle gerçek PATH'e ekliyor.

**LİSANS UYARISI:** XTTS-v2 model ağırlıkları Coqui Public Model License
(CPML) ile geliyor — yalnızca ticari OLMAYAN kullanım. UI, xtts motorunu
seçmeden önce bunu kullanıcıya açıkça göstermeli.

params:
- op: stt | tts (zorunlu)

stt:
- audio_path: ses dosyası (zorunlu)
- model_size: tiny|base|small|medium|large-v3 (vars. "base") ya da Model
  Browser'dan kayıtlı bir CTranslate2 model klasörü yolu
- language: dil kodu (vars. otomatik algıla, "tr" dahil çok dilli destek) ·
  task: transcribe|translate (vars. transcribe)
- çıktı: json (segment+zaman damgası) + txt + srt + vtt

tts:
- text: seslendirilecek metin (zorunlu)
- engine: kokoro (vars.) | xtts
- kokoro: voice (vars. "af_heart") · lang_code (vars. "a", Amerikan İngilizcesi)
- xtts: speaker_wav (klonlanacak referans ses, zorunlu) · language (vars. "en";
  tr dahil ~17 dil) · consent (zorunlu, true olmalı — sesin sahibinden izin
  alındığının beyanı, plan.md etik onay şartı)
- speed: 0.5-2.0 (vars. 1.0, yalnızca kokoro)
- çıktı: WAV (24kHz kokoro / 24kHz xtts)
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def _import_error(hint: str, exc: ImportError) -> RuntimeError:
    return RuntimeError(
        f"audio worker ortamı eksik ({hint}). "
        "Kurulum reçetesi: docs/worker-envs.md  |  " + str(exc)
    )


def _ensure_ffmpeg_on_path() -> None:
    """torchcodec'in (XTTS-v2/torch>=2.9 için) aradığı FFmpeg paylaşımlı
    DLL'lerini bulabilmesi için elle indirilen build'i gerçek PATH'e ekler."""
    data_dir = Path(os.environ.get("STUDIO_DATA_DIR") or (Path.home() / "LocalAIStudio"))
    ffmpeg_bin = data_dir / "tools" / "ffmpeg" / "bin"
    if not ffmpeg_bin.is_dir():
        return
    path = os.environ.get("PATH", "")
    if str(ffmpeg_bin) not in path:
        os.environ["PATH"] = str(ffmpeg_bin) + os.pathsep + path


def _fmt_ts_srt(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _make_srt(segments: list[dict]) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{_fmt_ts_srt(seg['start'])} --> {_fmt_ts_srt(seg['end'])}")
        lines.append(seg["text"].strip())
        lines.append("")
    return "\n".join(lines)


def _make_vtt(segments: list[dict]) -> str:
    lines = ["WEBVTT", ""]
    for seg in segments:
        start = _fmt_ts_srt(seg["start"]).replace(",", ".")
        end = _fmt_ts_srt(seg["end"]).replace(",", ".")
        lines.append(f"{start} --> {end}")
        lines.append(seg["text"].strip())
        lines.append("")
    return "\n".join(lines)


class AudioAdapter:
    def __init__(self) -> None:
        # anahtar: "stt:model_size" / "tts:lang_code" / "tts:xtts" -> yüklü nesne
        self._cache: dict[str, Any] = {}

    def load(self, model_path: str, **kwargs: Any) -> None:
        pass  # her op kendi modelini infer() içinde tembel yükler + cache'ler

    def infer(self, params: dict) -> Iterator[dict]:
        op = params.get("op", "")
        if op == "stt":
            yield from self._stt(params)
        elif op == "tts":
            yield from self._tts(params)
        else:
            raise RuntimeError(f"bilinmeyen op: {op!r} (stt|tts)")

    # -- STT (faster-whisper) ------------------------------------------------

    def _get_whisper(self, model_size: str):
        key = f"stt:{model_size}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        try:
            import torch
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise _import_error("faster_whisper bulunamadı", exc) from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._cache[key] = model
        return model

    def _stt(self, params: dict) -> Iterator[dict]:
        audio_path = params.get("audio_path", "")
        if not audio_path:
            raise RuntimeError("audio_path gerekli")
        output_path = params.get("output_path", "")
        if not output_path:
            raise RuntimeError("output_path gerekli")

        model_size = params.get("model_size") or "base"
        language = params.get("language") or None
        task = params.get("task", "transcribe")
        if task not in ("transcribe", "translate"):
            raise RuntimeError(f"bilinmeyen task: {task!r} (transcribe|translate)")

        yield {"event": "progress", "value": 0.05, "message": "model yükleniyor…"}
        model = self._get_whisper(model_size)

        seg_iter, info = model.transcribe(audio_path, language=language, task=task)
        duration = float(getattr(info, "duration", 0.0) or 0.0)

        segments: list[dict] = []
        for seg in seg_iter:
            text = seg.text.strip()
            segments.append({"start": seg.start, "end": seg.end, "text": text})
            value = min(0.99, seg.end / duration) if duration > 0 else 0.5
            yield {
                "event": "progress",
                "value": value,
                "message": text,
                "delta": text + " ",
            }

        full_text = " ".join(s["text"] for s in segments)
        base = output_path.rsplit(".", 1)[0]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "text": full_text,
                    "language": getattr(info, "language", language or ""),
                    "duration": duration,
                    "segments": segments,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        with open(base + ".txt", "w", encoding="utf-8") as f:
            f.write(full_text)
        with open(base + ".srt", "w", encoding="utf-8") as f:
            f.write(_make_srt(segments))
        with open(base + ".vtt", "w", encoding="utf-8") as f:
            f.write(_make_vtt(segments))

        yield {
            "event": "done",
            "result": {
                "path": output_path,
                "op": "stt",
                "text": full_text,
                "language": getattr(info, "language", language or ""),
                "duration": duration,
                "segment_count": len(segments),
                "srt_path": base + ".srt",
                "vtt_path": base + ".vtt",
            },
        }

    # -- TTS (Kokoro) ---------------------------------------------------------

    def _get_kokoro(self, lang_code: str):
        key = f"tts:{lang_code}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        try:
            import torch
            from kokoro import KPipeline
        except ImportError as exc:
            raise _import_error("kokoro bulunamadı", exc) from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = KPipeline(lang_code=lang_code, device=device)
        self._cache[key] = pipeline
        return pipeline

    def _tts(self, params: dict) -> Iterator[dict]:
        text = params.get("text", "").strip()
        if not text:
            raise RuntimeError("text gerekli")
        output_path = params.get("output_path", "")
        if not output_path:
            raise RuntimeError("output_path gerekli")

        engine = params.get("engine") or "kokoro"
        if engine == "kokoro":
            yield from self._tts_kokoro(params, text, output_path)
        elif engine == "xtts":
            yield from self._tts_xtts(params, text, output_path)
        else:
            raise RuntimeError(f"bilinmeyen engine: {engine!r} (kokoro|xtts)")

    def _tts_kokoro(self, params: dict, text: str, output_path: str) -> Iterator[dict]:
        lang_code = params.get("lang_code") or "a"
        voice = params.get("voice") or "af_heart"
        speed = float(params.get("speed", 1.0))

        yield {"event": "progress", "value": 0.1, "message": "model yükleniyor…"}
        pipeline = self._get_kokoro(lang_code)

        import numpy as np
        import soundfile as sf

        yield {"event": "progress", "value": 0.3, "message": "sentezleniyor…"}
        chunks = []
        for _gs, _ps, audio in pipeline(text, voice=voice, speed=speed):
            chunks.append(audio)
        if not chunks:
            raise RuntimeError("ses üretilemedi (boş çıktı)")
        full_audio = np.concatenate(chunks)
        sf.write(output_path, full_audio, 24000)

        duration = len(full_audio) / 24000
        yield {
            "event": "done",
            "result": {
                "path": output_path,
                "op": "tts",
                "engine": "kokoro",
                "duration": duration,
                "voice": voice,
            },
        }

    # -- TTS (XTTS-v2 — ses klonlama, çok dilli) ------------------------------

    def _get_xtts(self):
        key = "tts:xtts"
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        try:
            # Etkileşimsiz kabul: lisans metni kullanıcıya UI'da ayrıca gösterilir
            # (CPML — ticari olmayan kullanım); worker sürecinde prompt olamaz.
            os.environ.setdefault("COQUI_TOS_AGREED", "1")
            _ensure_ffmpeg_on_path()
            import torch
            from TTS.api import TTS
        except ImportError as exc:
            raise _import_error("coqui-tts (TTS.api) bulunamadı", exc) from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        self._cache[key] = tts
        return tts

    def _tts_xtts(self, params: dict, text: str, output_path: str) -> Iterator[dict]:
        speaker_wav = params.get("speaker_wav", "")
        if not speaker_wav:
            raise RuntimeError(
                "speaker_wav gerekli: xtts motoru klonlanacak bir referans ses dosyası ister"
            )
        if not params.get("consent"):
            raise RuntimeError(
                "consent=true gerekli: ses klonlama yalnızca, seslendirilecek sesin "
                "sahibinden izin alındığını beyan ettiğinizde çalışır (bkz. plan.md "
                "etik onay şartı)."
            )
        language = params.get("language") or "en"

        yield {"event": "progress", "value": 0.1, "message": "XTTS-v2 model yükleniyor…"}
        tts = self._get_xtts()

        yield {"event": "progress", "value": 0.4, "message": "ses klonlanıyor…"}
        tts.tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            language=language,
            file_path=output_path,
        )

        import soundfile as sf

        duration = float(sf.info(output_path).duration)
        yield {
            "event": "done",
            "result": {
                "path": output_path,
                "op": "tts",
                "engine": "xtts",
                "duration": duration,
                "language": language,
            },
        }

    def unload(self) -> None:
        self._cache = {}
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def vram_estimate(self, params: dict) -> int:
        if params.get("op") == "tts" and params.get("engine") == "xtts":
            return 4000
        return {"stt": 3000, "tts": 2000}.get(params.get("op", ""), 1000)
