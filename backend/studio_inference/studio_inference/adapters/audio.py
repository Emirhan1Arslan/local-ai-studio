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
- op: stt | stt_batch | tts (zorunlu)

stt:
- audio_path: ses dosyası (zorunlu)
- word_timestamps: bool (vars. true) — segmentlere kelime düzeyi zaman damgaları
  (faster-whisper'ın yerleşik cross-attention DTW hizalaması; JSON `segments[].words`)

- model_path: Model Browser'dan seçilen CTranslate2 ağırlık dosyası (ör.
  .../stt/faster-whisper-base/model.bin) — verilirse üst klasör yüklenir,
  model_size'ı geçersiz kılar
- model_size: tiny|base|small|medium|large-v3 (vars. "base") — model_path
  verilmezse HF Hub'dan kütüphanenin kendi önbelleğine iner
- language: dil kodu (vars. otomatik algıla, "tr" dahil çok dilli destek) ·
  task: transcribe|translate (vars. transcribe)
- çıktı: json (segment+zaman damgası) + txt + srt + vtt

stt_batch (toplu klasör transkripsiyonu):
- audio_dir: klasör (zorunlu) · recursive (vars. false) · stt ile aynı model/dil/task/
  word_timestamps parametreleri; output_path özet json, her dosyanın json/txt/srt/vtt'si
  onun yanına `<özet-adı>_<sıra>_<dosya>.…` olarak yazılır; bozuk dosya işi düşürmez

tts:
- text: seslendirilecek metin (zorunlu)
- engine: kokoro (vars.) | xtts | bark
- bark: voice_preset (vars. "v2/en_speaker_6"; Türkçe: "v2/tr_speaker_0".."9"; 13 dil) ·
  max_chars (vars. 220, cümle bölme eşiği) · model_path (Model Browser'daki
  bark-small/bark klasöründen bir dosya; boşsa suno/bark-small HF'den iner)
- model_path: Model Browser'dan seçilen ağırlık dosyası (kokoro:
  .../tts/Kokoro-82M/kokoro-v1_0.pth · xtts: .../tts/XTTS-v2/model.pth) —
  verilirse üst klasörden tamamen yerel yüklenir; verilmezse kütüphanenin
  kendi önbelleğine iner (ilk kullanımda ağ ister)
- kokoro: voice (vars. "af_heart") · lang_code (vars. "a", Amerikan İngilizcesi)
- xtts: language (vars. "en"; tr dahil ~17 dil) · ikisinden biri zorunlu:
  - speaker: 58 hazır sesten biri (model checkpoint'ine gömülü — ör. "Ana
    Florence"); klonlama değildir, consent gerekmez
  - speaker_wav: klonlanacak referans ses dosyası; verilirse consent=true
    ZORUNLU (sesin sahibinden izin alındığının beyanı, plan.md etik onay şartı)
- speed: 0.5-2.0 (vars. 1.0, yalnızca kokoro)
- çıktı: WAV (24kHz kokoro / 24kHz xtts)
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

from studio_inference.adapters._ffmpeg import ensure_ffmpeg_on_path as _ensure_ffmpeg_on_path


def _import_error(hint: str, exc: ImportError) -> RuntimeError:
    return RuntimeError(
        f"audio worker ortamı eksik ({hint}). Kurulum reçetesi: docs/worker-envs.md  |  " + str(exc)
    )


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
        elif op == "stt_batch":
            yield from self._stt_batch(params)
        elif op == "tts":
            yield from self._tts(params)
        else:
            raise RuntimeError(f"bilinmeyen op: {op!r} (stt|stt_batch|tts)")

    # -- STT (faster-whisper) ------------------------------------------------

    def _get_whisper(self, model_size_or_path: str):
        key = f"stt:{model_size_or_path}"
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
        model = WhisperModel(model_size_or_path, device=device, compute_type=compute_type)
        self._cache[key] = model
        return model

    def _stt(self, params: dict) -> Iterator[dict]:
        audio_path = params.get("audio_path", "")
        if not audio_path:
            raise RuntimeError("audio_path gerekli")
        output_path = params.get("output_path", "")
        if not output_path:
            raise RuntimeError("output_path gerekli")

        model_path = params.get("model_path", "")
        model_size = params.get("model_size") or "base"
        # Model Browser'dan seçilen ağırlık dosyası (ör. .../model.bin) verilmişse
        # CTranslate2'nin beklediği üst klasöre çözülür; yoksa preset adı kullanılır.
        model_target = os.path.dirname(model_path) if model_path else model_size
        language = params.get("language") or None
        task = params.get("task", "transcribe")
        if task not in ("transcribe", "translate"):
            raise RuntimeError(f"bilinmeyen task: {task!r} (transcribe|translate)")

        yield {"event": "progress", "value": 0.05, "message": "model yükleniyor…"}
        model = self._get_whisper(model_target)

        summary: dict = {}
        yield from self._transcribe_file(
            model,
            audio_path,
            output_path,
            language=language,
            task=task,
            word_timestamps=bool(params.get("word_timestamps", True)),
            out=summary,
        )
        yield {"event": "done", "result": {"op": "stt", **summary}}

    def _transcribe_file(
        self,
        model: Any,
        audio_path: str,
        output_path: str,
        *,
        language: str | None,
        task: str,
        word_timestamps: bool,
        out: dict,
        progress_scale: tuple[float, float] = (0.05, 0.99),
    ) -> Iterator[dict]:
        """Tek dosya: segmentler (+ kelime zaman damgaları) → json/txt/srt/vtt.
        Kelime hizalaması faster-whisper'ın yerleşik cross-attention DTW'si
        (`word_timestamps=True`) — WhisperX'in wav2vec2 hizalayıcısına ayrı env
        gerekmeden aynı çıktıyı verir (bkz. plan.md Faz 7 kalanları)."""
        lo, hi = progress_scale
        seg_iter, info = model.transcribe(
            audio_path, language=language, task=task, word_timestamps=word_timestamps
        )
        duration = float(getattr(info, "duration", 0.0) or 0.0)

        segments: list[dict] = []
        for seg in seg_iter:
            text = seg.text.strip()
            entry: dict[str, Any] = {"start": seg.start, "end": seg.end, "text": text}
            if word_timestamps and getattr(seg, "words", None):
                entry["words"] = [
                    {
                        "start": w.start,
                        "end": w.end,
                        "word": w.word.strip(),
                        "probability": round(float(w.probability), 3),
                    }
                    for w in seg.words
                ]
            segments.append(entry)
            frac = min(0.99, seg.end / duration) if duration > 0 else 0.5
            yield {
                "event": "progress",
                "value": lo + (hi - lo) * frac,
                "message": text,
                "delta": text + " ",
            }

        full_text = " ".join(s["text"] for s in segments)
        detected = getattr(info, "language", language or "")
        base = output_path.rsplit(".", 1)[0]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "source": audio_path,
                    "text": full_text,
                    "language": detected,
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

        out.update(
            {
                "path": output_path,
                "source": audio_path,
                "text": full_text,
                "language": detected,
                "duration": duration,
                "segment_count": len(segments),
                "word_count": sum(len(s.get("words", [])) for s in segments),
                "srt_path": base + ".srt",
                "vtt_path": base + ".vtt",
            }
        )

    def _stt_batch(self, params: dict) -> Iterator[dict]:
        """Toplu klasör transkripsiyonu (Faz 7 adım 2 kalanı): klasördeki tüm ses/video
        dosyaları tek işte, tek model yüklemesiyle; her dosyanın json/txt/srt/vtt'si
        `output_path`'in yanına `<ad>.<uzantı>` olarak yazılır, output_path özet json'dur."""
        audio_dir = str(params.get("audio_dir", "") or "")
        if not audio_dir or not os.path.isdir(audio_dir):
            raise RuntimeError("audio_dir gerekli (var olan bir klasör)")
        output_path = params.get("output_path", "")
        if not output_path:
            raise RuntimeError("output_path gerekli")
        files = list_audio_files(audio_dir, recursive=bool(params.get("recursive", False)))
        if not files:
            raise RuntimeError(f"klasörde ses dosyası yok: {audio_dir} ({', '.join(AUDIO_EXTS)})")

        model_path = params.get("model_path", "")
        model_target = (
            os.path.dirname(model_path) if model_path else (params.get("model_size") or "base")
        )
        language = params.get("language") or None
        task = params.get("task", "transcribe")
        word_timestamps = bool(params.get("word_timestamps", True))

        yield {
            "event": "progress",
            "value": 0.02,
            "message": f"model yükleniyor… ({len(files)} dosya)",
        }
        model = self._get_whisper(model_target)

        out_dir = os.path.dirname(os.path.abspath(output_path))
        stem = os.path.splitext(os.path.basename(output_path))[0]
        results: list[dict] = []
        for i, path in enumerate(files):
            name = os.path.splitext(os.path.basename(path))[0]
            per_file = os.path.join(out_dir, f"{stem}_{i + 1:03d}_{name}.json")
            lo = 0.05 + 0.94 * (i / len(files))
            hi = 0.05 + 0.94 * ((i + 1) / len(files))
            yield {
                "event": "progress",
                "value": lo,
                "message": f"[{i + 1}/{len(files)}] {os.path.basename(path)}",
            }
            summary: dict = {}
            try:
                yield from self._transcribe_file(
                    model,
                    path,
                    per_file,
                    language=language,
                    task=task,
                    word_timestamps=word_timestamps,
                    out=summary,
                    progress_scale=(lo, hi),
                )
                results.append({"ok": True, **summary})
            except Exception as exc:  # noqa: BLE001 - tek bozuk dosya toplu işi düşürmesin
                results.append({"ok": False, "source": path, "error": str(exc)})

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"audio_dir": audio_dir, "files": results}, f, ensure_ascii=False, indent=2)
        ok = sum(1 for r in results if r["ok"])
        yield {
            "event": "done",
            "result": {
                "path": output_path,
                "op": "stt_batch",
                "audio_dir": audio_dir,
                "count": len(results),
                "ok": ok,
                "failed": len(results) - ok,
                "files": [
                    {k: r.get(k) for k in ("ok", "source", "path", "language", "duration", "error")}
                    for r in results
                ],
            },
        }

    # -- TTS (Kokoro) ---------------------------------------------------------

    def _get_kokoro(self, lang_code: str, model_dir: str = ""):
        key = f"tts:{lang_code}:{model_dir}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        try:
            import torch
            from kokoro import KModel, KPipeline
        except ImportError as exc:
            raise _import_error("kokoro bulunamadı", exc) from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if model_dir:
            # Model Browser'dan seçilen yerel snapshot — HF Hub'a hiç gitmez.
            kmodel = (
                KModel(
                    repo_id="hexgrad/Kokoro-82M",
                    config=os.path.join(model_dir, "config.json"),
                    model=os.path.join(model_dir, "kokoro-v1_0.pth"),
                )
                .to(device)
                .eval()
            )
            pipeline = KPipeline(
                lang_code=lang_code, repo_id="hexgrad/Kokoro-82M", model=kmodel, device=device
            )
        else:
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
        elif engine == "bark":
            yield from self._tts_bark(params, text, output_path)
        else:
            raise RuntimeError(f"bilinmeyen engine: {engine!r} (kokoro|xtts|bark)")

    def _tts_kokoro(self, params: dict, text: str, output_path: str) -> Iterator[dict]:
        lang_code = params.get("lang_code") or "a"
        voice = params.get("voice") or "af_heart"
        speed = float(params.get("speed", 1.0))
        model_path = params.get("model_path", "")
        model_dir = os.path.dirname(model_path) if model_path else ""

        yield {"event": "progress", "value": 0.1, "message": "model yükleniyor…"}
        pipeline = self._get_kokoro(lang_code, model_dir)

        import numpy as np
        import soundfile as sf

        # Yerel snapshot'ta bu ses varsa (voices/<isim>.pt) HF Hub'a hiç gidilmez.
        voice_arg = voice
        if model_dir:
            local_voice = os.path.join(model_dir, "voices", f"{voice}.pt")
            if os.path.isfile(local_voice):
                voice_arg = local_voice

        yield {"event": "progress", "value": 0.3, "message": "sentezleniyor…"}
        chunks = []
        for _gs, _ps, audio in pipeline(text, voice=voice_arg, speed=speed):
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

    def _get_xtts(self, model_dir: str = ""):
        key = f"tts:xtts:{model_dir}"
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
        if model_dir:
            # Model Browser'dan seçilen yerel snapshot — coqui'nin kendi
            # ModelManager önbelleğine hiç gitmez.
            tts = TTS(
                model_path=model_dir,
                config_path=os.path.join(model_dir, "config.json"),
            ).to(device)
        else:
            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        self._cache[key] = tts
        return tts

    def _tts_xtts(self, params: dict, text: str, output_path: str) -> Iterator[dict]:
        speaker_wav = params.get("speaker_wav", "")
        speaker = params.get("speaker", "")
        if not speaker_wav and not speaker:
            raise RuntimeError(
                "speaker_wav ya da speaker gerekli: xtts motoru ya klonlanacak bir "
                "referans ses dosyası ya da 58 hazır sesten birinin adını ister"
            )
        if speaker_wav and not params.get("consent"):
            raise RuntimeError(
                "consent=true gerekli: ses klonlama yalnızca, seslendirilecek sesin "
                "sahibinden izin alındığını beyan ettiğinizde çalışır (bkz. plan.md "
                "etik onay şartı)."
            )
        language = params.get("language") or "en"
        model_path = params.get("model_path", "")
        model_dir = os.path.dirname(model_path) if model_path else ""

        yield {"event": "progress", "value": 0.1, "message": "XTTS-v2 model yükleniyor…"}
        tts = self._get_xtts(model_dir)

        yield {
            "event": "progress",
            "value": 0.4,
            "message": "ses klonlanıyor…" if speaker_wav else "sentezleniyor…",
        }
        tts_kwargs: dict[str, Any] = {"text": text, "language": language, "file_path": output_path}
        if speaker_wav:
            tts_kwargs["speaker_wav"] = speaker_wav
        else:
            tts_kwargs["speaker"] = speaker
        tts.tts_to_file(**tts_kwargs)

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
                "cloned": bool(speaker_wav),
            },
        }

    # -- TTS (Bark — transformers, çok dilli hazır sesler, müzik/efekt işaretleri) --

    def _get_bark(self, model_dir: str = ""):
        key = f"tts:bark:{model_dir}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        try:
            import torch
            from transformers import AutoProcessor, BarkModel
        except ImportError as exc:
            raise _import_error("transformers BarkModel bulunamadı", exc) from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        source = model_dir or BARK_DEFAULT_REPO
        processor = AutoProcessor.from_pretrained(source)
        model = BarkModel.from_pretrained(
            source, torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        model.eval()
        self._cache[key] = (processor, model)
        return processor, model

    def _tts_bark(self, params: dict, text: str, output_path: str) -> Iterator[dict]:
        """Bark (Suno, MIT): 13 dilde hazır konuşmacı presetleri (`v2/tr_speaker_0`…9),
        `[laughs]`/`♪` gibi ifade işaretleri. Tek geçişte ~13 sn üretir; uzun metin
        cümlelere bölünüp birleştirilir (`_split_sentences`). Klonlama yapmaz."""
        voice_preset = str(params.get("voice_preset") or "v2/en_speaker_6")
        model_path = params.get("model_path", "")
        model_dir = os.path.dirname(model_path) if model_path else ""

        yield {"event": "progress", "value": 0.1, "message": "Bark model yükleniyor…"}
        processor, model = self._get_bark(model_dir)

        import numpy as np
        import soundfile as sf
        import torch

        sample_rate = int(getattr(model.generation_config, "sample_rate", 24000))
        chunks = _split_sentences(text, max_chars=int(params.get("max_chars", 220)))
        pieces = []
        for i, chunk in enumerate(chunks):
            yield {
                "event": "progress",
                "value": 0.15 + 0.8 * (i / len(chunks)),
                "message": f"sentezleniyor… ({i + 1}/{len(chunks)})",
            }
            inputs = processor(chunk, voice_preset=voice_preset, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                audio = model.generate(**inputs)
            pieces.append(audio[0].float().cpu().numpy())
            if i + 1 < len(chunks):
                pieces.append(np.zeros(int(sample_rate * 0.25), dtype=np.float32))
        full_audio = np.concatenate(pieces)
        sf.write(output_path, full_audio, sample_rate)

        yield {
            "event": "done",
            "result": {
                "path": output_path,
                "op": "tts",
                "engine": "bark",
                "duration": len(full_audio) / sample_rate,
                "voice_preset": voice_preset,
                "chunks": len(chunks),
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
        if params.get("op") == "tts" and params.get("engine") == "bark":
            return 6000  # bark (büyük) fp16 ~5GB; bark-small ~2GB
        return {"stt": 3000, "stt_batch": 3000, "tts": 2000}.get(params.get("op", ""), 1000)


AUDIO_EXTS = (
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".webm",
    ".mp4",
    ".mov",
    ".mkv",
    ".aac",
    ".wma",
)
BARK_DEFAULT_REPO = "suno/bark-small"


def list_audio_files(directory: str, recursive: bool = False) -> list[str]:
    """Klasördeki ses/video dosyaları (uzantıya göre), deterministik sırada."""
    found: list[str] = []
    if recursive:
        for root, _dirs, files in os.walk(directory):
            found.extend(os.path.join(root, f) for f in files if f.lower().endswith(AUDIO_EXTS))
    else:
        for f in os.listdir(directory):
            p = os.path.join(directory, f)
            if os.path.isfile(p) and f.lower().endswith(AUDIO_EXTS):
                found.append(p)
    return sorted(found)


def _split_sentences(text: str, max_chars: int = 220) -> list[str]:
    """Metni cümle sınırlarından, her parça en çok `max_chars` olacak şekilde böler
    (Bark tek geçişte ~13 sn üretir; daha uzun girdi kesilir)."""
    import re

    sentences = [s.strip() for s in re.split(r"(?<=[.!?…])\s+", text.strip()) if s.strip()]
    chunks: list[str] = []
    current = ""
    for s in sentences:
        while len(s) > max_chars:  # tek başına çok uzun cümle: kelimeden kır
            cut = s.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            piece, s = s[:cut].strip(), s[cut:].strip()
            if current:
                chunks.append(current)
                current = ""
            chunks.append(piece)
        if not current:
            current = s
        elif len(current) + 1 + len(s) <= max_chars:
            current = current + " " + s
        else:
            chunks.append(current)
            current = s
    if current:
        chunks.append(current)
    return chunks or [text.strip()]
