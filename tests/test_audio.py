"""Faz 7 iskelet testleri: audio işi kablolaması (gerçek STT/TTS GPU smoke testinde).

Test ortamında audio worker env'i (faster_whisper/kokoro) yoktur → iş, kurulum
talimatı içeren anlaşılır bir hatayla FAILED olmalı; sunucu süreci torch import
etmemeli. Saf yardımcılar (SRT/VTT biçimlendirme) doğrudan test edilir.
"""

from __future__ import annotations

import sys

from studio_inference.scheduler import worker_python
from test_scheduler import _poll_status


def test_audio_kind_available(client):
    r = client.post("/v1/jobs", json={"kind": "audio", "params": {"op": "tts", "text": "merhaba"}})
    assert r.status_code == 200


def test_server_process_has_no_torch(client):
    assert "torch" not in sys.modules


def test_worker_python_audio_fallback(client):
    assert worker_python("audio") == sys.executable  # test veri dizininde env yok → düşer


def test_audio_output_path_tts_is_wav(client):
    r = client.post("/v1/jobs", json={"kind": "audio", "params": {"op": "tts", "text": "merhaba"}})
    assert r.json()["params"]["output_path"].endswith(".wav")


def test_audio_output_path_stt_is_json(client):
    r = client.post(
        "/v1/jobs", json={"kind": "audio", "params": {"op": "stt", "audio_path": "x.wav"}}
    )
    assert r.json()["params"]["output_path"].endswith(".json")


def test_audio_job_requires_known_op(client):
    r = client.post("/v1/jobs", json={"kind": "audio", "params": {"op": "yok"}})
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "bilinmeyen op" in d["error"]


def test_stt_requires_audio_path(client):
    r = client.post("/v1/jobs", json={"kind": "audio", "params": {"op": "stt"}})
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "audio_path" in d["error"] or "worker ortamı eksik" in d["error"]


def test_tts_requires_text(client):
    r = client.post("/v1/jobs", json={"kind": "audio", "params": {"op": "tts"}})
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "text" in d["error"] or "worker ortamı eksik" in d["error"]


def test_audio_job_fails_gracefully_without_env(client):
    r = client.post(
        "/v1/jobs",
        json={"kind": "audio", "params": {"op": "tts", "text": "merhaba dünya"}},
    )
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "worker ortamı eksik" in d["error"] or "kokoro" in d["error"]


def test_stt_model_path_reaches_adapter(client):
    """Model Browser'dan seçilen model_path, model_size'ın yerine geçer — istek
    boru hattını (job oluşturma, parametre doğrulama) kırmadan adapter'a ulaşmalı."""
    r = client.post(
        "/v1/jobs",
        json={
            "kind": "audio",
            "params": {
                "op": "stt",
                "audio_path": "x.wav",
                "model_path": r"C:\LocalAIStudio\models\stt\faster-whisper-base\model.bin",
            },
        },
    )
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "worker ortamı eksik" in d["error"] or "faster_whisper" in d["error"]


def test_kokoro_model_path_reaches_adapter(client):
    r = client.post(
        "/v1/jobs",
        json={
            "kind": "audio",
            "params": {
                "op": "tts",
                "engine": "kokoro",
                "text": "merhaba",
                "model_path": r"C:\LocalAIStudio\models\tts\Kokoro-82M\kokoro-v1_0.pth",
            },
        },
    )
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "worker ortamı eksik" in d["error"] or "kokoro" in d["error"]


def test_xtts_model_path_reaches_adapter(client):
    r = client.post(
        "/v1/jobs",
        json={
            "kind": "audio",
            "params": {
                "op": "tts",
                "engine": "xtts",
                "text": "merhaba",
                "speaker": "Ana Florence",
                "model_path": r"C:\LocalAIStudio\models\tts\XTTS-v2\model.pth",
            },
        },
    )
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "worker ortamı eksik" in d["error"] or "TTS" in d["error"]


def test_tts_unknown_engine_fails(client):
    r = client.post(
        "/v1/jobs",
        json={"kind": "audio", "params": {"op": "tts", "text": "merhaba", "engine": "yok"}},
    )
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "bilinmeyen engine" in d["error"]


def test_xtts_requires_speaker_wav_or_speaker(client):
    r = client.post(
        "/v1/jobs",
        json={
            "kind": "audio",
            "params": {"op": "tts", "engine": "xtts", "text": "merhaba", "consent": True},
        },
    )
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "speaker_wav" in d["error"] and "speaker" in d["error"]


def test_xtts_preset_speaker_does_not_require_consent(client):
    r = client.post(
        "/v1/jobs",
        json={
            "kind": "audio",
            "params": {
                "op": "tts",
                "engine": "xtts",
                "text": "merhaba",
                "speaker": "Ana Florence",
            },
        },
    )
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    # consent kontrolünü geçip model yükleme aşamasına ulaştığını doğrular
    # (bu ortamda xtts paketi kurulu değil → "worker ortamı eksik" ile düşer)
    assert "consent" not in d["error"]
    assert "worker ortamı eksik" in d["error"] or "TTS" in d["error"]


def test_xtts_requires_consent(client):
    r = client.post(
        "/v1/jobs",
        json={
            "kind": "audio",
            "params": {
                "op": "tts",
                "engine": "xtts",
                "text": "merhaba",
                "speaker_wav": "ref.wav",
            },
        },
    )
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "consent" in d["error"]


def test_make_srt_and_vtt_formatting():
    from studio_inference.adapters.audio import _make_srt, _make_vtt

    segments = [
        {"start": 0.0, "end": 1.5, "text": " Merhaba "},
        {"start": 1.5, "end": 3.25, "text": "dünya"},
    ]
    srt = _make_srt(segments)
    assert "1\n00:00:00,000 --> 00:00:01,500\nMerhaba" in srt
    assert "2\n00:00:01,500 --> 00:00:03,250\ndünya" in srt

    vtt = _make_vtt(segments)
    assert vtt.startswith("WEBVTT\n\n")
    assert "00:00:00.000 --> 00:00:01.500\nMerhaba" in vtt


def test_fmt_ts_srt_rounds_and_pads():
    from studio_inference.adapters.audio import _fmt_ts_srt

    assert _fmt_ts_srt(0) == "00:00:00,000"
    assert _fmt_ts_srt(65.5) == "00:01:05,500"
    assert _fmt_ts_srt(3661.001) == "01:01:01,001"


def test_list_audio_files_filters_and_recurses(tmp_path):
    from studio_inference.adapters.audio import list_audio_files

    (tmp_path / "b.WAV").write_bytes(b"x")
    (tmp_path / "a.mp3").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.webm").write_bytes(b"x")

    flat = [p.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for p in list_audio_files(str(tmp_path))]
    assert flat == ["a.mp3", "b.WAV"]
    deep = list_audio_files(str(tmp_path), recursive=True)
    assert len(deep) == 3 and any(p.endswith("c.webm") for p in deep)


def test_split_sentences_respects_max_chars():
    from studio_inference.adapters.audio import _split_sentences

    text = "Merhaba dünya. Bu bir test cümlesi! Üçüncü cümle burada? Dördüncü."
    chunks = _split_sentences(text, max_chars=40)
    assert all(len(c) <= 40 for c in chunks)
    assert " ".join(chunks) == text
    # tek başına çok uzun cümle kelimeden kırılır, boş parça üretmez
    long = _split_sentences("kelime " * 50, max_chars=30)
    assert all(0 < len(c) <= 30 for c in long)
    assert _split_sentences("   ", max_chars=10) == [""]


def test_stt_batch_output_is_json_and_requires_dir(client):
    r = client.post("/v1/jobs", json={"kind": "audio", "params": {"op": "stt_batch"}})
    assert r.status_code == 200
    assert r.json()["params"]["output_path"].endswith(".json")
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "audio_dir" in d["error"]


def test_tts_bark_engine_is_known(client):
    r = client.post(
        "/v1/jobs", json={"kind": "audio", "params": {"op": "tts", "engine": "yok", "text": "a"}}
    )
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "bark" in d["error"]  # bilinen motorlar listesinde artık bark da var
