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
    r = client.post(
        "/v1/jobs", json={"kind": "audio", "params": {"op": "tts", "text": "merhaba"}}
    )
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


def test_tts_unknown_engine_fails(client):
    r = client.post(
        "/v1/jobs",
        json={"kind": "audio", "params": {"op": "tts", "text": "merhaba", "engine": "yok"}},
    )
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "bilinmeyen engine" in d["error"]


def test_xtts_requires_speaker_wav(client):
    r = client.post(
        "/v1/jobs",
        json={
            "kind": "audio",
            "params": {"op": "tts", "engine": "xtts", "text": "merhaba", "consent": True},
        },
    )
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "speaker_wav" in d["error"]


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
