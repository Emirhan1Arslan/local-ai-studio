"""Faz 8 iskelet testleri: video işi kablolaması (gerçek LTX-Video GPU smoke testinde).

Test ortamında video worker env'i (torch/diffusers) yoktur → iş, kurulum talimatı
içeren anlaşılır bir hatayla FAILED olmalı; sunucu süreci torch import etmemeli.
videotool yalnızca FFmpeg subprocess kullanır — worker env'i olmadan da çalışır
(FFmpeg PATH'te yoksa FileNotFoundError'a düşer, bu da anlaşılır şekilde sarmalanır
değil ham haliyle patlar — test bunu tolere eder).
"""

from __future__ import annotations

import sys

from studio_inference.scheduler import worker_python
from test_scheduler import _poll_status


def test_video_kind_available(client):
    r = client.post(
        "/v1/jobs", json={"kind": "video", "params": {"mode": "t2v", "prompt": "a cat"}}
    )
    assert r.status_code == 200


def test_videotool_kind_available(client):
    r = client.post("/v1/jobs", json={"kind": "videotool", "params": {"op": "trim"}})
    assert r.status_code == 200


def test_server_process_has_no_torch(client):
    assert "torch" not in sys.modules


def test_worker_python_video_shares_image_env(client):
    # test veri dizininde envs/image yok → ana venv'e düşer (imgtool ile aynı desen)
    assert worker_python("video") == worker_python("image")
    assert worker_python("videotool") == worker_python("image")


def test_video_output_path_is_mp4(client):
    r = client.post(
        "/v1/jobs", json={"kind": "video", "params": {"mode": "t2v", "prompt": "a cat"}}
    )
    assert r.json()["params"]["output_path"].endswith(".mp4")


def test_videotool_output_path_follows_format(client):
    r = client.post(
        "/v1/jobs",
        json={"kind": "videotool", "params": {"op": "convert", "format": "gif"}},
    )
    assert r.json()["params"]["output_path"].endswith(".gif")


def test_videotool_output_path_defaults_mp4(client):
    r = client.post("/v1/jobs", json={"kind": "videotool", "params": {"op": "trim"}})
    assert r.json()["params"]["output_path"].endswith(".mp4")


def test_video_job_requires_known_mode(client):
    r = client.post(
        "/v1/jobs", json={"kind": "video", "params": {"mode": "yok", "prompt": "x"}}
    )
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "bilinmeyen mode" in d["error"]


def test_video_i2v_requires_init_image(client):
    r = client.post(
        "/v1/jobs", json={"kind": "video", "params": {"mode": "i2v", "prompt": "x"}}
    )
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "init_image" in d["error"] or "worker ortamı eksik" in d["error"]


def test_video_job_fails_gracefully_without_env(client):
    r = client.post(
        "/v1/jobs", json={"kind": "video", "params": {"mode": "t2v", "prompt": "a cat"}}
    )
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "worker ortamı eksik" in d["error"] or "diffusers" in d["error"]


def test_videotool_unknown_op_fails(client):
    r = client.post("/v1/jobs", json={"kind": "videotool", "params": {"op": "yok"}})
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "bilinmeyen op" in d["error"]


def test_videotool_trim_requires_video_path(client):
    r = client.post("/v1/jobs", json={"kind": "videotool", "params": {"op": "trim"}})
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "video_path" in d["error"]


def test_videotool_concat_requires_two_paths(client):
    r = client.post(
        "/v1/jobs",
        json={"kind": "videotool", "params": {"op": "concat", "video_paths": ["a.mp4"]}},
    )
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "en az 2" in d["error"]


def test_videotool_speed_requires_valid_factor(client):
    r = client.post(
        "/v1/jobs",
        json={
            "kind": "videotool",
            "params": {"op": "speed", "video_path": "a.mp4", "factor": 1.0},
        },
    )
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "factor" in d["error"]


def test_videotool_add_audio_requires_both_paths(client):
    r = client.post(
        "/v1/jobs",
        json={"kind": "videotool", "params": {"op": "add_audio", "video_path": "a.mp4"}},
    )
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "audio_path" in d["error"]


def test_videotool_convert_requires_known_format(client):
    r = client.post(
        "/v1/jobs",
        json={
            "kind": "videotool",
            "params": {"op": "convert", "video_path": "a.mp4", "format": "wmv"},
        },
    )
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "bilinmeyen format" in d["error"]
