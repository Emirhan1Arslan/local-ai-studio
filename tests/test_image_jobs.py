"""Faz 4 iskelet testleri: image işi kablolaması (gerçek üretim GPU smoke testinde).

Test ortamında image worker env'i yoktur → iş, kurulum talimatı içeren anlaşılır
bir hatayla FAILED olmalı; sunucu süreci torch import etmemeli.
"""

from __future__ import annotations

import sys

from studio_inference.scheduler import worker_python
from test_scheduler import _poll_status


def test_image_kind_available(client):
    r = client.post("/v1/jobs", json={"kind": "image", "params": {}})
    assert r.status_code == 200  # tür kabul edilir; hata worker'da yakalanır


def test_server_process_has_no_torch(client):
    assert "torch" not in sys.modules  # API süreci slim kalmalı (ADR-0013)


def test_worker_python_fallback(client):
    # Test veri dizininde env yok → ana venv'e düşer
    assert worker_python("image") == sys.executable
    assert worker_python("dummy") == sys.executable


def test_image_job_output_path_filled(client):
    r = client.post(
        "/v1/jobs",
        json={"kind": "image", "params": {"model_path": "yok.safetensors", "prompt": "test"}},
    )
    body = r.json()
    assert body["params"]["output_path"].endswith(".png")


def test_image_job_fails_gracefully_without_env(client):
    r = client.post(
        "/v1/jobs",
        json={"kind": "image", "params": {"model_path": "yok.safetensors", "prompt": "test"}},
    )
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "worker ortamı eksik" in d["error"] or "diffusers" in d["error"]


def test_image_job_requires_model_path(client):
    r = client.post("/v1/jobs", json={"kind": "image", "params": {"prompt": "test"}})
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "model_path" in d["error"]