"""Faz 5 iskelet testleri: imgtool işi kablolaması (gerçek araçlar GPU smoke testinde).

Test ortamında image worker env'i yoktur → iş, kurulum talimatı içeren anlaşılır
bir hatayla FAILED olmalı; sunucu süreci torch import etmemeli (ADR-0013).
"""

from __future__ import annotations

import os
import sys

from studio_inference.scheduler import worker_python
from test_scheduler import _poll_status


def test_imgtool_kind_available(client):
    r = client.post("/v1/jobs", json={"kind": "imgtool", "params": {"op": "bg_remove"}})
    assert r.status_code == 200  # tür kabul edilir; hata worker'da yakalanır


def test_server_process_has_no_torch(client):
    assert "torch" not in sys.modules


def test_imgtool_shares_image_env(client, tmp_path):
    """imgtool kendi venv'ini doğurmaz — 'image' venv'ini paylaşır (_ENV_ALIAS)."""
    # Test veri dizininde de gerçekten aynı env köküne düştüklerini kanıtla.
    from studio_core.config import get_settings

    settings = get_settings()
    sub = "Scripts" if os.name == "nt" else "bin"
    exe = "python.exe" if os.name == "nt" else "python"
    fake = settings.envs_dir / "image" / sub / exe
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text("")  # varlığı yeterli, worker_python içerik okumuyor
    try:
        assert worker_python("image") == str(fake)
        assert worker_python("imgtool") == str(fake)  # aynı venv'e düşer
    finally:
        fake.unlink()


def test_worker_python_fallback_no_env(client):
    assert worker_python("imgtool") == sys.executable


def test_imgtool_output_path_image_ops(client):
    for op in ("bg_remove", "upscale"):
        r = client.post(
            "/v1/jobs", json={"kind": "imgtool", "params": {"op": op, "image_path": "x.png"}}
        )
        assert r.json()["params"]["output_path"].endswith(".png")


def test_imgtool_output_path_text_ops(client):
    for op in ("caption", "ocr"):
        r = client.post(
            "/v1/jobs", json={"kind": "imgtool", "params": {"op": op, "image_path": "x.png"}}
        )
        assert r.json()["params"]["output_path"].endswith(".json")


def test_imgtool_job_fails_gracefully_without_env(client):
    r = client.post(
        "/v1/jobs",
        json={"kind": "imgtool", "params": {"op": "bg_remove", "image_path": "x.png"}},
    )
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "worker ortamı eksik" in d["error"] or "transformers" in d["error"]


def test_imgtool_job_requires_image_path(client):
    r = client.post("/v1/jobs", json={"kind": "imgtool", "params": {"op": "caption"}})
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "image_path" in d["error"]


def test_imgtool_job_requires_known_op(client):
    r = client.post(
        "/v1/jobs", json={"kind": "imgtool", "params": {"op": "yok", "image_path": "x.png"}}
    )
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "bilinmeyen op" in d["error"]
