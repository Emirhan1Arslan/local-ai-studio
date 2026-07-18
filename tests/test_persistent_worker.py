"""Faz 4: kalıcı worker (model cache) + proje sistemi v1 testleri."""

from __future__ import annotations

import time

import psutil
import pytest
from fastapi.testclient import TestClient
from studio_inference.scheduler import scheduler
from test_scheduler import _poll_status, _post_job


def test_worker_reused_between_jobs(client):
    """Aynı türdeki ardışık işler aynı worker sürecinde koşar (model cache temeli)."""
    j1 = _post_job(client, {"steps": 1, "step_ms": 10})
    _poll_status(client, j1, {"done"})
    pid1 = scheduler.worker_pid("dummy")
    assert pid1 is not None

    j2 = _post_job(client, {"steps": 1, "step_ms": 10})
    _poll_status(client, j2, {"done"})
    assert scheduler.worker_pid("dummy") == pid1


def test_cancel_respawns_fresh_worker(client):
    """İptal worker'ı öldürür; sonraki iş yeni worker'la sorunsuz koşar."""
    j1 = _post_job(client, {"hang": True})
    _poll_status(client, j1, {"running"})
    pid1 = scheduler.running_pid(j1)
    client.post(f"/v1/jobs/{j1}/cancel")
    _poll_status(client, j1, {"canceled"})

    j2 = _post_job(client, {"steps": 1, "step_ms": 10})
    _poll_status(client, j2, {"done"})
    pid2 = scheduler.worker_pid("dummy")
    assert pid2 is not None and pid2 != pid1


def test_worker_error_keeps_worker_alive(client):
    """Adapter hatası işi düşürür ama kalıcı worker'ı öldürmez."""
    j1 = _post_job(client, {"fail": True})
    _poll_status(client, j1, {"failed"})
    pid1 = scheduler.worker_pid("dummy")
    assert pid1 is not None and psutil.pid_exists(pid1)

    j2 = _post_job(client, {"steps": 1, "step_ms": 10})
    _poll_status(client, j2, {"done"})
    assert scheduler.worker_pid("dummy") == pid1


@pytest.fixture()
def short_idle_client(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDIO_DATA_DIR", str(tmp_path / "LocalAIStudio"))
    monkeypatch.setenv("STUDIO_WORKER_IDLE_TIMEOUT", "1")
    import studio_core.config as cfg
    import studio_core.db as db

    cfg._settings = None
    db._engine = None
    db._session_factory = None

    from studio_api.app import create_app

    with TestClient(create_app()) as c:
        yield c


def test_idle_worker_reaped(short_idle_client):
    """Boşta kalan worker idle timeout sonunda öldürülür (VRAM'i bırakır)."""
    j1 = _post_job(short_idle_client, {"steps": 1, "step_ms": 10})
    _poll_status(short_idle_client, j1, {"done"})
    pid = scheduler.worker_pid("dummy")
    assert pid is not None

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if scheduler.worker_pid("dummy") is None and not psutil.pid_exists(pid):
            return
        time.sleep(0.3)
    raise AssertionError("idle worker 10 sn içinde temizlenmedi")


def test_asset_registered_and_regenerate(client, tmp_path):
    """Dosya çıktılı iş asset olur; regenerate aynı parametrelerle yeni iş kuyruklar."""
    fake_out = tmp_path / "cikti.png"
    fake_out.write_bytes(b"png degil ama dosya")

    j1 = _post_job(client, {"steps": 1, "step_ms": 10, "result_path": str(fake_out)})
    _poll_status(client, j1, {"done"})

    # on_done kancası status yazımından hemen sonra koşar; asset için de kısa poll gerekir
    deadline = time.monotonic() + 5
    assets = []
    while time.monotonic() < deadline:
        assets = client.get("/v1/assets").json()
        if assets:
            break
        time.sleep(0.1)
    assert len(assets) == 1
    asset = assets[0]
    assert asset["path"] == str(fake_out)
    assert asset["meta"]["params"]["result_path"] == str(fake_out)

    projects = client.get("/v1/projects").json()
    assert any(p["name"] == "Varsayılan" for p in projects)

    r = client.post(f"/v1/assets/{asset['id']}/regenerate")
    job_id = r.json()["job_id"]
    _poll_status(client, job_id, {"done"})
    job = client.get(f"/v1/jobs/{job_id}").json()
    assert job["params"]["result_path"] == str(fake_out)
    assert job["project_id"] == asset["project_id"]
    # İkinci üretim de asset'e bağlandı
    assert len(client.get("/v1/assets").json()) == 2


def test_jobs_without_file_output_not_registered(client):
    j1 = _post_job(client, {"steps": 1, "step_ms": 10})
    _poll_status(client, j1, {"done"})
    assert client.get("/v1/assets").json() == []
