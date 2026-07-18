"""Faz 3 DoD testleri: öncelik sırası, çalışan iş iptali, kill -9 dayanıklılığı, pause.

Gerçek GPU/model gerekmez: dummy adapter'lı gerçek worker SÜREÇLERİ kullanılır —
süreç izolasyonu ve IPC bire bir üretimdeki gibi çalışır.
"""

from __future__ import annotations

import time

import psutil
from studio_inference.scheduler import scheduler


def _poll_status(client, job_id: str, want: set[str], timeout: float = 20.0) -> dict:
    deadline = time.monotonic() + timeout
    last: dict = {}
    while time.monotonic() < deadline:
        last = client.get(f"/v1/jobs/{job_id}").json()
        if last["status"] in want:
            return last
        time.sleep(0.1)
    raise AssertionError(f"{want} beklenirken durum: {last.get('status')!r} ({last.get('error')})")


def _post_job(client, params: dict, priority: int = 0) -> str:
    r = client.post("/v1/jobs", json={"kind": "dummy", "params": params, "priority": priority})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def test_unknown_kind_rejected(client):
    r = client.post("/v1/jobs", json={"kind": "yok-boyle-motor"})
    assert r.status_code == 400


def test_priority_order(client):
    """10 sahte iş öncelik sırasıyla işlenir (Faz 3 DoD)."""
    # Tek slotu süresiz bir işle tut: 9 iş kuyruğa girmeden hiçbiri başlayamaz.
    blocker = _post_job(client, {"hang": True}, priority=100)
    _poll_status(client, blocker, {"running"})
    priorities = [3, 7, 1, 9, 5, 0, 8, 2, 6]
    ids = {p: _post_job(client, {"steps": 1, "step_ms": 10}, priority=p) for p in priorities}

    # Hepsi kuyruktayken slotu serbest bırak → sıralama artık deterministik.
    assert client.post(f"/v1/jobs/{blocker}/cancel").json()["ok"]
    for job_id in ids.values():
        _poll_status(client, job_id, {"done"})

    jobs = {j["id"]: j for j in client.get("/v1/jobs", params={"limit": 50}).json()}
    started = sorted((jobs[jid]["started_at"], p) for p, jid in ids.items())
    executed_order = [p for _, p in started]
    assert executed_order == sorted(priorities, reverse=True), executed_order


def test_cancel_running_job_kills_worker(client):
    """Çalışan iş iptal edilir; worker süreci ölür (VRAM garantisi = süreç kill)."""
    job_id = _post_job(client, {"hang": True})
    _poll_status(client, job_id, {"running"})
    pid = scheduler.running_pid(job_id)
    assert pid is not None and psutil.pid_exists(pid)

    assert client.post(f"/v1/jobs/{job_id}/cancel").json()["ok"]
    d = _poll_status(client, job_id, {"canceled"})
    assert d["status"] == "canceled"

    deadline = time.monotonic() + 5
    while psutil.pid_exists(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not psutil.pid_exists(pid)


def test_worker_error_marks_failed(client):
    job_id = _post_job(client, {"fail": True})
    d = _poll_status(client, job_id, {"failed"})
    assert "kasıtlı test hatası" in d["error"]


def test_kill9_worker_does_not_kill_app(client):
    """Worker'ı zorla öldürmek uygulamayı düşürmez; iş failed olur (Faz 3 DoD)."""
    job_id = _post_job(client, {"hang": True})
    _poll_status(client, job_id, {"running"})
    pid = scheduler.running_pid(job_id)
    assert pid is not None

    psutil.Process(pid).kill()  # kill -9 eşdeğeri (TerminateProcess)

    d = _poll_status(client, job_id, {"failed"})
    assert "çıkış kodu" in d["error"]

    # Uygulama ayakta: yeni iş sorunsuz çalışır.
    next_id = _post_job(client, {"steps": 1, "step_ms": 10})
    assert _poll_status(client, next_id, {"done"})["status"] == "done"


def test_pause_resume_queued(client):
    blocker = _post_job(client, {"steps": 1, "step_ms": 1200})
    paused_id = _post_job(client, {"steps": 1, "step_ms": 10}, priority=50)
    assert client.post(f"/v1/jobs/{paused_id}/pause").json()["ok"]
    other_id = _post_job(client, {"steps": 1, "step_ms": 10}, priority=0)

    _poll_status(client, other_id, {"done"})
    assert client.get(f"/v1/jobs/{paused_id}").json()["status"] == "paused"

    assert client.post(f"/v1/jobs/{paused_id}/resume").json()["ok"]
    _poll_status(client, paused_id, {"done"})
    _poll_status(client, blocker, {"done"})
