"""Faz 10 /v1/workflows API testleri: kaydet/yükle/çalıştır uçtan uca.

Yalnızca process-içi (GPU'suz) core.prompt/core.save düğümlerini kullanır —
image.generate/upscale/bg_remove'un gerçek Job sistemine dispatch'i zaten
tests/test_workflow_dispatch.py'de dummy adapter ile ayrıca doğrulanıyor.
Buradaki amaç: API/DB/arka-plan-görev/cache/hata katmanının bütünüyle
çalıştığını göstermek (DoD: grafiği kurulur ve çalışır; ikinci çalıştırmada
değişmeyen dallar cache'ten gelir; bozuk workflow'a anlaşılır hata verir)."""

from __future__ import annotations

import time


def _prompt_save_graph(text: str = "merhaba", path: str = "C:/out/note.txt") -> dict:
    return {
        "nodes": [
            {"id": "p", "type": "core.prompt", "params": {"text": text}},
            {"id": "s", "type": "core.save", "params": {"path": path}},
        ],
        "edges": [
            {"id": "e1", "from_node": "p", "from_port": "text", "to_node": "s", "to_port": "value"},
        ],
    }


def _poll_run(client, run_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    last: dict = {}
    while time.monotonic() < deadline:
        last = client.get(f"/v1/workflows/runs/{run_id}").json()
        if last["status"] != "running":
            return last
        time.sleep(0.05)
    raise AssertionError(f"run bitmedi: {last}")


def test_create_and_get_workflow(client, tmp_path):
    dest = str(tmp_path / "note.txt")
    r = client.post("/v1/workflows", json={"name": "t1", "graph": _prompt_save_graph(path=dest)})
    assert r.status_code == 200, r.text
    wf = r.json()
    assert wf["name"] == "t1"

    got = client.get(f"/v1/workflows/{wf['id']}")
    assert got.status_code == 200
    assert got.json()["id"] == wf["id"]


def test_get_unknown_workflow_404s(client):
    assert client.get("/v1/workflows/yok-boyle-id").status_code == 404


def test_create_rejects_unknown_node_type(client):
    graph = {"nodes": [{"id": "a", "type": "yok.boyle.tur"}], "edges": []}
    r = client.post("/v1/workflows", json={"name": "bozuk", "graph": graph})
    assert r.status_code == 400
    assert "bilinmeyen düğüm türü" in r.json()["error"]["message"]


def test_create_rejects_cycle(client):
    graph = {
        "nodes": [
            {"id": "a", "type": "core.save", "params": {"path": "a.txt"}},
            {"id": "b", "type": "core.save", "params": {"path": "b.txt"}},
        ],
        "edges": [
            {"id": "e1", "from_node": "a", "from_port": "path", "to_node": "b", "to_port": "value"},
            {"id": "e2", "from_node": "b", "from_port": "path", "to_node": "a", "to_port": "value"},
        ],
    }
    r = client.post("/v1/workflows", json={"name": "dongu", "graph": graph})
    assert r.status_code == 400
    assert "döngü" in r.json()["error"]["message"]


def test_create_rejects_missing_required_input(client):
    graph = {"nodes": [{"id": "s", "type": "core.save", "params": {"path": "x.txt"}}], "edges": []}
    r = client.post("/v1/workflows", json={"name": "eksik-giris", "graph": graph})
    assert r.status_code == 400
    assert "zorunlu giriş portu" in r.json()["error"]["message"]


def test_create_rejects_malformed_shape(client):
    r = client.post("/v1/workflows", json={"name": "sekil-bozuk", "graph": {"nodes": "olmaz"}})
    assert r.status_code == 400


def test_run_prompt_to_save_and_second_run_hits_cache(client, tmp_path):
    dest = str(tmp_path / "note.txt")
    graph = _prompt_save_graph(text="bir kedi", path=dest)
    wf = client.post("/v1/workflows", json={"name": "t2", "graph": graph}).json()

    run1 = client.post(f"/v1/workflows/{wf['id']}/run", json={}).json()
    assert run1["status"] == "running"
    done1 = _poll_run(client, run1["id"])
    assert done1["status"] == "done", done1
    assert done1["node_results"]["s"]["outputs"]["path"] == dest
    assert done1["node_results"]["p"]["from_cache"] is False

    run2 = client.post(f"/v1/workflows/{wf['id']}/run", json={}).json()
    done2 = _poll_run(client, run2["id"])
    assert done2["status"] == "done"
    assert done2["node_results"]["p"]["from_cache"] is True
    assert done2["node_results"]["s"]["from_cache"] is True


def test_run_marks_failed_node_and_run_error(client):
    graph = {
        "nodes": [
            {"id": "p", "type": "core.prompt", "params": {"text": "x"}},
            {"id": "s", "type": "core.save"},  # 'path' parametresi kasıtlı olarak eksik
        ],
        "edges": [
            {"id": "e1", "from_node": "p", "from_port": "text", "to_node": "s", "to_port": "value"},
        ],
    }
    wf = client.post("/v1/workflows", json={"name": "hatali", "graph": graph}).json()

    run = client.post(f"/v1/workflows/{wf['id']}/run", json={}).json()
    done = _poll_run(client, run["id"])

    assert done["status"] == "failed"
    assert done["node_results"]["s"]["status"] == "failed"
    assert "path" in done["error"]


def test_run_unknown_workflow_404s(client):
    assert client.post("/v1/workflows/yok/run", json={}).status_code == 404
