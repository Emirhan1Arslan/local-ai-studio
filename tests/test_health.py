"""Walking skeleton smoke testleri: health, system/info, hata modeli, token auth."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from studio_api.app import create_app
from studio_core.config import get_settings


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDIO_DATA_DIR", str(tmp_path / "LocalAIStudio"))
    # Ayar singleton'ını test başına sıfırla
    import studio_core.config as cfg
    import studio_core.db as db

    cfg._settings = None
    db._engine = None
    db._session_factory = None
    with TestClient(create_app()) as c:
        yield c


def test_health(client):
    r = client.get("/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_system_info(client):
    r = client.get("/v1/system/info")
    assert r.status_code == 200
    body = r.json()
    assert body["cpu_count"] >= 1
    assert body["gpu_backend"] in {"cuda", "rocm", "mps", "cpu"}
    assert body["ram_total_mb"] > 0


def test_error_model(client):
    r = client.get("/v1/yok-boyle-bir-yol")
    assert r.status_code == 404
    assert "error" in r.json()


def test_token_auth(client):
    get_settings().api_token = "gizli"
    try:
        assert client.get("/v1/system/info").status_code == 401
        ok = client.get("/v1/system/info", headers={"Authorization": "Bearer gizli"})
        assert ok.status_code == 200
    finally:
        get_settings().api_token = ""


def test_ws_events(client):
    from studio_api.bus import bus

    with client.websocket_connect("/v1/ws") as ws:
        first = ws.receive_json()  # bağlanınca system.status yayınlanır
        assert first["topic"] == "system.status"
        bus.emit("log", level="info", message="merhaba")
        msg = ws.receive_json()
        assert msg["topic"] == "log"
        assert msg["data"]["message"] == "merhaba"
