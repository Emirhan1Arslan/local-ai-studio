"""Faz 1 cila: /v1/settings (canlı-etkili override'lar) + tanı paketi."""

from __future__ import annotations

import zipfile

from studio_inference.scheduler import scheduler
from test_scheduler import _poll_status, _post_job


def test_settings_shape(client):
    r = client.get("/v1/settings")
    assert r.status_code == 200
    body = r.json()
    for key in (
        "hf_token_set",
        "telemetry_interval",
        "worker_idle_timeout",
        "worker_concurrency",
        "download_speed_limit_mbps",
        "paths",
    ):
        assert key in body
    for key in ("data_dir", "models_dir", "outputs_dir", "logs_dir", "db_path"):
        assert key in body["paths"]


def test_settings_put_updates_live_singleton(client):
    import studio_core.config as cfg

    r = client.put(
        "/v1/settings",
        json={
            "hf_token": "hf_test_token",
            "telemetry_interval": 5,
            "worker_idle_timeout": 120,
            "worker_concurrency": 2,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["hf_token_set"] is True
    assert body["telemetry_interval"] == 5
    assert body["worker_idle_timeout"] == 120
    assert body["worker_concurrency"] == 2

    # Yalnız yanıt değil, süreç genelindeki tekil ayar nesnesi de değişti mi?
    # (scheduler/telemetri restart'sız bu nesneden taze okur.)
    settings = cfg.get_settings()
    assert settings.hf_token == "hf_test_token"
    assert settings.telemetry_interval == 5
    assert settings.worker_idle_timeout == 120
    assert settings.worker_concurrency == 2
    assert scheduler._concurrency == 2

    r2 = client.get("/v1/settings")
    assert r2.json()["telemetry_interval"] == 5


def test_settings_hf_token_clear(client):
    client.put("/v1/settings", json={"hf_token": "gizli"})
    assert client.get("/v1/settings").json()["hf_token_set"] is True
    client.put("/v1/settings", json={"hf_token": ""})
    assert client.get("/v1/settings").json()["hf_token_set"] is False


def test_settings_persist_across_process_restart(client):
    """PUT'lanan değer DB'ye yazılır; yeni bir süreç açılışında (Settings() + load_overrides())
    ortam değişkeni olmadan da geri gelir — kalıcılığın gerçek kanıtı."""
    import studio_core.config as cfg
    from studio_core.settings_store import load_overrides

    client.put("/v1/settings", json={"worker_idle_timeout": 222})

    # "Yeni süreç" simülasyonu: env'den taze bir Settings() (override YOK), sonra DB'den yükle.
    cfg._settings = None
    fresh = cfg.get_settings()
    assert fresh.worker_idle_timeout != 222  # env'de yok, henüz DB'den yüklenmedi
    load_overrides()
    assert cfg.get_settings().worker_idle_timeout == 222


def test_settings_validates_bounds(client):
    r = client.put("/v1/settings", json={"telemetry_interval": 999})
    assert r.status_code == 422
    r2 = client.put("/v1/settings", json={"worker_concurrency": 0})
    assert r2.status_code == 422


def test_download_speed_limit_default_unlimited(client):
    assert client.get("/v1/settings").json()["download_speed_limit_mbps"] == 0


def test_download_speed_limit_updates_live(client):
    """Faz 8: kullanıcı İndirmeler/Ayarlar panelinden hız sınırını değiştirebilir —
    canlı etkili (studio_models.downloader._RateLimiter her chunk'ta taze okur)."""
    import studio_core.config as cfg

    r = client.put("/v1/settings", json={"download_speed_limit_mbps": 5})
    assert r.status_code == 200
    assert r.json()["download_speed_limit_mbps"] == 5
    assert cfg.get_settings().download_speed_limit_mbps == 5

    client.put("/v1/settings", json={"download_speed_limit_mbps": 0})
    assert cfg.get_settings().download_speed_limit_mbps == 0


def test_download_speed_limit_rejects_negative(client):
    r = client.put("/v1/settings", json={"download_speed_limit_mbps": -1})
    assert r.status_code == 422


def test_diagnostics_bundle_contains_expected_files(client, tmp_path):
    j = _post_job(client, {"fail": True})
    _poll_status(client, j, {"failed"})

    r = client.post("/v1/settings/diagnostics")
    assert r.status_code == 200
    path = r.json()["path"]
    assert path.endswith(".zip")

    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        assert "system_info.json" in names
        assert "settings.json" in names
        assert "son_basarisiz_isler.json" in names
        content = zf.read("son_basarisiz_isler.json").decode("utf-8")
        assert "kasıtlı test hatası" in content
