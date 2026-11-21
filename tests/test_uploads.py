"""Faz 4: /v1/uploads — img2img/inpaint kaynak görsel ve maske yükleme ucu."""

from __future__ import annotations

import base64
from pathlib import Path


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def test_upload_roundtrip(client):
    r = client.post(
        "/v1/uploads", json={"filename": "kaynak.png", "data_base64": _b64(b"sahte-png-verisi")}
    )
    assert r.status_code == 200
    body = r.json()
    p = Path(body["path"])
    assert p.exists()
    assert p.read_bytes() == b"sahte-png-verisi"
    assert body["size_bytes"] == len(b"sahte-png-verisi")
    assert p.parent.name == "uploads"
    assert p.name.endswith("_kaynak.png")


def test_upload_path_traversal_blocked(client):
    r = client.post(
        "/v1/uploads",
        json={"filename": "..\\..\\evil.png", "data_base64": _b64(b"x")},
    )
    assert r.status_code == 200
    p = Path(r.json()["path"])
    # Yalnız dosya adı kullanılır; uploads dizininden kaçış yok
    assert p.parent.name == "uploads"
    assert ".." not in p.name


def test_upload_rejects_bad_extension(client):
    r = client.post("/v1/uploads", json={"filename": "zarar.exe", "data_base64": _b64(b"x")})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "bad_request"


def test_upload_rejects_invalid_base64(client):
    r = client.post("/v1/uploads", json={"filename": "a.png", "data_base64": "%%%bozuk%%%"})
    assert r.status_code == 400


def test_uploads_served_statically(client):
    """Transkript kartı kaynak kaydı yeniden çalabilsin diye uploads/ statik sunulur."""
    r = client.post(
        "/v1/uploads", json={"filename": "kayit.webm", "data_base64": _b64(b"sahte-ses-verisi")}
    )
    assert r.status_code == 200
    name = Path(r.json()["path"]).name

    got = client.get(f"/uploads/{name}")
    assert got.status_code == 200
    assert got.content == b"sahte-ses-verisi"


def test_upload_accepts_audio_extensions(client):
    for ext in ("wav", "mp3", "webm", "m4a", "flac", "ogg"):
        r = client.post(
            "/v1/uploads", json={"filename": f"kayit.{ext}", "data_base64": _b64(b"sahte-ses")}
        )
        assert r.status_code == 200, f"{ext}: {r.text}"
        assert r.json()["path"].endswith(f".{ext}")
