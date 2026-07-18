"""Faz 2 indirme motoru testleri: tam indirme, Range resume, sha mismatch, pause, traversal.

Ağ yerine Range destekli sahte bir ASGI sunucusu kullanılır (httpx.ASGITransport) —
testler tamamen çevrimdışıdır.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from pathlib import Path

import httpx
import pytest
from studio_core.config import get_settings
from studio_models import downloader

DATA = bytes(range(256)) * 512  # 128 KB deterministik veri
SHA = hashlib.sha256(DATA).hexdigest()


async def fake_server(scope, receive, send):
    """/file: Range destekli · /slow: parça parça yavaş akış · /norange: Range'i yok sayar."""
    assert scope["type"] == "http"
    path = scope["path"]
    headers = {k.decode().lower(): v.decode() for k, v in scope["headers"]}
    body = DATA
    status = 200
    resp_headers = [(b"content-type", b"application/octet-stream")]

    rng = headers.get("range", "")
    if path == "/file" and rng.startswith("bytes="):
        start = int(rng.split("=")[1].rstrip("-"))
        body = DATA[start:]
        status = 206
        resp_headers.append(
            (b"content-range", f"bytes {start}-{len(DATA) - 1}/{len(DATA)}".encode())
        )

    resp_headers.append((b"content-length", str(len(body)).encode()))
    await send({"type": "http.response.start", "status": status, "headers": resp_headers})

    if path == "/slow":
        for i in range(0, len(body), 8192):
            await send(
                {"type": "http.response.body", "body": body[i : i + 8192], "more_body": True}
            )
            await asyncio.sleep(0.05)
        await send({"type": "http.response.body", "body": b""})
    else:
        await send({"type": "http.response.body", "body": body})


@pytest.fixture()
def dl_client(client):
    downloader.manager._transport = httpx.ASGITransport(app=fake_server)
    yield client
    downloader.manager._transport = None


def _poll(client, download_id: str, want: set[str], timeout: float = 15.0) -> dict:
    deadline = time.monotonic() + timeout
    last = {}
    while time.monotonic() < deadline:
        for d in client.get("/v1/downloads").json():
            if d["id"] == download_id:
                last = d
                if d["status"] in want:
                    return d
        time.sleep(0.1)
    raise AssertionError(f"{want} beklenirken durum: {last.get('status')!r} ({last.get('error')})")


def test_full_download_lands_in_models(dl_client):
    r = dl_client.post(
        "/v1/downloads",
        json={"url": "http://fake/file", "subdir": "llm", "filename": "indirme.bin", "sha256": SHA},
    )
    assert r.status_code == 200
    d = _poll(dl_client, r.json()["id"], {"done", "failed"})
    assert d["status"] == "done"
    assert d["sha256_actual"] == SHA
    dest = Path(d["dest_path"])
    assert dest.read_bytes() == DATA
    assert not dest.with_suffix(dest.suffix + ".part").exists()


def test_resume_continues_from_part(dl_client):
    """Yarıda kesilen indirme kaldığı yerden sürer (Faz 2 DoD)."""
    dest = get_settings().models_dir / "llm" / "resume.bin"
    dest.parent.mkdir(parents=True, exist_ok=True)
    Path(str(dest) + ".part").write_bytes(DATA[:40_000])  # önceki oturumdan kalan parça

    r = dl_client.post(
        "/v1/downloads",
        json={"url": "http://fake/file", "subdir": "llm", "filename": "resume.bin", "sha256": SHA},
    )
    d = _poll(dl_client, r.json()["id"], {"done", "failed"})
    assert d["status"] == "done", d["error"]
    assert d["sha256_actual"] == SHA  # hash, parça + devamın birleşiminden doğru çıktı
    assert dest.read_bytes() == DATA


def test_sha_mismatch_fails_and_cleans(dl_client):
    r = dl_client.post(
        "/v1/downloads",
        json={
            "url": "http://fake/file",
            "subdir": "llm",
            "filename": "bozuk.bin",
            "sha256": "0" * 64,
        },
    )
    d = _poll(dl_client, r.json()["id"], {"done", "failed"})
    assert d["status"] == "failed"
    assert "sha256" in d["error"]
    assert not Path(d["dest_path"]).exists()
    assert not Path(d["dest_path"] + ".part").exists()


def test_pause_then_resume(dl_client):
    r = dl_client.post(
        "/v1/downloads",
        json={"url": "http://fake/slow", "subdir": "llm", "filename": "yavas.bin"},
    )
    download_id = r.json()["id"]
    _poll(dl_client, download_id, {"downloading"})
    assert dl_client.post(f"/v1/downloads/{download_id}/pause").json()["ok"]
    d = _poll(dl_client, download_id, {"paused"})
    assert d["status"] == "paused"

    assert dl_client.post(f"/v1/downloads/{download_id}/resume").json()["ok"]
    d = _poll(dl_client, download_id, {"done", "failed"})
    assert d["status"] == "done"
    assert Path(d["dest_path"]).read_bytes() == DATA


def test_invalid_subdir_rejected(dl_client):
    r = dl_client.post(
        "/v1/downloads", json={"url": "http://fake/file", "subdir": "hack", "filename": "x.bin"}
    )
    assert r.status_code == 400


def test_traversal_filename_sanitized(dl_client):
    r = dl_client.post(
        "/v1/downloads",
        json={"url": "http://fake/file", "subdir": "llm", "filename": "../../kacak.bin"},
    )
    assert r.status_code == 200
    dest = Path(r.json()["dest_path"])
    models_dir = get_settings().models_dir.resolve()
    assert models_dir in dest.resolve().parents
    assert dest.name == "kacak.bin"
