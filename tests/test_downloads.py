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


SNAPSHOT_FILES = ["config.json", "model.safetensors", "preprocessor_config.json"]


SEARCH_RESULTS = [
    {
        "id": "ZhengPeng7/BiRefNet",
        "pipeline_tag": "image-segmentation",
        "library_name": "birefnet",
        "downloads": 699873,
        "likes": 606,
        "tags": ["safetensors", "license:mit"],
        "safetensors": {"parameters": {"F16": 1000, "I64": 10}, "total": 1010},
    },
    {
        "id": "gated/model",
        "pipeline_tag": "text-to-image",
        "downloads": 10,
        "likes": 1,
        "tags": [],
        "gated": True,
    },
]


async def fake_server(scope, receive, send):
    """/file: Range destekli · /slow: parça parça yavaş akış · /norange: Range'i yok sayar ·
    /api/models: HF arama (Keşfet paneli) · /api/models/<repo>: HF repo dosya listesi."""
    assert scope["type"] == "http"
    path = scope["path"]
    headers = {k.decode().lower(): v.decode() for k, v in scope["headers"]}

    if path == "/api/models":
        import json

        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": json.dumps(SEARCH_RESULTS).encode()})
        return

    if path.startswith("/api/models/"):
        import json

        repo = path.removeprefix("/api/models/")
        if repo == "yok/repo":
            body = b"{}"
            status = 404
        else:
            siblings = [
                {"rfilename": f, "size": 1024 * (i + 1)} for i, f in enumerate(SNAPSHOT_FILES)
            ]
            if repo == "nested/repo":
                siblings = [{"rfilename": "onnx/model.onnx", "size": 2048}]
            elif repo == "evil/repo":
                siblings = [{"rfilename": "../../kacak.bin", "size": 2048}]
            elif repo == "empty/repo":
                siblings = []
            body = json.dumps({"siblings": siblings}).encode()
            status = 200
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": body})
        return

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
        # Yük altında bile pause isteğine yetecek genişlikte pencere (~3 sn)
        for i in range(0, len(body), 4096):
            await send(
                {"type": "http.response.body", "body": body[i : i + 4096], "more_body": True}
            )
            await asyncio.sleep(0.1)
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


def test_hf_search_returns_curated_fields(dl_client):
    r = dl_client.get("/v1/downloads/search", params={"q": "birefnet"})
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 2
    first = rows[0]
    assert first["id"] == "ZhengPeng7/BiRefNet"
    assert first["pipeline_tag"] == "image-segmentation"
    assert first["downloads"] == 699873
    assert first["license"] == "mit"  # tags içindeki "license:mit"tan çıkarıldı
    assert first["size_bytes"] == 1000 * 2 + 10 * 8  # F16:1000*2 + I64:10*8
    assert rows[1]["gated"] is True
    assert rows[1]["size_bytes"] == 0  # safetensors metadata'sı yok


def test_hf_search_limit_bounds(dl_client):
    r = dl_client.get("/v1/downloads/search", params={"q": "x", "limit": 999})
    assert r.status_code == 400


def test_hf_list_files_endpoint(dl_client):
    r = dl_client.get("/v1/downloads/files", params={"repo_id": "org/some-repo"})
    assert r.status_code == 200
    rows = r.json()
    assert [row["filename"] for row in rows] == SNAPSHOT_FILES
    assert [row["size"] for row in rows] == [1024, 2048, 3072]


def test_hf_list_files_repo_not_found(dl_client):
    r = dl_client.get("/v1/downloads/files", params={"repo_id": "yok/repo"})
    assert r.status_code == 400


def test_snapshot_download_only_patterns_filters_files(dl_client):
    """Faz 8: Mochi gibi aynı repo'da hem fp32 hem bf16 varyantını tutan modellerde
    yalnızca istenen varyantı seçmek için only_patterns kullanılabilir."""
    r = dl_client.post(
        "/v1/downloads/snapshot",
        json={
            "repo_id": "org/some-repo",
            "subdir": "captioning",
            "only_patterns": ["config.json"],
        },
    )
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 1
    assert Path(rows[0]["dest_path"]).name == "config.json"


def test_snapshot_download_only_patterns_no_match_rejected(dl_client):
    r = dl_client.post(
        "/v1/downloads/snapshot",
        json={
            "repo_id": "org/some-repo",
            "subdir": "captioning",
            "only_patterns": ["*.nonexistent"],
        },
    )
    assert r.status_code == 400


def test_snapshot_download_queues_all_files(dl_client):
    """Faz 5: çok dosyalı HF repo'su tek istekte kuyruğa alınır, hepsi aynı repo
    klasörüne iner (BiRefNet/Florence-2/SAM gibi tek-dosya olmayan modeller)."""
    r = dl_client.post(
        "/v1/downloads/snapshot", json={"repo_id": "org/some-repo", "subdir": "captioning"}
    )
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == len(SNAPSHOT_FILES)
    dest_names = {Path(row["dest_path"]).name for row in rows}
    assert dest_names == set(SNAPSHOT_FILES)
    for row in rows:
        p = Path(row["dest_path"])
        assert p.parent.name == "some-repo"
        assert p.parent.parent.name == "captioning"

    for row in rows:
        d = _poll(dl_client, row["id"], {"done", "failed"})
        assert d["status"] == "done", d["error"]
        assert Path(d["dest_path"]).read_bytes() == DATA


def test_snapshot_download_repo_not_found(dl_client):
    r = dl_client.post(
        "/v1/downloads/snapshot", json={"repo_id": "yok/repo", "subdir": "captioning"}
    )
    assert r.status_code == 400


def test_snapshot_download_empty_repo_rejected(dl_client):
    r = dl_client.post(
        "/v1/downloads/snapshot", json={"repo_id": "empty/repo", "subdir": "captioning"}
    )
    assert r.status_code == 400


def test_snapshot_download_preserves_nested_subpath(dl_client):
    """Faz 7: Kokoro-82M gibi repo'ların voices/ alt klasörü artık desteklenir —
    hedef yol repo-içi göreli yapıyı korur (models/<subdir>/<repo>/onnx/model.onnx)."""
    r = dl_client.post(
        "/v1/downloads/snapshot", json={"repo_id": "nested/repo", "subdir": "captioning"}
    )
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 1
    p = Path(rows[0]["dest_path"])
    assert p.name == "model.onnx"
    assert p.parent.name == "onnx"
    assert p.parent.parent.name == "repo"
    d = _poll(dl_client, rows[0]["id"], {"done", "failed"})
    assert d["status"] == "done", d["error"]


def test_snapshot_download_nested_traversal_rejected(dl_client):
    """İç-içe klasör desteği path-traversal'a açık kapı bırakmamalı — repo
    (gerçekte hiç dönmeyecek) bir '../' içeren rfilename dönse bile reddedilir."""
    r = dl_client.post(
        "/v1/downloads/snapshot", json={"repo_id": "evil/repo", "subdir": "captioning"}
    )
    assert r.status_code == 400


def test_snapshot_download_invalid_subdir_rejected(dl_client):
    r = dl_client.post(
        "/v1/downloads/snapshot", json={"repo_id": "org/some-repo", "subdir": "hack"}
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


def test_rate_limiter_noop_when_unlimited(client, monkeypatch):
    """Faz 8: hız sınırı 0 (sınırsız) iken throttle() hiç uyumamalı."""
    from studio_core.config import get_settings as _gs

    _gs().download_speed_limit_mbps = 0
    limiter = downloader._RateLimiter()
    slept = []
    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda s: slept.append(s) or real_sleep(0))

    asyncio.run(limiter.throttle(10 * 1024 * 1024))
    assert slept == []


def test_rate_limiter_throttles_to_configured_rate(client, monkeypatch):
    """1 MB/sn sınırlanmış limitte art arda 3 adet 1 MB'lık chunk — ilki hiç
    beklemez (kova dolu başlar), sonraki her chunk ~1 sn'lik bekleme üretir
    (sanal zaman damgalı ortak kova — bkz. downloader.py)."""
    from studio_core.config import get_settings as _gs

    _gs().download_speed_limit_mbps = 1  # 1 MB/sn
    limiter = downloader._RateLimiter()
    slept = []
    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda s: slept.append(s) or real_sleep(0))

    async def _run():
        for _ in range(3):
            await limiter.throttle(1024 * 1024)

    asyncio.run(_run())
    _gs().download_speed_limit_mbps = 0  # diğer testleri etkilememesi için sıfırla
    # mock'lanmış sleep gerçek zaman geçirmediği için 3 chunk da art arda "anında"
    # gelmiş gibi ölçülür: 1. chunk beklemez, 2. ~1sn, 3. ~2sn (kümülatif zamanlama) bekler.
    assert len(slept) == 2
    assert sum(slept) == pytest.approx(3.0, abs=0.1)
