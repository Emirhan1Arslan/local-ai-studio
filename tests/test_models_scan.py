"""Faz 2 testleri: metadata çıkarımı, tarayıcı senkronu, /v1/models API, canlı algılama."""

from __future__ import annotations

import time
from pathlib import Path

from fakes import write_fake_gguf, write_fake_safetensors


def _models_dir(client) -> Path:
    from studio_core.config import get_settings

    return get_settings().models_dir


def test_scan_and_list(client):
    root = _models_dir(client)
    write_fake_safetensors(root / "lora" / "test-lora.safetensors")
    write_fake_gguf(root / "llm" / "qwen-test.gguf", name="Qwen Test", arch="qwen2")

    assert client.post("/v1/models/rescan").json()["added"] == 2

    models = {m["name"]: m for m in client.get("/v1/models").json()}
    lora = models["test-lora"]
    assert lora["kind"] == "lora"
    assert lora["format"] == "safetensors"
    assert lora["meta"]["tensor_count"] == 1
    assert lora["sha256"]  # küçük dosya → hash hesaplanır

    gguf = models["qwen-test"]
    assert gguf["kind"] == "llm"
    assert gguf["family"] == "qwen2"
    assert gguf["meta"]["metadata"]["general.name"] == "Qwen Test"


def test_scan_removes_deleted(client):
    root = _models_dir(client)
    f = root / "llm" / "gecici.gguf"
    write_fake_gguf(f)
    client.post("/v1/models/rescan")
    f.unlink()
    summary = client.post("/v1/models/rescan").json()
    assert summary["removed"] == 1
    assert client.get("/v1/models").json() == []


def test_kind_filter(client):
    root = _models_dir(client)
    write_fake_gguf(root / "llm" / "a.gguf")
    write_fake_safetensors(root / "vae" / "b.safetensors")
    client.post("/v1/models/rescan")
    only_llm = client.get("/v1/models", params={"kind": "llm"}).json()
    assert [m["name"] for m in only_llm] == ["a"]


def test_corrupt_file_still_listed(client):
    root = _models_dir(client)
    bad = root / "llm" / "bozuk.gguf"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"BUBIRGGUFDEGIL")
    client.post("/v1/models/rescan")
    (m,) = client.get("/v1/models").json()
    assert "error" in m["meta"]


def test_watcher_detects_new_model(client):
    """Klasöre elle atılan model restart'sız algılanır (Faz 2 DoD: < 5 sn)."""
    root = _models_dir(client)
    write_fake_gguf(root / "llm" / "canli.gguf", name="Canli")
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if any(m["name"] == "canli" for m in client.get("/v1/models").json()):
            return
        time.sleep(0.25)
    raise AssertionError("watcher yeni modeli 10 sn içinde algılamadı")
