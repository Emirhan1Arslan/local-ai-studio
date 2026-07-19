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


def test_multi_component_pipeline_collapses_to_one_model(client):
    """Faz 8 bulgusu: LTX-Video gibi çok bileşenli bir diffusers pipeline'ı
    (model_index.json + alt klasörler, bazı bileşenler parçalı) tarayıcıda TEK
    bir model olarak görünmeli — her bileşen/parça ayrı satır olmamalı."""
    root = _models_dir(client)
    pipeline = root / "video" / "MyVideoModel"
    (pipeline / "transformer").mkdir(parents=True)
    (pipeline / "vae").mkdir(parents=True)
    (pipeline / "model_index.json").write_text("{}", encoding="utf-8")
    files = [
        pipeline / "transformer" / "diffusion_pytorch_model-00001-of-00002.safetensors",
        pipeline / "transformer" / "diffusion_pytorch_model-00002-of-00002.safetensors",
        pipeline / "vae" / "diffusion_pytorch_model.safetensors",
    ]
    for f in files:
        write_fake_safetensors(f)

    summary = client.post("/v1/models/rescan").json()
    assert summary["added"] == 1

    models = client.get("/v1/models", params={"kind": "video"}).json()
    assert len(models) == 1
    assert models[0]["name"] == "MyVideoModel"
    assert models[0]["size_bytes"] == sum(f.stat().st_size for f in files)
    assert models[0]["meta"]["components"] == [f.name for f in files]  # sıralı: transformer önce, vae sonra


def test_same_folder_multi_weight_collapses_to_one_model(client):
    """XTTS-v2 deseni: aynı klasörde config.json + birden çok ağırlık dosyası
    (ör. model.pth + speakers_xtts.pth) — tek model, en büyük dosya temsilci."""
    root = _models_dir(client)
    sub = root / "tts" / "MyTTS"
    sub.mkdir(parents=True)
    (sub / "config.json").write_text("{}", encoding="utf-8")
    write_fake_safetensors(sub / "model.safetensors", tensor_names=["a", "b"])
    write_fake_safetensors(sub / "sidecar.safetensors")

    client.post("/v1/models/rescan")
    models = client.get("/v1/models", params={"kind": "tts"}).json()
    assert len(models) == 1
    assert models[0]["name"] == "MyTTS"


def test_subfolder_without_own_config_still_joins_parent_model(client):
    """Kokoro'nun voices/ deseni: config.json'suz bir alt klasördeki dosyalar
    (ör. ses paketleri) da üst modelin (config.json'lı en yakın ata) parçası
    sayılmalı — kullanıcı geri bildirimi: 'kokoro ses dosyaları da kokora
    modeline ait ya'. Tek istisna: üst klasörlerden biri model_index.json
    içeriyorsa (çok bileşenli pipeline) o kök öncelikli kalır."""
    root = _models_dir(client)
    core = root / "tts" / "MyKokoro"
    core.mkdir(parents=True)
    (core / "config.json").write_text("{}", encoding="utf-8")
    write_fake_safetensors(core / "core.safetensors")
    voices = core / "voices"
    voices.mkdir()
    write_fake_safetensors(voices / "voice_a.safetensors")
    write_fake_safetensors(voices / "voice_b.safetensors")

    client.post("/v1/models/rescan")
    models = client.get("/v1/models", params={"kind": "tts"}).json()
    assert len(models) == 1
    assert models[0]["name"] == "MyKokoro"
    assert set(models[0]["meta"]["components"]) == {
        "core.safetensors",
        "voice_a.safetensors",
        "voice_b.safetensors",
    }


def test_unrelated_top_level_files_stay_separate(client):
    """config.json/model_index.json ile hiç ilişkisi olmayan, kendi başına
    duran dosyalar (ör. lora/'daki tek bir LoRA) ayrı ayrı kalmaya devam eder."""
    root = _models_dir(client)
    write_fake_safetensors(root / "lora" / "lora-a.safetensors")
    write_fake_safetensors(root / "lora" / "lora-b.safetensors")

    client.post("/v1/models/rescan")
    names = {m["name"] for m in client.get("/v1/models", params={"kind": "lora"}).json()}
    assert names == {"lora-a", "lora-b"}
