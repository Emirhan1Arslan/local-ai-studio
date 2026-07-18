"""Faz 4: batch üretim + seed varyasyonları + LoRA parametre/metadata testleri."""

from __future__ import annotations

from studio_inference.adapters.sdxl import _a1111_metadata
from test_scheduler import _poll_status


def test_batch_seed_variation(client):
    """Sabit seed s → işler s, s+1, … seed'leriyle üretilir (Faz 4 planı, adım 8)."""
    r = client.post(
        "/v1/jobs/batch",
        json={"kind": "dummy", "params": {"steps": 1, "step_ms": 10, "seed": 100}, "count": 4},
    )
    assert r.status_code == 200
    jobs = r.json()
    assert len(jobs) == 4
    assert [j["params"]["seed"] for j in jobs] == [100, 101, 102, 103]
    for j in jobs:
        _poll_status(client, j["id"], {"done"})


def test_batch_random_seed_stays_random(client):
    r = client.post(
        "/v1/jobs/batch",
        json={"kind": "dummy", "params": {"steps": 1, "step_ms": 10, "seed": -1}, "count": 3},
    )
    assert all(j["params"]["seed"] == -1 for j in r.json())


def test_batch_image_unique_outputs(client):
    """Batch'teki her görüntü işi kendi benzersiz çıktı yolunu alır."""
    r = client.post(
        "/v1/jobs/batch",
        json={"kind": "image", "params": {"model_path": "yok.safetensors"}, "count": 3},
    )
    paths = [j["params"]["output_path"] for j in r.json()]
    assert len(set(paths)) == 3


def test_batch_count_limit(client):
    r = client.post("/v1/jobs/batch", json={"kind": "dummy", "params": {}, "count": 99})
    assert r.status_code == 422  # pydantic sınırı (1-16)


def test_a1111_metadata_includes_lora():
    text = _a1111_metadata(
        {
            "prompt": "test",
            "model_path": "C:/m/sdxl.safetensors",
            "loras": [
                {"path": "C:/m/lora/detay.safetensors", "weight": 0.8},
                {"path": "C:/m/lora/stil.safetensors", "weight": 1.0},
            ],
        },
        seed=42,
    )
    assert "Lora: detay:0.8, stil:1.0" in text
    assert "Seed: 42" in text
