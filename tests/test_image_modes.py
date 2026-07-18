"""Faz 4: img2img/inpaint/outpaint yardımcıları, aile algısı, sampler metadata'sı.

Gerçek üretim GPU smoke testindedir; burada adapter'ın saf (torch'suz) yardımcıları
ve API kablolaması test edilir.
"""

from __future__ import annotations

import time

from fakes import write_fake_safetensors
from studio_inference.adapters.sdxl import (
    _a1111_metadata,
    _align,
    _detect_family,
    _expand_canvas,
)
from test_scheduler import _poll_status, _post_job


def test_align_multiples():
    assert _align(1024, 8) == 1024
    assert _align(1015, 8) == 1008
    assert _align(1024, 16) == 1024
    assert _align(1000, 16) == 992
    assert _align(10, 8) == 256  # alt sınır
    assert _align(9999, 8) == 4096  # üst sınır


def test_detect_family_from_safetensors_keys(tmp_path):
    flux = tmp_path / "flux.safetensors"
    write_fake_safetensors(flux, tensor_names=["double_blocks.0.img_attn.qkv.weight"])
    sd3 = tmp_path / "sd3.safetensors"
    write_fake_safetensors(sd3, tensor_names=["model.diffusion_model.joint_blocks.0.x_block.w"])
    sdxl = tmp_path / "sdxl.safetensors"
    write_fake_safetensors(sdxl, tensor_names=["model.diffusion_model.input_blocks.0.0.weight"])

    assert _detect_family(str(flux)) == "flux"
    assert _detect_family(str(sd3)) == "sd3"
    assert _detect_family(str(sdxl)) == "sdxl"
    assert _detect_family(str(tmp_path / "yok.safetensors")) == "sdxl"  # okunamayan → varsayılan


def test_detect_family_from_diffusers_dir(tmp_path):
    d = tmp_path / "repo"
    d.mkdir()
    (d / "model_index.json").write_text('{"_class_name": "FluxPipeline"}', encoding="utf-8")
    assert _detect_family(str(d)) == "flux"
    (d / "model_index.json").write_text(
        '{"_class_name": "StableDiffusion3Pipeline"}', encoding="utf-8"
    )
    assert _detect_family(str(d)) == "sd3"


def test_expand_canvas_geometry_and_mask():
    from PIL import Image

    img = Image.new("RGB", (512, 256), (200, 30, 30))
    canvas, mask = _expand_canvas(img, left=64, top=0, right=64, bottom=128, feather=16)
    assert canvas.size == (640, 384)
    assert mask.size == (640, 384)
    # Yeni alan (sağ-alt köşe) beyaza yakın: yeniden çizilecek
    assert mask.getpixel((639, 383)) > 200
    # Orijinalin merkezi siyaha yakın: korunacak
    assert mask.getpixel((64 + 256, 128)) < 30
    # Orijinal içerik yerinde duruyor
    assert canvas.getpixel((64 + 256, 128)) == (200, 30, 30)


def test_a1111_metadata_sampler_strength_and_size_override():
    text = _a1111_metadata(
        {
            "prompt": "test",
            "model_path": "C:/m/sdxl.safetensors",
            "sampler": "dpmpp_2m_karras",
            "mode": "img2img",
            "strength": 0.6,
            "width": 1024,
            "height": 1024,
        },
        seed=7,
        size=(768, 512),
    )
    assert "Sampler: DPM++ 2M Karras" in text
    assert "Denoising strength: 0.6" in text
    assert "Size: 768x512" in text  # gerçek çıktı boyutu params'ı ezer


def test_a1111_metadata_default_has_no_sampler_line():
    text = _a1111_metadata({"prompt": "x", "model_path": "m.safetensors"}, seed=1)
    assert "Sampler:" not in text
    assert "Denoising strength" not in text


def test_asset_parent_link(client, tmp_path):
    """img2img türevi, kaynağına parent_id ile bağlanır (Asset Graph, önemli.md)."""
    out1 = tmp_path / "kaynak.png"
    out1.write_bytes(b"x")
    j1 = _post_job(client, {"steps": 1, "step_ms": 10, "result_path": str(out1)})
    _poll_status(client, j1, {"done"})

    deadline = time.monotonic() + 5
    assets = []
    while time.monotonic() < deadline:
        assets = client.get("/v1/assets").json()
        if assets:
            break
        time.sleep(0.1)
    assert len(assets) == 1
    parent_id = assets[0]["id"]

    out2 = tmp_path / "turev.png"
    out2.write_bytes(b"y")
    j2 = _post_job(
        client,
        {"steps": 1, "step_ms": 10, "result_path": str(out2), "parent_asset_id": parent_id},
    )
    _poll_status(client, j2, {"done"})

    deadline = time.monotonic() + 5
    child = None
    while time.monotonic() < deadline:
        found = [a for a in client.get("/v1/assets").json() if a["path"] == str(out2)]
        if found:
            child = found[0]
            break
        time.sleep(0.1)
    assert child is not None
    assert child["parent_id"] == parent_id


def test_asset_parent_link_ignores_unknown_parent(client, tmp_path):
    out = tmp_path / "tek.png"
    out.write_bytes(b"z")
    j = _post_job(
        client, {"steps": 1, "step_ms": 10, "result_path": str(out), "parent_asset_id": "yok"}
    )
    _poll_status(client, j, {"done"})
    deadline = time.monotonic() + 5
    assets = []
    while time.monotonic() < deadline:
        assets = client.get("/v1/assets").json()
        if assets:
            break
        time.sleep(0.1)
    assert assets and assets[0]["parent_id"] is None
