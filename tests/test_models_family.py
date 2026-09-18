"""Çok bileşenli model klasörlerinin aile tespiti (Z-Image / HiDream-O1 / diffusers)
— Model Manager kaydı `family` alanını doldurur, UI/MCP işi image|imagex'e buna göre yollar."""

from __future__ import annotations

import json

from fakes import write_fake_safetensors
from studio_inference.adapters.imagex import defaults_for, detect_family, resolve_root
from studio_models.metadata import extract_group, group_family, group_key


def _zimage(root):
    (root / "transformer").mkdir(parents=True)
    (root / "model_index.json").write_text(json.dumps({"_class_name": "ZImagePipeline"}))
    write_fake_safetensors(
        root / "transformer" / "diffusion_pytorch_model-00001-of-00002.safetensors"
    )
    write_fake_safetensors(
        root / "transformer" / "diffusion_pytorch_model-00002-of-00002.safetensors"
    )


def _hidream(root, name_hint=True):
    root.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps({"model_type": "qwen3_vl", "architectures": ["Qwen3VLForConditionalGeneration"]})
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"final_layer2.weight": "model-00001-of-00002.safetensors"}})
    )
    write_fake_safetensors(root / "model-00001-of-00002.safetensors")
    write_fake_safetensors(root / "model-00002-of-00002.safetensors")


def test_group_family_zimage(tmp_path):
    root = tmp_path / "checkpoints" / "Z-Image-Turbo"
    _zimage(root)
    assert group_family(root) == ("zimage", "ZImagePipeline")
    assert detect_family(str(root)) == "zimage"
    assert defaults_for("zimage")["steps"] == 9


def test_group_family_hidream_by_head_marker(tmp_path):
    root = (
        tmp_path / "checkpoints" / "SomeModel"
    )  # adı ipucu vermiyor → ağırlık başlığı karar verir
    _hidream(root)
    assert group_family(root)[0] == "hidream_o1"
    assert detect_family(str(root)) == "hidream_o1"
    assert defaults_for("hidream_o1", str(tmp_path / "HiDream-O1-Image-Dev"))["steps"] == 28
    assert defaults_for("hidream_o1", str(tmp_path / "HiDream-O1-Image"))["steps"] == 50


def test_plain_qwen3_vl_is_not_hidream(tmp_path):
    root = tmp_path / "llm" / "Qwen3-VL-8B"
    root.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}))
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"lm_head.weight": "model.safetensors"}})
    )
    assert group_family(root) == ("qwen3_vl", "")
    assert detect_family(str(root)) == ""


def test_extract_group_fills_family_and_pipeline_class(tmp_path):
    models_root = tmp_path
    root = tmp_path / "checkpoints" / "Z-Image-Turbo"
    _zimage(root)
    files = sorted((root / "transformer").glob("*.safetensors"))
    key = group_key(files[0], models_root)
    assert key == str(root)
    rec = extract_group(key, files, models_root)
    assert rec["name"] == "Z-Image-Turbo"
    assert rec["family"] == "zimage"
    assert rec["kind"] == "checkpoint"
    assert rec["meta"]["pipeline_class"] == "ZImagePipeline"


def test_resolve_root_from_component_file(tmp_path):
    root = tmp_path / "checkpoints" / "Z-Image-Turbo"
    _zimage(root)
    shard = root / "transformer" / "diffusion_pytorch_model-00001-of-00002.safetensors"
    assert resolve_root(str(shard)) == str(root)
    assert resolve_root(str(root)) == str(root)
