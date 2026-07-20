"""Faz 10 gerçek node kütüphanesi testleri: Prompt/Görsel Üret/Upscale/
BG Remove/Kaydet düğümlerinin param inşa + çıktı eşleme mantığı (GPU'suz, saf
fonksiyon testleri) ve DoD grafiğinin ("Prompt → FLUX → Upscale → BG Remove →
Save") gerçek NODE_TYPES kayıt defteriyle yapısal olarak geçerli olduğu."""

from __future__ import annotations

import pytest
from studio_workflows import NODE_TYPES, Edge, Graph, NodeInstance
from studio_workflows.node_library import (
    _build_bg_remove_params,
    _build_image_gen_params,
    _build_upscale_params,
    _map_image_output,
    _map_imgtool_image_output,
    _prompt_exec,
    _save_exec,
)


def test_prompt_exec_returns_text():
    assert _prompt_exec({}, {"text": "bir kedi"}) == {"text": "bir kedi"}
    assert _prompt_exec({}, {}) == {"text": ""}


def test_image_generate_requires_model_path():
    with pytest.raises(RuntimeError, match="model_path"):
        _build_image_gen_params({"prompt": "x"}, {})


def test_image_generate_builds_job_params():
    params = _build_image_gen_params(
        {"prompt": "bir kedi"}, {"model_path": "C:/models/flux", "steps": 4, "seed": 7}
    )
    assert params["prompt"] == "bir kedi"
    assert params["model_path"] == "C:/models/flux"
    assert params["steps"] == 4
    assert params["seed"] == 7
    assert params["mode"] == "txt2img"
    assert params["model_family"] == "flux"


def test_image_generate_maps_output():
    mapped = _map_image_output({"path": "out.png", "width": 512, "height": 512, "seed": 3})
    assert mapped == {"image": {"path": "out.png", "width": 512, "height": 512, "seed": 3}}


def test_upscale_requires_model_path():
    with pytest.raises(RuntimeError, match="model_path"):
        _build_upscale_params({"image": {"path": "in.png"}}, {})


def test_upscale_builds_job_params():
    params = _build_upscale_params({"image": {"path": "in.png"}}, {"model_path": "esrgan.pth"})
    assert params == {"op": "upscale", "image_path": "in.png", "model_path": "esrgan.pth"}


def test_bg_remove_builds_job_params_with_default_model():
    params = _build_bg_remove_params({"image": {"path": "in.png"}}, {})
    assert params == {"op": "bg_remove", "image_path": "in.png", "model_path": ""}


def test_imgtool_output_mapping():
    mapped = _map_imgtool_image_output({"path": "out.png", "width": 10, "height": 20})
    assert mapped == {"image": {"path": "out.png", "width": 10, "height": 20}}


def test_save_requires_path_param():
    with pytest.raises(RuntimeError, match="path"):
        _save_exec({"value": {"path": "in.png"}}, {})


def test_save_copies_file_value(tmp_path):
    src = tmp_path / "src.png"
    src.write_bytes(b"fake-png-bytes")
    dest = tmp_path / "nested" / "dest.png"

    out = _save_exec({"value": {"path": str(src)}}, {"path": str(dest)})

    assert out == {"path": str(dest)}
    assert dest.read_bytes() == b"fake-png-bytes"


def test_save_writes_text_value(tmp_path):
    dest = tmp_path / "note.txt"

    out = _save_exec({"value": "merhaba"}, {"path": str(dest)})

    assert out == {"path": str(dest)}
    assert dest.read_text(encoding="utf-8") == "merhaba"


def test_dod_graph_is_structurally_valid():
    """Faz 10 DoD: 'Prompt → FLUX → Upscale → BG Remove → Save' grafiği kurulur."""
    g = Graph(
        nodes=[
            NodeInstance(
                id="prompt", type="core.prompt", params={"text": "bir kedi, fotogerçekçi"}
            ),
            NodeInstance(
                id="gen", type="image.generate", params={"model_path": "C:/models/flux-schnell"}
            ),
            NodeInstance(
                id="up", type="image.upscale", params={"model_path": "C:/models/esrgan.pth"}
            ),
            NodeInstance(id="bg", type="image.bg_remove"),
            NodeInstance(id="save", type="core.save", params={"path": "C:/out/final.png"}),
        ],
        edges=[
            Edge(id="e1", from_node="prompt", from_port="text", to_node="gen", to_port="prompt"),
            Edge(id="e2", from_node="gen", from_port="image", to_node="up", to_port="image"),
            Edge(id="e3", from_node="up", from_port="image", to_node="bg", to_port="image"),
            Edge(id="e4", from_node="bg", from_port="image", to_node="save", to_port="value"),
        ],
    )
    g.validate_structure(NODE_TYPES)
    assert g.topological_order() == ["prompt", "gen", "up", "bg", "save"]
