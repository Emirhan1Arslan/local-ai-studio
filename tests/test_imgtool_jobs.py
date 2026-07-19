"""Faz 5 iskelet testleri: imgtool işi kablolaması (gerçek araçlar GPU smoke testinde).

Test ortamında image worker env'i yoktur → iş, kurulum talimatı içeren anlaşılır
bir hatayla FAILED olmalı; sunucu süreci torch import etmemeli (ADR-0013).
"""

from __future__ import annotations

import os
import sys

from studio_inference.scheduler import worker_python
from test_scheduler import _poll_status


def test_imgtool_kind_available(client):
    r = client.post("/v1/jobs", json={"kind": "imgtool", "params": {"op": "bg_remove"}})
    assert r.status_code == 200  # tür kabul edilir; hata worker'da yakalanır


def test_server_process_has_no_torch(client):
    assert "torch" not in sys.modules


def test_imgtool_shares_image_env(client, tmp_path):
    """imgtool kendi venv'ini doğurmaz — 'image' venv'ini paylaşır (_ENV_ALIAS)."""
    # Test veri dizininde de gerçekten aynı env köküne düştüklerini kanıtla.
    from studio_core.config import get_settings

    settings = get_settings()
    sub = "Scripts" if os.name == "nt" else "bin"
    exe = "python.exe" if os.name == "nt" else "python"
    fake = settings.envs_dir / "image" / sub / exe
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text("")  # varlığı yeterli, worker_python içerik okumuyor
    try:
        assert worker_python("image") == str(fake)
        assert worker_python("imgtool") == str(fake)  # aynı venv'e düşer
    finally:
        fake.unlink()


def test_worker_python_fallback_no_env(client):
    assert worker_python("imgtool") == sys.executable


def test_imgtool_output_path_image_ops(client):
    for op in ("bg_remove", "upscale", "segment", "crop", "adjust"):
        r = client.post(
            "/v1/jobs", json={"kind": "imgtool", "params": {"op": op, "image_path": "x.png"}}
        )
        assert r.json()["params"]["output_path"].endswith(".png")


def test_imgtool_output_path_text_ops(client):
    for op in ("caption", "ocr"):
        r = client.post(
            "/v1/jobs", json={"kind": "imgtool", "params": {"op": op, "image_path": "x.png"}}
        )
        assert r.json()["params"]["output_path"].endswith(".json")


def test_imgtool_job_fails_gracefully_without_env(client):
    r = client.post(
        "/v1/jobs",
        json={"kind": "imgtool", "params": {"op": "bg_remove", "image_path": "x.png"}},
    )
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "worker ortamı eksik" in d["error"] or "transformers" in d["error"]


def test_imgtool_job_requires_image_path(client):
    r = client.post("/v1/jobs", json={"kind": "imgtool", "params": {"op": "caption"}})
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "image_path" in d["error"]


def test_imgtool_job_requires_known_op(client):
    r = client.post(
        "/v1/jobs", json={"kind": "imgtool", "params": {"op": "yok", "image_path": "x.png"}}
    )
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "bilinmeyen op" in d["error"]


def test_segment_requires_known_mode(client):
    r = client.post(
        "/v1/jobs",
        json={
            "kind": "imgtool",
            "params": {"op": "segment", "image_path": "x.png", "mode": "yok"},
        },
    )
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "segment mode" in d["error"]


# -- crop / adjust: torch gerektirmez, adapter doğrudan (worker'sız) test edilir ----


def test_crop_produces_correct_dimensions(tmp_path):
    from PIL import Image
    from studio_inference.adapters.imgtool import ImgToolAdapter

    src = tmp_path / "src.png"
    Image.new("RGB", (200, 100), (255, 0, 0)).save(src)
    out = tmp_path / "out.png"
    events = list(
        ImgToolAdapter().infer(
            {
                "op": "crop",
                "image_path": str(src),
                "output_path": str(out),
                "box": [10, 10, 60, 60],
            }
        )
    )
    assert events[-1]["event"] == "done"
    assert Image.open(out).size == (50, 50)


def test_crop_requires_box(tmp_path):
    from PIL import Image
    from studio_inference.adapters.imgtool import ImgToolAdapter

    src = tmp_path / "src.png"
    Image.new("RGB", (100, 100)).save(src)
    try:
        list(
            ImgToolAdapter().infer(
                {"op": "crop", "image_path": str(src), "output_path": str(tmp_path / "o.png")}
            )
        )
        raise AssertionError("beklenen RuntimeError fırlatılmadı")
    except RuntimeError as exc:
        assert "box" in str(exc)


def test_crop_clamps_out_of_bounds_box(tmp_path):
    from PIL import Image
    from studio_inference.adapters.imgtool import ImgToolAdapter

    src = tmp_path / "src.png"
    Image.new("RGB", (100, 100)).save(src)
    out = tmp_path / "out.png"
    list(
        ImgToolAdapter().infer(
            {
                "op": "crop",
                "image_path": str(src),
                "output_path": str(out),
                "box": [-50, -50, 200, 200],
            }
        )
    )
    assert Image.open(out).size == (100, 100)


def test_adjust_brightness_changes_pixels(tmp_path):
    from PIL import Image
    from studio_inference.adapters.imgtool import ImgToolAdapter

    src = tmp_path / "src.png"
    Image.new("RGB", (10, 10), (100, 100, 100)).save(src)
    out = tmp_path / "out.png"
    list(
        ImgToolAdapter().infer(
            {
                "op": "adjust",
                "image_path": str(src),
                "output_path": str(out),
                "brightness": 2.0,
            }
        )
    )
    px = Image.open(out).getpixel((5, 5))
    assert px[0] > 150


def test_adjust_no_params_is_noop(tmp_path):
    from PIL import Image
    from studio_inference.adapters.imgtool import ImgToolAdapter

    src = tmp_path / "src.png"
    Image.new("RGB", (10, 10), (77, 88, 99)).save(src)
    out = tmp_path / "out.png"
    list(
        ImgToolAdapter().infer(
            {"op": "adjust", "image_path": str(src), "output_path": str(out)}
        )
    )
    assert Image.open(out).getpixel((5, 5)) == (77, 88, 99)


def test_make_hocr_contains_bbox_text_and_escapes_html():
    from studio_inference.adapters.imgtool import _make_hocr

    items = [
        {"text": "Merhaba <b>", "confidence": 0.91, "bbox": [[1, 2], [10, 2], [10, 8], [1, 8]]}
    ]
    hocr = _make_hocr(items, width=20, height=10)
    assert 'class="ocr_page"' in hocr
    assert "bbox 1 2 10 8" in hocr
    assert "x_wconf 91" in hocr
    assert "&lt;b&gt;" in hocr
