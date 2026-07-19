"""Faz 8 iskelet testleri: video işi kablolaması (gerçek LTX-Video GPU smoke testinde).

Test ortamında video worker env'i (torch/diffusers) yoktur → iş, kurulum talimatı
içeren anlaşılır bir hatayla FAILED olmalı; sunucu süreci torch import etmemeli.
videotool yalnızca FFmpeg subprocess kullanır — worker env'i olmadan da çalışır
(FFmpeg PATH'te yoksa FileNotFoundError'a düşer, bu da anlaşılır şekilde sarmalanır
değil ham haliyle patlar — test bunu tolere eder).
"""

from __future__ import annotations

import sys

from studio_inference.scheduler import worker_python
from test_scheduler import _poll_status


def test_video_kind_available(client):
    r = client.post(
        "/v1/jobs", json={"kind": "video", "params": {"mode": "t2v", "prompt": "a cat"}}
    )
    assert r.status_code == 200


def test_videotool_kind_available(client):
    r = client.post("/v1/jobs", json={"kind": "videotool", "params": {"op": "trim"}})
    assert r.status_code == 200


def test_server_process_has_no_torch(client):
    assert "torch" not in sys.modules


def test_worker_python_video_shares_image_env(client):
    # test veri dizininde envs/image yok → ana venv'e düşer (imgtool ile aynı desen)
    assert worker_python("video") == worker_python("image")
    assert worker_python("videotool") == worker_python("image")


def test_video_output_path_is_mp4(client):
    r = client.post(
        "/v1/jobs", json={"kind": "video", "params": {"mode": "t2v", "prompt": "a cat"}}
    )
    assert r.json()["params"]["output_path"].endswith(".mp4")


def test_videotool_output_path_follows_format(client):
    r = client.post(
        "/v1/jobs",
        json={"kind": "videotool", "params": {"op": "convert", "format": "gif"}},
    )
    assert r.json()["params"]["output_path"].endswith(".gif")


def test_videotool_output_path_defaults_mp4(client):
    r = client.post("/v1/jobs", json={"kind": "videotool", "params": {"op": "trim"}})
    assert r.json()["params"]["output_path"].endswith(".mp4")


def test_video_job_requires_known_mode(client):
    r = client.post(
        "/v1/jobs", json={"kind": "video", "params": {"mode": "yok", "prompt": "x"}}
    )
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "bilinmeyen mode" in d["error"]


def test_video_i2v_requires_init_image(client):
    r = client.post(
        "/v1/jobs", json={"kind": "video", "params": {"mode": "i2v", "prompt": "x"}}
    )
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "init_image" in d["error"] or "worker ortamı eksik" in d["error"]


def test_video_job_fails_gracefully_without_env(client):
    r = client.post(
        "/v1/jobs", json={"kind": "video", "params": {"mode": "t2v", "prompt": "a cat"}}
    )
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "worker ortamı eksik" in d["error"] or "diffusers" in d["error"]


def test_videotool_unknown_op_fails(client):
    r = client.post("/v1/jobs", json={"kind": "videotool", "params": {"op": "yok"}})
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "bilinmeyen op" in d["error"]


def test_videotool_trim_requires_video_path(client):
    r = client.post("/v1/jobs", json={"kind": "videotool", "params": {"op": "trim"}})
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "video_path" in d["error"]


def test_videotool_concat_requires_two_paths(client):
    r = client.post(
        "/v1/jobs",
        json={"kind": "videotool", "params": {"op": "concat", "video_paths": ["a.mp4"]}},
    )
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "en az 2" in d["error"]


def test_videotool_speed_requires_valid_factor(client):
    r = client.post(
        "/v1/jobs",
        json={
            "kind": "videotool",
            "params": {"op": "speed", "video_path": "a.mp4", "factor": 1.0},
        },
    )
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "factor" in d["error"]


def test_videotool_add_audio_requires_both_paths(client):
    r = client.post(
        "/v1/jobs",
        json={"kind": "videotool", "params": {"op": "add_audio", "video_path": "a.mp4"}},
    )
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "audio_path" in d["error"]


def test_videotool_convert_requires_known_format(client):
    r = client.post(
        "/v1/jobs",
        json={
            "kind": "videotool",
            "params": {"op": "convert", "video_path": "a.mp4", "format": "wmv"},
        },
    )
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "bilinmeyen format" in d["error"]


def test_videotool_upscale_requires_video_path(client):
    r = client.post(
        "/v1/jobs",
        json={"kind": "videotool", "params": {"op": "upscale", "model_path": "m.pth"}},
    )
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "video_path" in d["error"]


def test_videotool_upscale_requires_model_path(client):
    r = client.post(
        "/v1/jobs",
        json={"kind": "videotool", "params": {"op": "upscale", "video_path": "a.mp4"}},
    )
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "model_path" in d["error"]


def test_resolve_pipeline_root_walks_up_to_model_index(tmp_path):
    """LTX-Video gibi çok-bileşenli pipeline'larda Model Browser'da hangi
    bileşen (transformer/vae/text_encoder) seçilirse seçilsin aynı köke
    çözülmeli — bkz. adapters/video.py modül docstring'i."""
    from studio_inference.adapters.video import _resolve_pipeline_root

    root = tmp_path / "LTX-Video"
    (root / "vae").mkdir(parents=True)
    (root / "transformer").mkdir()
    (root / "model_index.json").write_text("{}")
    (root / "vae" / "config.json").write_text("{}")
    weight = root / "vae" / "diffusion_pytorch_model.safetensors"
    weight.write_bytes(b"")

    assert _resolve_pipeline_root(str(weight)) == str(root)


def test_resolve_pipeline_root_falls_back_to_config_sibling(tmp_path):
    """model_index.json yoksa (tek-bileşenli, ör. ControlNet) sdxl.py'nin
    _resolve_repo desenine düşer: config.json kardeşi varsa üst klasör."""
    from studio_inference.adapters.video import _resolve_pipeline_root

    d = tmp_path / "some-controlnet"
    d.mkdir()
    (d / "config.json").write_text("{}")
    weight = d / "diffusion_pytorch_model.safetensors"
    weight.write_bytes(b"")

    assert _resolve_pipeline_root(str(weight)) == str(d)


def test_resolve_pipeline_root_passthrough_for_repo_id():
    from studio_inference.adapters.video import _resolve_pipeline_root

    assert _resolve_pipeline_root("Lightricks/LTX-Video") == "Lightricks/LTX-Video"


def test_video_unknown_family_fails(client):
    r = client.post(
        "/v1/jobs",
        json={"kind": "video", "params": {"family": "yok", "mode": "t2v", "prompt": "x"}},
    )
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "bilinmeyen family" in d["error"]


def test_video_hunyuan_i2v_rejected(client):
    """Hunyuan/Mochi bu dilimde yalnızca t2v destekliyor (i2v_cls=None)."""
    r = client.post(
        "/v1/jobs",
        json={
            "kind": "video",
            "params": {"family": "hunyuan", "mode": "i2v", "prompt": "x", "init_image": "x.png"},
        },
    )
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "yalnızca t2v destekliyor" in d["error"]


def test_video_wan_t2v_fails_gracefully_without_env(client):
    r = client.post(
        "/v1/jobs",
        json={"kind": "video", "params": {"family": "wan", "mode": "t2v", "prompt": "a cat"}},
    )
    d = _poll_status(client, r.json()["id"], {"failed"}, timeout=30)
    assert "worker ortamı eksik" in d["error"] or "diffusers" in d["error"]


def test_video_jobs_default_to_low_priority(client):
    """Faz 8 adım 2: video/videotool işleri, istemci açıkça belirtmediyse
    (vars. 0) düşük öncelik sınıfına (-1) düşer — hızlı iş türleri önce koşsun."""
    r = client.post(
        "/v1/jobs", json={"kind": "video", "params": {"mode": "t2v", "prompt": "a cat"}}
    )
    assert r.json()["priority"] == -1

    r2 = client.post("/v1/jobs", json={"kind": "videotool", "params": {"op": "trim"}})
    assert r2.json()["priority"] == -1


def test_video_job_explicit_priority_respected(client):
    r = client.post(
        "/v1/jobs",
        json={"kind": "video", "params": {"mode": "t2v", "prompt": "a cat"}, "priority": 5},
    )
    assert r.json()["priority"] == 5


def test_image_job_priority_unaffected(client):
    """Yalnızca video/videotool düşük önceliğe düşer — diğer iş türleri etkilenmez."""
    r = client.post(
        "/v1/jobs", json={"kind": "audio", "params": {"op": "tts", "text": "merhaba"}}
    )
    assert r.json()["priority"] == 0


def test_recommend_settings_noop_without_torch():
    """Faz 8 adım 3: VRAM'e göre otomatik çözünürlük önerisi — test ortamında
    torch/GPU yok, bu yüzden güvenli bir no-op'a düşmeli (girdi aynen döner),
    RuntimeError/ImportError sızdırmamalı."""
    from studio_inference.adapters.video import recommend_settings

    w, h, f, vm = recommend_settings("ltx", 704, 480, 121, "max")
    assert (w, h, f, vm) == (704, 480, 121, "max")


def test_estimate_seconds_uses_measured_rates():
    from studio_inference.adapters.video import estimate_seconds

    # LTX en hızlısı (~13 adım/sn canlıda ölçüldü) — Mochi/Hunyuan çok daha yavaş
    assert estimate_seconds("ltx", 30) < estimate_seconds("mochi", 30)
    assert estimate_seconds("ltx", 30) < estimate_seconds("hunyuan", 30)


def test_estimate_seconds_unknown_family_falls_back():
    from studio_inference.adapters.video import estimate_seconds

    # bilinmeyen aile de wan ile aynı varsayılan (2.0 adım/sn) hıza düşer
    assert estimate_seconds("yok", 30) == estimate_seconds("wan", 30)
