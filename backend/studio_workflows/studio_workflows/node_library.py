"""Gerçek node türleri (Faz 10 DoD): Prompt → Görsel Üret (FLUX) → Upscale →
BG Remove → Kaydet. GPU-ağır düğümler mevcut Scheduler/Job altyapısına
dispatch edilir (`JobDispatch`); Prompt/Kaydet process içinde çalışır
(`InlineFn`) — bu modülü import etmek `NODE_TYPES` kayıt defterini doldurur.

Port sözleşmesi: image/mesh/audio/video değerleri en az `{"path": <dosya
yolu>}` içeren bir sözlüktür (job sonuçlarıyla aynı şekil); text/number/
boolean değerleri düz Python skalerleridir.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .nodes import JobDispatch, NodeTypeDef, ParamSpec, PortSpec, register_node_type


def _prompt_exec(inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    return {"text": str(params.get("text", ""))}


register_node_type(
    NodeTypeDef(
        type_id="core.prompt",
        category="core",
        label="Prompt",
        outputs=(PortSpec("text", "text"),),
        params=(ParamSpec("text", "string", default=""),),
        execute=_prompt_exec,
    )
)


def _build_image_gen_params(inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    if not params.get("model_path"):
        raise RuntimeError("image.generate: 'model_path' parametresi gerekli")
    return {
        "prompt": inputs["prompt"],
        "negative_prompt": params.get("negative_prompt", ""),
        "width": int(params.get("width", 1024)),
        "height": int(params.get("height", 1024)),
        "steps": int(params.get("steps", 20)),
        "seed": int(params.get("seed", -1)),
        "cfg": float(params.get("cfg", 3.5)),
        "model_path": params["model_path"],
        "model_family": params.get("model_family", "flux"),
        "mode": "txt2img",
    }


def _map_image_output(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "image": {
            "path": result["path"],
            "width": result.get("width"),
            "height": result.get("height"),
            "seed": result.get("seed"),
        }
    }


register_node_type(
    NodeTypeDef(
        type_id="image.generate",
        category="image",
        label="Görsel Üret (FLUX)",
        inputs=(PortSpec("prompt", "text"),),
        outputs=(PortSpec("image", "image"),),
        params=(
            ParamSpec("model_path", "string"),
            ParamSpec("model_family", "enum", default="flux", choices=("flux", "sd3", "sdxl")),
            ParamSpec("negative_prompt", "string", default=""),
            ParamSpec("width", "number", default=1024),
            ParamSpec("height", "number", default=1024),
            ParamSpec("steps", "number", default=20),
            ParamSpec("cfg", "number", default=3.5),
            ParamSpec("seed", "number", default=-1),
        ),
        execute=JobDispatch(
            kind="image", build_params=_build_image_gen_params, map_output=_map_image_output
        ),
    )
)


def _map_imgtool_image_output(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "image": {
            "path": result["path"],
            "width": result.get("width"),
            "height": result.get("height"),
        }
    }


def _build_upscale_params(inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    if not params.get("model_path"):
        raise RuntimeError("image.upscale: 'model_path' parametresi gerekli (bir ESRGAN modeli)")
    return {
        "op": "upscale",
        "image_path": inputs["image"]["path"],
        "model_path": params["model_path"],
    }


register_node_type(
    NodeTypeDef(
        type_id="image.upscale",
        category="image",
        label="Büyüt (Upscale)",
        inputs=(PortSpec("image", "image"),),
        outputs=(PortSpec("image", "image"),),
        params=(ParamSpec("model_path", "string"),),
        execute=JobDispatch(
            kind="imgtool", build_params=_build_upscale_params, map_output=_map_imgtool_image_output
        ),
    )
)


def _build_bg_remove_params(inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    return {
        "op": "bg_remove",
        "image_path": inputs["image"]["path"],
        "model_path": params.get("model_path", ""),
    }


register_node_type(
    NodeTypeDef(
        type_id="image.bg_remove",
        category="image",
        label="Arka Plan Kaldır",
        inputs=(PortSpec("image", "image"),),
        outputs=(PortSpec("image", "image"),),
        params=(ParamSpec("model_path", "string", default=""),),
        execute=JobDispatch(
            kind="imgtool",
            build_params=_build_bg_remove_params,
            map_output=_map_imgtool_image_output,
        ),
    )
)


def _save_exec(inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    dest = params.get("path")
    if not dest:
        raise RuntimeError("core.save: 'path' parametresi gerekli")
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    value = inputs["value"]
    if isinstance(value, dict) and "path" in value:
        shutil.copy2(value["path"], dest_path)
    else:
        dest_path.write_text(str(value), encoding="utf-8")
    return {"path": str(dest_path)}


register_node_type(
    NodeTypeDef(
        type_id="core.save",
        category="core",
        label="Kaydet",
        inputs=(PortSpec("value", "any"),),
        outputs=(PortSpec("path", "any"),),
        params=(ParamSpec("path", "string"),),
        execute=_save_exec,
    )
)
