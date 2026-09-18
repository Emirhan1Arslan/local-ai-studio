"""Yeni nesil görsel üretim adapter'ı (2026 modelleri): Z-Image + HiDream-O1.

DİKKAT: Bu modül worker env'inde (envs/imagex — Python 3.10 + sistem torch 2.11/cu128)
çalışır; 3.11+ sözdizimi KULLANMA. torch/diffusers/transformers importları metod
içindedir; API süreci bu modülü hiç yüklemez.

Neden ayrı bir iş türü ("imagex") ve ayrı env: bu modeller diffusers>=0.36 ve
transformers>=4.51 (Qwen3/Qwen3-VL kodlayıcıları) ister; `envs/image`'ın
diffusers 0.33 + transformers 4.46.3 pinleri SDXL/LTX/Wan için dondurulmuş
(docs/worker-envs.md). Aynı venv'i yükseltmek yerine `envs/imagex` kuruldu —
SDXL zinciri regresyon riski almaz.

Aileler (model klasöründen otomatik algılanır — `detect_family`):
- zimage: Tongyi-MAI Z-Image-Turbo (6B S3-DiT, Apache-2.0). diffusers
  `ZImagePipeline`; turbo için 9 adım (8 NFE) + guidance 0. img2img ve inpaint
  `from_pipe` ile aynı ağırlıklardan.
- hidream_o1: HiDream-ai HiDream-O1-Image / -Dev (8B piksel-native DiT, Qwen3-VL
  omurgası, MIT). Vendor edilmiş çıkarım kodu (vendor/hidream_o1). Dev: 28 adım,
  guidance 0, flash scheduler; Full: 50 adım, guidance 5, UniPC. Çıktı HER ZAMAN
  ~4 MP'dir: width/height yalnız en-boy oranını seçer, model kendi çözünürlük
  setine (2048², 2304×1728, 2560×1440, 3104×1312 …) oturtur — 1024² isteği 2048²
  üretir (canlı: 28 adım 2048² ≈ 30 sn, tepe VRAM 17.5 GB). Tek referans görselle
  düzenleme (mode=edit, init_image) desteklenir.

params:
- model_path: klasör kökü ya da içindeki herhangi bir dosya (Model Browser)
- family: zimage | hidream_o1 (boşsa otomatik)
- mode: txt2img (vars.) | img2img (zimage) | edit (hidream_o1, init_image referans)
- prompt / negative_prompt (yalnız guidance > 1 iken anlamlı) · width/height
- steps · cfg (boş/-1 → ailenin varsayılanı) · seed (-1 rastgele)
- init_image · strength (img2img, vars. 0.6)
- vram_mode: max (vars.) | balanced (model CPU offload)
- preview: canlı önizleme (vars. true; hidream_o1 x0 tahmini, zimage yok)
- output_path (zorunlu)
"""

from __future__ import annotations

import json
import os
import queue
import random
import sys
import threading
from collections.abc import Iterator
from typing import Any

_VENDOR_HIDREAM = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "vendor", "hidream_o1")
)

FAMILIES = ("zimage", "hidream_o1")

# Aile varsayılanları — UI/MCP boş bırakırsa bunlar kullanılır.
DEFAULTS: dict[str, dict[str, Any]] = {
    "zimage": {"steps": 9, "cfg": 0.0, "width": 1024, "height": 1024},
    "hidream_o1:dev": {"steps": 28, "cfg": 0.0, "shift": 1.0, "width": 1024, "height": 1024},
    "hidream_o1:full": {"steps": 50, "cfg": 5.0, "shift": 3.0, "width": 1024, "height": 1024},
}


def resolve_root(model_path: str) -> str:
    """Model Browser tekil bir dosya verebilir (ör. transformer/…-00001.safetensors);
    model_index.json (diffusers) ya da config.json + model.safetensors.index.json
    (transformers-tarzı HiDream) içeren en yakın üst klasöre çıkar."""
    p = os.path.abspath(str(model_path))
    cur = p if os.path.isdir(p) else os.path.dirname(p)
    for _ in range(5):
        if os.path.isfile(os.path.join(cur, "model_index.json")) or (
            os.path.isfile(os.path.join(cur, "config.json"))
            and os.path.isfile(os.path.join(cur, "model.safetensors.index.json"))
        ):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return p if os.path.isdir(p) else os.path.dirname(p)


def detect_family(root: str) -> str:
    """Klasör içeriğinden aile: model_index.json `_class_name` (ZImage…) ya da
    HiDream'in Qwen3-VL config'i + `final_layer2` görüntü başı ağırlıkları."""
    idx = os.path.join(root, "model_index.json")
    if os.path.isfile(idx):
        try:
            with open(idx, encoding="utf-8") as f:
                cname = str(json.load(f).get("_class_name", ""))
        except (OSError, ValueError):
            cname = ""
        if cname.startswith("ZImage"):
            return "zimage"
        return ""
    cfg = os.path.join(root, "config.json")
    if os.path.isfile(cfg):
        try:
            with open(cfg, encoding="utf-8") as f:
                model_type = str(json.load(f).get("model_type", ""))
        except (OSError, ValueError):
            model_type = ""
        if model_type == "qwen3_vl" and _has_hidream_head(root):
            return "hidream_o1"
    return ""


def _has_hidream_head(root: str) -> bool:
    if "hidream" in os.path.basename(root).lower():
        return True
    index = os.path.join(root, "model.safetensors.index.json")
    try:
        with open(index, encoding="utf-8") as f:
            weight_map = json.load(f).get("weight_map", {})
    except (OSError, ValueError):
        return False
    return any("final_layer2" in k for k in weight_map)


def hidream_variant(root: str) -> str:
    return "dev" if "dev" in os.path.basename(root).lower() else "full"


def defaults_for(family: str, root: str = "") -> dict[str, Any]:
    key = family if family != "hidream_o1" else "hidream_o1:" + hidream_variant(root)
    return dict(DEFAULTS.get(key, DEFAULTS["zimage"]))


def _param(params: dict, name: str, default: Any, cast: Any) -> Any:
    raw = params.get(name, None)
    negative = isinstance(raw, (int, float)) and raw < 0 and name != "seed"
    if raw is None or raw == "" or negative:
        return default
    return cast(raw)


def _a1111_metadata(
    params: dict, seed: int, size: tuple, family: str, steps: int, cfg: float
) -> str:
    lines = [str(params.get("prompt", ""))]
    neg = params.get("negative_prompt", "")
    if neg:
        lines.append("Negative prompt: " + str(neg))
    # Py3.10: f-string ifadesi içinde ters bölü olamaz — basename önce hesaplanır.
    model_name = os.path.basename(str(params.get("model_path", "")).rstrip("/\\"))
    lines.append(
        f"Steps: {steps}, CFG scale: {cfg}, Seed: {seed}, Size: {size[0]}x{size[1]}, "
        f"Model: {model_name}, Family: {family}"
    )
    return "\n".join(lines)


def _thumb_data_uri(image: Any) -> str:
    import base64
    import io

    thumb = image.copy()
    thumb.thumbnail((192, 192))
    buf = io.BytesIO()
    thumb.convert("RGB").save(buf, format="JPEG", quality=65)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class ImageXAdapter:
    def __init__(self) -> None:
        self._pipe: Any = None
        self._model: Any = None
        self._processor: Any = None
        self._root = ""
        self._family = ""
        self._vram_mode = "max"

    # -- yükleme ---------------------------------------------------------------
    def load(
        self, model_path: str, family: str = "", vram_mode: str = "max", **kwargs: Any
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "imagex worker ortamı eksik (torch bulunamadı). Kurulum: docs/worker-envs.md "
                "'imagex env'  |  " + str(exc)
            ) from exc
        root = resolve_root(model_path)
        family = family or detect_family(root)
        if family not in FAMILIES:
            raise RuntimeError(
                f"model ailesi algılanamadı: {root!r} — Z-Image (model_index.json) ya da "
                "HiDream-O1 (config.json + model.safetensors.index.json) klasörü bekleniyor"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("imagex aileleri için CUDA gerekli")
        if family == "zimage":
            self._load_zimage(root, vram_mode)
        else:
            self._load_hidream(root, vram_mode)
        self._root = root
        self._family = family
        self._vram_mode = vram_mode

    def _load_zimage(self, root: str, vram_mode: str) -> None:
        import torch

        try:
            from diffusers import ZImagePipeline
        except ImportError as exc:
            raise RuntimeError(
                "diffusers'ta ZImagePipeline yok — envs/imagex'e diffusers>=0.36 gerekir "
                "(docs/worker-envs.md)  |  " + str(exc)
            ) from exc
        pipe = ZImagePipeline.from_pretrained(root, torch_dtype=torch.bfloat16)
        if vram_mode == "balanced":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe

    def _load_hidream(self, root: str, vram_mode: str) -> None:
        import torch

        if _VENDOR_HIDREAM not in sys.path:
            sys.path.insert(0, _VENDOR_HIDREAM)
        try:
            from hidream_o1.qwen3_vl_transformers import Qwen3VLForConditionalGeneration
            from transformers import AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "HiDream-O1 için transformers==4.57.x gerekir (envs/imagex, docs/worker-envs.md)"
                "  |  " + str(exc)
            ) from exc
        processor = AutoProcessor.from_pretrained(root)
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        # upstream inference.py::add_special_tokens — pipeline bu kısayolları okur
        tokenizer.boi_token = "<|boi_token|>"
        tokenizer.bor_token = "<|bor_token|>"
        tokenizer.eor_token = "<|eor_token|>"
        tokenizer.bot_token = "<|bot_token|>"
        tokenizer.tms_token = "<|tms_token|>"
        device_map = "auto" if vram_mode == "balanced" else "cuda"
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            root, torch_dtype=torch.bfloat16, device_map=device_map
        ).eval()
        self._model = model
        self._processor = processor

    # -- üretim -----------------------------------------------------------------
    def infer(self, params: dict) -> Iterator[dict]:
        model_path = str(params.get("model_path", "") or "")
        if not model_path:
            raise RuntimeError(
                "model_path gerekli (Model Browser'dan bir Z-Image/HiDream klasörü seçin)"
            )
        output_path = str(params.get("output_path", "") or "")
        if not output_path:
            raise RuntimeError("output_path gerekli")
        vram_mode = str(params.get("vram_mode", "max") or "max")
        if vram_mode not in ("max", "balanced"):
            raise RuntimeError(f"bilinmeyen vram_mode: {vram_mode!r} (max|balanced)")

        root = resolve_root(model_path)
        family = str(params.get("family", "") or "") or detect_family(root)
        if self._family and (self._root != root or self._vram_mode != vram_mode):
            self.unload()
        if not self._family:
            yield {"event": "progress", "value": 0.0, "message": "model yükleniyor…"}
            self.load(model_path, family=family, vram_mode=vram_mode)
        family = self._family

        mode = str(params.get("mode", "txt2img") or "txt2img")
        d = defaults_for(family, root)
        steps = _param(params, "steps", d["steps"], int)
        cfg = _param(params, "cfg", d["cfg"], float)
        width = _param(params, "width", d["width"], int)
        height = _param(params, "height", d["height"], int)
        seed = int(params.get("seed", -1) if params.get("seed") not in (None, "") else -1)
        if seed < 0:
            seed = random.randrange(2**31)
        want_preview = bool(params.get("preview", True))

        events: queue.Queue[dict | None] = queue.Queue()
        outcome: dict = {}

        if family == "zimage":
            target = self._run_zimage
        else:
            target = self._run_hidream

        def run() -> None:
            try:
                outcome["image"] = target(
                    params, mode, width, height, steps, cfg, seed, want_preview, events
                )
            except Exception as exc:  # noqa: BLE001 - thread sınırı: hata ana akışa taşınır
                outcome["error"] = str(exc)
            finally:
                events.put(None)

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        while True:
            item = events.get()
            if item is None:
                break
            yield item
        worker.join()
        if "error" in outcome:
            raise RuntimeError(outcome["error"])

        from PIL.PngImagePlugin import PngInfo

        image = outcome["image"]
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        meta = PngInfo()
        meta.add_text(
            "parameters",
            _a1111_metadata(params, seed, (image.width, image.height), family, steps, cfg),
        )
        image.save(output_path, pnginfo=meta)
        yield {
            "event": "done",
            "result": {
                "path": output_path,
                "seed": seed,
                "width": image.width,
                "height": image.height,
                "mode": mode,
                "family": family,
                "steps": steps,
                "cfg": cfg,
            },
        }

    def _run_zimage(
        self,
        params: dict,
        mode: str,
        width: int,
        height: int,
        steps: int,
        cfg: float,
        seed: int,
        want_preview: bool,
        events: queue.Queue,
    ) -> Any:
        import torch
        from PIL import Image

        width = max(256, (width // 16) * 16)
        height = max(256, (height // 16) * 16)
        generator = torch.Generator("cuda").manual_seed(seed)

        def on_step(pipe_obj: Any, step: int, timestep: Any, cb_kwargs: dict) -> dict:
            total = getattr(pipe_obj, "_num_timesteps", None) or steps
            events.put(
                {
                    "event": "progress",
                    "value": (step + 1) / max(1, total),
                    "message": f"adım {step + 1}/{total}",
                }
            )
            return cb_kwargs

        call: dict[str, Any] = {
            "prompt": str(params.get("prompt", "")),
            "num_inference_steps": steps,
            "guidance_scale": cfg,
            "generator": generator,
            "callback_on_step_end": on_step,
        }
        if cfg > 1.0 and params.get("negative_prompt"):
            call["negative_prompt"] = str(params["negative_prompt"])
        if mode == "img2img":
            from diffusers import ZImageImg2ImgPipeline

            init_path = str(params.get("init_image", "") or "")
            if not init_path:
                raise RuntimeError("img2img için init_image gerekli")
            work = ZImageImg2ImgPipeline.from_pipe(self._pipe)
            call["image"] = Image.open(init_path).convert("RGB").resize((width, height))
            call["strength"] = float(params.get("strength", 0.6))
        elif mode == "txt2img":
            work = self._pipe
            call["width"] = width
            call["height"] = height
        else:
            raise RuntimeError(f"zimage için bilinmeyen mode: {mode!r} (txt2img|img2img)")
        return work(**call).images[0]

    def _run_hidream(
        self,
        params: dict,
        mode: str,
        width: int,
        height: int,
        steps: int,
        cfg: float,
        seed: int,
        want_preview: bool,
        events: queue.Queue,
    ) -> Any:
        from hidream_o1.pipeline import DEFAULT_TIMESTEPS, generate_image

        variant = hidream_variant(self._root)
        d = defaults_for("hidream_o1", self._root)
        shift = float(params.get("shift", d["shift"]))
        ref_paths: list[str] = []
        if mode == "edit":
            init_path = str(params.get("init_image", "") or "")
            if not init_path:
                raise RuntimeError("edit için init_image (referans görsel) gerekli")
            ref_paths = [init_path]
        elif mode != "txt2img":
            raise RuntimeError(f"hidream_o1 için bilinmeyen mode: {mode!r} (txt2img|edit)")

        if variant == "dev":
            # upstream inference.py: dev → önceden tanımlı 28 zaman adımı; düzenlemede flow_match
            scheduler_name = "flow_match" if ref_paths else "flash"
            timesteps_list = DEFAULT_TIMESTEPS if steps == 28 else None
        else:
            scheduler_name = "default"
            timesteps_list = None

        preview_every = 1 if steps <= 12 else 3

        def callback(step_idx: int, total: int, decode_preview: Any) -> None:
            ev = {
                "event": "progress",
                "value": (step_idx + 1) / max(1, total),
                "message": f"adım {step_idx + 1}/{total}",
            }
            if want_preview and (step_idx % preview_every == 0 or step_idx + 1 == total):
                try:
                    ev["preview"] = _thumb_data_uri(decode_preview())
                except Exception:
                    pass  # önizleme asla üretimi düşürmesin
            events.put(ev)

        extra: dict[str, Any] = {}
        if scheduler_name == "flash":
            extra.update(
                {
                    "noise_scale_start": float(params.get("noise_scale_start", 7.5)),
                    "noise_scale_end": float(params.get("noise_scale_end", 7.5)),
                    "noise_clip_std": float(params.get("noise_clip_std", 2.5)),
                }
            )
        return generate_image(
            model=self._model,
            processor=self._processor,
            prompt=str(params.get("prompt", "")),
            ref_image_paths=ref_paths,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=cfg,
            shift=shift,
            timesteps_list=timesteps_list,
            scheduler_name=scheduler_name,
            seed=seed,
            callback=callback,
            **extra,
        )

    def unload(self) -> None:
        self._pipe = None
        self._model = None
        self._processor = None
        self._root = ""
        self._family = ""
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def vram_estimate(self, params: dict) -> int:
        root = resolve_root(str(params.get("model_path", "") or ""))
        family = str(params.get("family", "") or "") or detect_family(root)
        balanced = params.get("vram_mode") == "balanced"
        if family == "hidream_o1":
            # 8B bf16 ≈ 16GB ağırlık + 2048² için uzun dizi aktivasyonları — profillenecek
            return 12_000 if balanced else 22_000
        # zimage: 6B DiT bf16 ≈ 12GB + Qwen3 metin kodlayıcı (~8GB) — upstream "16GB'a sığar" diyor
        return 9_000 if balanced else 17_000
