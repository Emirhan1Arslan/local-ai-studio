"""Diffusers görüntü adapter'ı (Faz 4): SDXL/SD3/FLUX + img2img/inpaint/outpaint.

DİKKAT: Bu modül worker env'inde (Python 3.10 + sistem torch 2.11/cu128) çalışır —
3.11+ sözdizimi KULLANMA. torch/diffusers importları metod içindedir; API süreci
bu modülü hiç yüklemez.

params:
- model_path: .safetensors dosyası veya diffusers dizini (zorunlu)
- model_family: sdxl | sd3 | flux — boşsa dosya başlığından otomatik algılanır
- mode: txt2img (vars.) | img2img | inpaint | outpaint
- init_image: kaynak görsel yolu (img2img/inpaint/outpaint için zorunlu)
- mask_image: maske yolu, beyaz = yeniden çizilecek alan (inpaint için zorunlu)
- strength: 0-1 denoise gücü (img2img vars. 0.75; inpaint/outpaint 1.0)
- outpaint: {"left","right","top","bottom"} piksel cinsinden tuval genişletme
- sampler: euler | euler_a | dpmpp_2m | dpmpp_2m_karras | dpmpp_sde_karras |
           unipc | ddim | heun | lms (yalnız SDXL; boşsa modelin varsayılanı)
- vram_mode: max (vars.) | balanced (model offload) | low (sequential+slicing+tiling)
- preview: canlı latent önizlemesi (vars. true; yalnız SDXL, VAE decode'suz ucuz yaklaşım)
- controlnets: [{"model": yol|repo, "image": yol, "scale", "preprocess": none|canny}]
- ip_adapter: {"image": yol, "scale", "repo", "subfolder", "weight_name"}
- prompt / negative_prompt · width/height · steps (30) · cfg · seed · loras · output_path
"""

from __future__ import annotations

import json
import os
import queue
import random
import struct
import threading
from collections.abc import Iterator
from typing import Any

_SAMPLERS = {
    "euler": ("EulerDiscreteScheduler", {}),
    "euler_a": ("EulerAncestralDiscreteScheduler", {}),
    "dpmpp_2m": ("DPMSolverMultistepScheduler", {"algorithm_type": "dpmsolver++"}),
    "dpmpp_2m_karras": (
        "DPMSolverMultistepScheduler",
        {"algorithm_type": "dpmsolver++", "use_karras_sigmas": True},
    ),
    "dpmpp_sde_karras": (
        "DPMSolverMultistepScheduler",
        {"algorithm_type": "sde-dpmsolver++", "use_karras_sigmas": True},
    ),
    "unipc": ("UniPCMultistepScheduler", {}),
    "ddim": ("DDIMScheduler", {}),
    "heun": ("HeunDiscreteScheduler", {}),
    "lms": ("LMSDiscreteScheduler", {}),
}

_SAMPLER_LABELS = {
    "euler": "Euler",
    "euler_a": "Euler a",
    "dpmpp_2m": "DPM++ 2M",
    "dpmpp_2m_karras": "DPM++ 2M Karras",
    "dpmpp_sde_karras": "DPM++ SDE Karras",
    "unipc": "UniPC",
    "ddim": "DDIM",
    "heun": "Heun",
    "lms": "LMS",
}

# SDXL 4 kanallı latent → RGB doğrusal yaklaşımı (canlı önizleme; VAE decode yok)
_SDXL_LATENT_RGB = [
    [0.3651, 0.4232, 0.4341],
    [-0.2533, -0.0042, 0.1068],
    [0.1076, 0.1111, -0.0362],
    [-0.3165, -0.2690, -0.2188],
]


def _align(value: Any, multiple: int, lo: int = 256, hi: int = 4096) -> int:
    v = max(lo, min(hi, int(value)))
    return max(lo, (v // multiple) * multiple)


def _lora_name(path: str) -> str:
    return str(path).replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]


def _detect_family(model_path: str) -> str:
    """Model ailesini dosyadan algılar: safetensors başlığındaki anahtar adları
    (flux: double_blocks, sd3: joint_blocks) ya da diffusers dizininde model_index.json."""
    p = str(model_path)
    if not p.endswith(".safetensors"):
        try:
            with open(os.path.join(p, "model_index.json"), encoding="utf-8") as f:
                cname = json.load(f).get("_class_name", "")
        except (OSError, ValueError):
            return "sdxl"
        if "Flux" in cname:
            return "flux"
        if "StableDiffusion3" in cname:
            return "sd3"
        return "sdxl"
    try:
        with open(p, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(min(n, 100_000_000)))
    except (OSError, ValueError, struct.error):
        return "sdxl"
    for key in header:
        if "double_blocks." in key:
            return "flux"
        if "joint_blocks" in key:
            return "sd3"
    return "sdxl"


def _expand_canvas(image: Any, left: int, top: int, right: int, bottom: int, feather: int = 24):
    """Outpaint hazırlığı: tuvali büyütür, yeni alanı gerilmiş kenarlarla doldurur.
    Maske yeni alanı beyaz işaretler ve dikiş kaybolsun diye feather kadar içeri taşar.
    (genişletilmiş_görsel, maske) döner."""
    from PIL import Image, ImageFilter

    w, h = image.size
    nw, nh = w + left + right, h + top + bottom
    canvas = image.resize((nw, nh))
    canvas.paste(image, (left, top))
    mask = Image.new("L", (nw, nh), 255)
    kw, kh = max(1, w - 2 * feather), max(1, h - 2 * feather)
    mask.paste(Image.new("L", (kw, kh), 0), (left + feather, top + feather))
    if feather > 1:
        mask = mask.filter(ImageFilter.GaussianBlur(feather // 2))
    return canvas, mask


def _a1111_metadata(params: dict, seed: int, size: tuple | None = None) -> str:
    """Üretim metadata'sını A1111 uyumlu 'parameters' metnine dönüştürür (Faz 4 planı)."""
    neg = params.get("negative_prompt", "")
    lines = [params.get("prompt", "")]
    if neg:
        lines.append("Negative prompt: " + neg)
    w, h = size if size else (params.get("width", 1024), params.get("height", 1024))
    parts = ["Steps: {}".format(params.get("steps", 30))]
    sampler = str(params.get("sampler", "") or "")
    if sampler:
        parts.append(f"Sampler: {_SAMPLER_LABELS.get(sampler, sampler)}")
    parts.append("CFG scale: {}".format(params.get("cfg", 7.0)))
    parts.append(f"Seed: {seed}")
    parts.append(f"Size: {w}x{h}")
    parts.append(
        "Model: {}".format(str(params.get("model_path", "")).replace("\\", "/").rsplit("/", 1)[-1])
    )
    mode = str(params.get("mode", "txt2img") or "txt2img")
    if mode != "txt2img" and params.get("strength") is not None:
        parts.append("Denoising strength: {}".format(params.get("strength")))
    loras = params.get("loras") or []
    if loras:
        parts.append(
            "Lora: "
            + ", ".join(
                "{}:{}".format(_lora_name(lo.get("path", "")), lo.get("weight", 1.0))
                for lo in loras
            )
        )
    lines.append(", ".join(parts))
    return "\n".join(lines)


def _make_sampler(name: str, config: Any):
    import diffusers

    spec = _SAMPLERS.get(name)
    if spec is None:
        raise RuntimeError(
            "bilinmeyen sampler: {!r} (geçerli: {})".format(name, ", ".join(sorted(_SAMPLERS)))
        )
    return getattr(diffusers, spec[0]).from_config(config, **spec[1])


def _latent_preview_jpeg(latents: Any) -> str | None:
    """4 kanallı SDXL latent'ini küçük bir JPEG data-URI'ye çevirir (ucuz yaklaşım)."""
    import base64
    import io

    import torch
    from PIL import Image

    if latents is None or latents.ndim != 4 or latents.shape[1] != 4:
        return None
    coefs = torch.tensor(_SDXL_LATENT_RGB, dtype=latents.dtype, device=latents.device)
    rgb = torch.einsum("chw,cr->rhw", latents[0], coefs)
    arr = ((rgb + 1.0) / 2.0).clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    img = Image.fromarray(arr)
    img.thumbnail((192, 192))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=65)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _control_image(spec: dict, size: tuple):
    """ControlNet koşul görselini hazırlar (none: olduğu gibi, canny: kenar çıkarımı)."""
    from PIL import Image

    path = str(spec.get("image", ""))
    if not path:
        raise RuntimeError("controlnet 'image' alanı gerekli")
    img = Image.open(path).convert("RGB").resize(size)
    if str(spec.get("preprocess", "none")) == "canny":
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "canny ön işlemcisi için opencv gerekli: image env'inde "
                "'pip install opencv-python-headless' (docs/worker-envs.md)"
            ) from exc
        edges = cv2.Canny(
            np.asarray(img), int(spec.get("canny_low", 100)), int(spec.get("canny_high", 200))
        )
        img = Image.fromarray(np.stack([edges] * 3, axis=-1))
    return img


class SdxlAdapter:
    def __init__(self) -> None:
        self._pipe = None
        self._model_path = ""
        self._family = "sdxl"
        self._vram_mode = "max"
        self._default_scheduler = None
        self._controlnets: dict[str, Any] = {}

    def load(
        self, model_path: str, family: str = "", vram_mode: str = "max", **kwargs: Any
    ) -> None:
        try:
            import torch
            from diffusers import (
                AutoPipelineForText2Image,
                FluxPipeline,
                StableDiffusion3Pipeline,
                StableDiffusionXLPipeline,
            )
        except ImportError as exc:
            raise RuntimeError(
                "image worker ortamı eksik (torch/diffusers bulunamadı). "
                "Kurulum reçetesi: docs/worker-envs.md  |  " + str(exc)
            ) from exc

        family = family or _detect_family(model_path)
        use_cuda = torch.cuda.is_available()
        if family in ("flux", "sd3"):
            dtype = torch.bfloat16 if use_cuda else torch.float32
        else:
            dtype = torch.float16 if use_cuda else torch.float32

        if str(model_path).endswith(".safetensors"):
            cls = {"flux": FluxPipeline, "sd3": StableDiffusion3Pipeline}.get(
                family, StableDiffusionXLPipeline
            )
            pipe = cls.from_single_file(model_path, torch_dtype=dtype)
        else:
            pipe = AutoPipelineForText2Image.from_pretrained(model_path, torch_dtype=dtype)

        if use_cuda:
            if vram_mode == "low":
                pipe.enable_attention_slicing()
                try:
                    pipe.vae.enable_tiling()
                except Exception:
                    pass
                pipe.enable_sequential_cpu_offload()
            elif vram_mode == "balanced":
                pipe.enable_model_cpu_offload()
            else:
                pipe.to("cuda")

        self._pipe = pipe
        self._model_path = str(model_path)
        self._family = family
        self._vram_mode = vram_mode
        self._default_scheduler = pipe.scheduler

    def _ensure_controlnets(self, specs: list, dtype: Any, use_cuda: bool) -> list:
        from diffusers import ControlNetModel

        new_cache: dict[str, Any] = {}
        models = []
        for spec in specs:
            key = str(spec.get("model", ""))
            if not key:
                raise RuntimeError("controlnet 'model' alanı gerekli (yol ya da HF repo id)")
            cn = self._controlnets.get(key)
            if cn is None:
                if key.endswith(".safetensors"):
                    cn = ControlNetModel.from_single_file(key, torch_dtype=dtype)
                else:
                    cn = ControlNetModel.from_pretrained(key, torch_dtype=dtype)
                if use_cuda:
                    cn.to("cuda")
            new_cache[key] = cn
            models.append(cn)
        dropped = set(self._controlnets) - set(new_cache)
        self._controlnets = new_cache
        if dropped and use_cuda:
            import torch

            torch.cuda.empty_cache()
        return models

    def infer(self, params: dict) -> Iterator[dict]:
        model_path = params.get("model_path", "")
        if not model_path:
            raise RuntimeError("model_path gerekli (Model Browser'dan bir checkpoint seçin)")
        output_path = params.get("output_path", "")
        if not output_path:
            raise RuntimeError("output_path gerekli")

        mode = str(params.get("mode", "txt2img") or "txt2img")
        if mode not in ("txt2img", "img2img", "inpaint", "outpaint"):
            raise RuntimeError(f"bilinmeyen mode: {mode!r}")
        vram_mode = str(params.get("vram_mode", "max") or "max")
        if vram_mode not in ("max", "balanced", "low"):
            raise RuntimeError(f"bilinmeyen vram_mode: {vram_mode!r} (max|balanced|low)")
        if vram_mode != "max" and mode != "txt2img":
            raise RuntimeError(
                "offload modlarında şimdilik yalnız txt2img desteklenir (vram_mode=max kullanın)"
            )

        # Kalıcı worker: model komutlar arasında bellekte kalır; yol/VRAM modu değişirse takas.
        if self._pipe is not None and (
            self._model_path != str(model_path) or self._vram_mode != vram_mode
        ):
            self.unload()
        if self._pipe is None:
            yield {"event": "progress", "value": 0.0, "message": "model yükleniyor…"}
            self.load(
                model_path,
                family=str(params.get("model_family", "") or ""),
                vram_mode=vram_mode,
            )

        import torch
        from diffusers import AutoPipelineForImage2Image, AutoPipelineForInpainting
        from PIL import Image
        from PIL.PngImagePlugin import PngInfo

        family = self._family
        mult = 16 if family == "flux" else 8
        steps = int(params.get("steps", 30))
        width = _align(params.get("width", 1024), mult)
        height = _align(params.get("height", 1024), mult)
        seed = int(params.get("seed", -1))
        if seed < 0:
            seed = random.randrange(2**31)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device).manual_seed(seed)

        # -- kaynak görsel / maske hazırlığı ---------------------------------
        init_img = mask_img = None
        strength = None
        if mode != "txt2img":
            init_path = str(params.get("init_image", ""))
            if not init_path:
                raise RuntimeError(f"{mode} için init_image gerekli")
            init_img = Image.open(init_path).convert("RGB")
            strength = float(params.get("strength", 0.75 if mode == "img2img" else 1.0))
        if mode == "img2img":
            init_img = init_img.resize((width, height))
        elif mode == "inpaint":
            mask_path = str(params.get("mask_image", ""))
            if not mask_path:
                raise RuntimeError("inpaint için mask_image gerekli (beyaz = yeniden çizilecek)")
            init_img = init_img.resize((width, height))
            mask_img = Image.open(mask_path).convert("L").resize((width, height))
        elif mode == "outpaint":
            op = params.get("outpaint") or {}
            left, top = max(0, int(op.get("left", 0))), max(0, int(op.get("top", 0)))
            right, bottom = max(0, int(op.get("right", 0))), max(0, int(op.get("bottom", 0)))
            if not (left or top or right or bottom):
                raise RuntimeError("outpaint için en az bir yön > 0 olmalı")
            w0, h0 = init_img.size
            right += (-(w0 + left + right)) % mult  # tuval boyutu 8'in katı kalsın
            bottom += (-(h0 + top + bottom)) % mult
            init_img, mask_img = _expand_canvas(init_img, left, top, right, bottom)
            width, height = init_img.size

        # -- çalışma pipeline'ı (mod + controlnet) ---------------------------
        cn_specs = list(params.get("controlnets") or [])
        control_images: list = []
        cn_scales: list = []
        if cn_specs:
            if family != "sdxl":
                raise RuntimeError("controlnet şimdilik yalnız SDXL ailesinde destekleniyor")
            if mode != "txt2img":
                raise RuntimeError("controlnet şimdilik yalnız txt2img modunda destekleniyor")
            if any(str(s.get("model", "")) not in self._controlnets for s in cn_specs):
                yield {"event": "progress", "value": 0.0, "message": "ControlNet yükleniyor…"}
            cn_models = self._ensure_controlnets(
                cn_specs, self._pipe.unet.dtype, device == "cuda"
            )
            control_images = [_control_image(s, (width, height)) for s in cn_specs]
            cn_scales = [float(s.get("scale", 0.8)) for s in cn_specs]
            from diffusers import AutoPipelineForText2Image

            work = AutoPipelineForText2Image.from_pipe(
                self._pipe, controlnet=cn_models if len(cn_models) > 1 else cn_models[0]
            )
        elif mode == "img2img":
            work = AutoPipelineForImage2Image.from_pipe(self._pipe)
        elif mode in ("inpaint", "outpaint"):
            work = AutoPipelineForInpainting.from_pipe(self._pipe)
        else:
            work = self._pipe

        # -- sampler (yalnız SDXL; SD3/FLUX flow-match scheduler'ını korur) ---
        sampler = str(params.get("sampler", "") or "")
        if family == "sdxl" and self._default_scheduler is not None:
            if sampler:
                work.scheduler = _make_sampler(sampler, self._default_scheduler.config)
            else:
                work.scheduler = self._default_scheduler

        # -- IP-Adapter (stil/kompozisyon referansı) --------------------------
        ip = params.get("ip_adapter") or None
        ip_image = None
        if ip:
            if family != "sdxl":
                raise RuntimeError("ip_adapter şimdilik yalnız SDXL ailesinde destekleniyor")
            if mode != "txt2img":
                raise RuntimeError("ip_adapter şimdilik yalnız txt2img modunda destekleniyor")
            ip_path = str(ip.get("image", ""))
            if not ip_path:
                raise RuntimeError("ip_adapter 'image' alanı gerekli")
            yield {"event": "progress", "value": 0.0, "message": "IP-Adapter yükleniyor…"}
            work.load_ip_adapter(
                str(ip.get("repo", "h94/IP-Adapter")),
                subfolder=str(ip.get("subfolder", "sdxl_models")),
                weight_name=str(ip.get("weight_name", "ip-adapter_sdxl.safetensors")),
            )
            work.set_ip_adapter_scale(float(ip.get("scale", 0.6)))
            ip_image = Image.open(ip_path).convert("RGB")

        # diffusers callback'i pipe() içinde çalışır; progress'i akıtmak için üretim
        # ayrı thread'de koşar, olaylar kuyruktan yield edilir.
        events: queue.Queue = queue.Queue()
        outcome: dict = {}
        want_preview = bool(params.get("preview", True)) and family == "sdxl"
        preview_every = 1 if steps <= 16 else 2

        def on_step(pipe_obj: Any, step: int, timestep: Any, cb_kwargs: dict) -> dict:
            # strength'li modlarda gerçek adım sayısı steps*strength olur
            total = getattr(pipe_obj, "_num_timesteps", None) or steps
            ev = {
                "event": "progress",
                "value": (step + 1) / max(1, total),
                "message": f"adım {step + 1}/{total}",
            }
            if want_preview and (step % preview_every == 0 or step + 1 == total):
                try:
                    b64 = _latent_preview_jpeg(cb_kwargs.get("latents"))
                    if b64:
                        ev["preview"] = b64
                except Exception:
                    pass  # önizleme asla üretimi düşürmesin
            events.put(ev)
            return cb_kwargs

        # -- çağrı argümanları -------------------------------------------------
        call: dict[str, Any] = {
            "prompt": params.get("prompt", ""),
            "num_inference_steps": steps,
            "generator": generator,
            "callback_on_step_end": on_step,
        }
        if family == "flux":
            # FLUX negative_prompt almaz; schnell için cfg anlamsızdır (vars. düşük)
            call["guidance_scale"] = float(params.get("cfg", 3.5))
            call["max_sequence_length"] = int(params.get("max_sequence_length", 256))
        else:
            call["negative_prompt"] = params.get("negative_prompt", "") or None
            call["guidance_scale"] = float(params.get("cfg", 7.0))
        if mode == "txt2img":
            call["width"] = width
            call["height"] = height
            if control_images:
                call["image"] = control_images if len(control_images) > 1 else control_images[0]
                call["controlnet_conditioning_scale"] = (
                    cn_scales if len(cn_scales) > 1 else cn_scales[0]
                )
        elif mode == "img2img":
            call["image"] = init_img
            call["strength"] = strength
        else:  # inpaint / outpaint — SDXL inpaint boyut verilmezse 1024'e döner, açıkça geçir
            call["image"] = init_img
            call["mask_image"] = mask_img
            call["strength"] = strength
            call["width"] = width
            call["height"] = height
        if ip_image is not None:
            call["ip_adapter_image"] = ip_image

        # Çoklu LoRA (Faz 4): iş başına yüklenir, iş sonunda kaldırılır — base model
        # cache'i temiz kalır, farklı LoRA kombinasyonları birbirini etkilemez.
        loras = params.get("loras") or []

        def run() -> None:
            try:
                if loras:
                    names = []
                    weights = []
                    for i, lo in enumerate(loras):
                        name = f"lora{i}"
                        work.load_lora_weights(lo["path"], adapter_name=name)
                        names.append(name)
                        weights.append(float(lo.get("weight", 1.0)))
                    work.set_adapters(names, weights)
                image = work(**call).images[0]
                meta = PngInfo()
                meta.add_text(
                    "parameters", _a1111_metadata(params, seed, size=(image.width, image.height))
                )
                image.save(output_path, pnginfo=meta)
                outcome["result"] = {
                    "path": output_path,
                    "seed": seed,
                    "width": image.width,
                    "height": image.height,
                    "mode": mode,
                }
            except Exception as exc:  # thread sınırı: hata ana akışa taşınır
                outcome["error"] = str(exc)
            finally:
                if loras:
                    try:
                        work.unload_lora_weights()
                    except Exception:
                        pass
                if ip is not None:
                    try:
                        work.unload_ip_adapter()
                    except Exception:
                        pass
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
        yield {"event": "done", "result": outcome["result"]}

    def unload(self) -> None:
        self._controlnets = {}
        self._default_scheduler = None
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            self._model_path = ""
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def vram_estimate(self, params: dict) -> int:
        return 10_000  # SDXL fp16 ~10 GB (1024²); Faz 14'te profillenecek
