"""SDXL Diffusers adapter'ı (Faz 4). Yalnızca worker sürecinde import edilir.

DİKKAT: Bu modül worker env'inde (Python 3.10 + sistem torch 2.11/cu128) çalışır —
3.11+ sözdizimi KULLANMA. torch/diffusers importları metod içindedir; API süreci
bu modülü hiç yüklemez.

params:
- model_path: .safetensors dosyası veya HF repo dizini (zorunlu)
- prompt / negative_prompt · width/height (varsayılan 1024) · steps (30) · cfg (7.0)
- seed: verilmezse rastgele; sonuçta gerçek seed döner (determinizm: aynı seed → aynı görsel)
- output_path: PNG hedefi (route doldurur)
"""

from __future__ import annotations

import queue
import random
import threading
from collections.abc import Iterator
from typing import Any


def _a1111_metadata(params: dict, seed: int) -> str:
    """Üretim metadata'sını A1111 uyumlu 'parameters' metnine dönüştürür (Faz 4 planı)."""
    neg = params.get("negative_prompt", "")
    lines = [params.get("prompt", "")]
    if neg:
        lines.append("Negative prompt: " + neg)
    lines.append(
        "Steps: {steps}, CFG scale: {cfg}, Seed: {seed}, Size: {w}x{h}, Model: {model}".format(
            steps=params.get("steps", 30),
            cfg=params.get("cfg", 7.0),
            seed=seed,
            w=params.get("width", 1024),
            h=params.get("height", 1024),
            model=str(params.get("model_path", "")).replace("\\", "/").rsplit("/", 1)[-1],
        )
    )
    return "\n".join(lines)


class SdxlAdapter:
    def __init__(self) -> None:
        self._pipe = None
        self._model_path = ""

    def load(self, model_path: str, **kwargs: Any) -> None:
        try:
            import torch
            from diffusers import StableDiffusionXLPipeline
        except ImportError as exc:
            raise RuntimeError(
                "image worker ortamı eksik (torch/diffusers bulunamadı). "
                "Kurulum reçetesi: docs/worker-envs.md  |  " + str(exc)
            ) from exc

        use_cuda = torch.cuda.is_available()
        dtype = torch.float16 if use_cuda else torch.float32
        if str(model_path).endswith(".safetensors"):
            pipe = StableDiffusionXLPipeline.from_single_file(model_path, torch_dtype=dtype)
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(model_path, torch_dtype=dtype)
        if use_cuda:
            pipe.to("cuda")
        self._pipe = pipe
        self._model_path = str(model_path)

    def infer(self, params: dict) -> Iterator[dict]:
        model_path = params.get("model_path", "")
        if not model_path:
            raise RuntimeError("model_path gerekli (Model Browser'dan bir checkpoint seçin)")
        output_path = params.get("output_path", "")
        if not output_path:
            raise RuntimeError("output_path gerekli")

        # Kalıcı worker: model komutlar arasında bellekte kalır; yol değişirse takas edilir.
        if self._pipe is not None and self._model_path != str(model_path):
            self.unload()
        if self._pipe is None:
            yield {"event": "progress", "value": 0.0, "message": "model yükleniyor…"}
            self.load(model_path)

        import torch
        from PIL.PngImagePlugin import PngInfo

        steps = int(params.get("steps", 30))
        seed = int(params.get("seed", -1))
        if seed < 0:
            seed = random.randrange(2**31)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device).manual_seed(seed)

        # diffusers callback'i pipe() içinde çalışır; progress'i akıtmak için üretim
        # ayrı thread'de koşar, olaylar kuyruktan yield edilir.
        events: queue.Queue[dict | None] = queue.Queue()
        outcome: dict = {}

        def on_step(pipe: Any, step: int, timestep: Any, cb_kwargs: dict) -> dict:
            events.put(
                {
                    "event": "progress",
                    "value": (step + 1) / steps,
                    "message": f"adım {step + 1}/{steps}",
                }
            )
            return cb_kwargs

        def run() -> None:
            try:
                image = self._pipe(
                    prompt=params.get("prompt", ""),
                    negative_prompt=params.get("negative_prompt", "") or None,
                    width=int(params.get("width", 1024)),
                    height=int(params.get("height", 1024)),
                    num_inference_steps=steps,
                    guidance_scale=float(params.get("cfg", 7.0)),
                    generator=generator,
                    callback_on_step_end=on_step,
                ).images[0]
                meta = PngInfo()
                meta.add_text("parameters", _a1111_metadata(params, seed))
                image.save(output_path, pnginfo=meta)
                outcome["result"] = {
                    "path": output_path,
                    "seed": seed,
                    "width": image.width,
                    "height": image.height,
                }
            except Exception as exc:  # thread sınırı: hata ana akışa taşınır
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
        yield {"event": "done", "result": outcome["result"]}

    def unload(self) -> None:
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
