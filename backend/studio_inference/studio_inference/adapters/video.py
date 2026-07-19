"""Video adapter'ı (Faz 8): LTX-Video ile text-to-video / image-to-video.

DİKKAT: Bu modül worker env'inde (envs/image ile paylaşılır — scheduler._ENV_ALIAS)
çalışır. 3.11+ sözdizimi KULLANMA. torch/diffusers importları metod içindedir.

Kademeli model desteğinden (plan.md Faz 8 adım 1: LTX-Video → Wan 2.x → CogVideoX →
Hunyuan Video → Mochi → Pyramid Flow → SkyReels) yalnızca LTX-Video (en hafif,
düşük VRAM, plan'ın belirttiği varsayılan) bu ilk dilimde uygulandı — kalanlar
plan.md'de "Faz 8 kalanları" olarak not düşüldü.

params:
- model_path: LTX-Video diffusers dizini/HF repo id (Model Browser'dan; boşsa
  "Lightricks/LTX-Video" varsayılanı kullanılır — ilk kullanımda ~13 GB indirir)
- mode: t2v (vars.) | i2v
- init_image: kaynak görsel yolu (i2v için zorunlu)
- prompt / negative_prompt · width (vars. 704) · height (vars. 480) ·
  num_frames (vars. 121, 8'in katı+1) · fps (vars. 25) · steps (vars. 30) ·
  cfg (vars. 3.0) · seed · output_path (.mp4)
- vram_mode: max (vars.) | balanced (model offload) | low (sequential+tiling)
"""

from __future__ import annotations

import queue
import random
import threading
from collections.abc import Iterator
from typing import Any

from studio_inference.adapters._ffmpeg import ensure_ffmpeg_on_path


def _align_frames(value: Any, lo: int = 9, hi: int = 257) -> int:
    """LTX kare sayısı 8k+1 biçiminde olmalı (VAE zamansal downsample oranı)."""
    v = max(lo, min(hi, int(value)))
    k = round((v - 1) / 8)
    return max(lo, 8 * k + 1)


def _align(value: Any, multiple: int = 32, lo: int = 256, hi: int = 1280) -> int:
    v = max(lo, min(hi, int(value)))
    return max(lo, (v // multiple) * multiple)


class VideoAdapter:
    def __init__(self) -> None:
        self._pipe = None
        self._model_path = ""
        self._vram_mode = "max"

    def load(self, model_path: str, vram_mode: str = "max", **kwargs: Any) -> None:
        try:
            import torch
            from diffusers import LTXPipeline
        except ImportError as exc:
            raise RuntimeError(
                "video worker ortamı eksik (torch/diffusers bulunamadı). "
                "Kurulum reçetesi: docs/worker-envs.md  |  " + str(exc)
            ) from exc

        use_cuda = torch.cuda.is_available()
        dtype = torch.bfloat16 if use_cuda else torch.float32
        pipe = LTXPipeline.from_pretrained(model_path or "Lightricks/LTX-Video", torch_dtype=dtype)

        if use_cuda:
            if vram_mode == "low":
                pipe.vae.enable_tiling()
                pipe.enable_sequential_cpu_offload()
            elif vram_mode == "balanced":
                pipe.enable_model_cpu_offload()
            else:
                pipe.to("cuda")

        self._pipe = pipe
        self._model_path = str(model_path)
        self._vram_mode = vram_mode

    def infer(self, params: dict) -> Iterator[dict]:
        output_path = params.get("output_path", "")
        if not output_path:
            raise RuntimeError("output_path gerekli")
        mode = str(params.get("mode", "t2v") or "t2v")
        if mode not in ("t2v", "i2v"):
            raise RuntimeError(f"bilinmeyen mode: {mode!r} (t2v|i2v)")
        vram_mode = str(params.get("vram_mode", "max") or "max")
        if vram_mode not in ("max", "balanced", "low"):
            raise RuntimeError(f"bilinmeyen vram_mode: {vram_mode!r} (max|balanced|low)")

        model_path = str(params.get("model_path", "") or "Lightricks/LTX-Video")
        # Kalıcı worker: model komutlar arasında bellekte kalır; yol/VRAM modu değişirse takas.
        if self._pipe is not None and (
            self._model_path != model_path or self._vram_mode != vram_mode
        ):
            self.unload()
        if self._pipe is None:
            yield {
                "event": "progress",
                "value": 0.0,
                "message": "model yükleniyor… (ilk kullanımda ~13 GB indirebilir)",
            }
            self.load(model_path, vram_mode=vram_mode)

        import torch

        ensure_ffmpeg_on_path()

        width = _align(params.get("width", 704))
        height = _align(params.get("height", 480))
        num_frames = _align_frames(params.get("num_frames", 121))
        fps = int(params.get("fps", 25))
        steps = int(params.get("steps", 30))
        cfg = float(params.get("cfg", 3.0))
        seed = int(params.get("seed", -1))
        if seed < 0:
            seed = random.randrange(2**31)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device).manual_seed(seed)

        call: dict[str, Any] = {
            "prompt": params.get("prompt", ""),
            "negative_prompt": params.get("negative_prompt", "") or None,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "frame_rate": fps,
            "num_inference_steps": steps,
            "guidance_scale": cfg,
            "generator": generator,
        }
        if mode == "i2v":
            from diffusers import LTXImageToVideoPipeline
            from PIL import Image

            init_path = str(params.get("init_image", ""))
            if not init_path:
                raise RuntimeError("i2v için init_image gerekli")
            call["image"] = Image.open(init_path).convert("RGB").resize((width, height))
            work = LTXImageToVideoPipeline.from_pipe(self._pipe)
        else:
            work = self._pipe

        events: queue.Queue = queue.Queue()
        outcome: dict = {}

        def on_step(pipe_obj: Any, step: int, timestep: Any, cb_kwargs: dict) -> dict:
            events.put(
                {
                    "event": "progress",
                    "value": (step + 1) / max(1, steps),
                    "message": f"adım {step + 1}/{steps}",
                }
            )
            return cb_kwargs

        call["callback_on_step_end"] = on_step

        def run() -> None:
            try:
                frames = work(**call).frames[0]
                events.put({"event": "progress", "value": 0.97, "message": "video kodlanıyor…"})
                _encode_mp4(frames, output_path, fps)
                outcome["result"] = {
                    "path": output_path,
                    "seed": seed,
                    "width": width,
                    "height": height,
                    "num_frames": len(frames),
                    "fps": fps,
                    "mode": mode,
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
        # LTX-Video bf16 (transformer + T5-XXL metin kodlayıcı) — Faz 14'te profillenecek
        return 14_000


def _encode_mp4(frames: list, output_path: str, fps: int) -> None:
    """PIL kare listesini FFmpeg ile H.264 MP4'e kodlar (diffusers'ın export_to_video'su
    yerine — ek bir imageio-ffmpeg bağımlılığı gerektirmez, Faz 7'nin FFmpeg build'ini
    yeniden kullanır)."""
    import subprocess
    import tempfile

    from studio_inference.adapters._ffmpeg import ffmpeg_exe

    with tempfile.TemporaryDirectory() as tmp:
        for i, frame in enumerate(frames):
            frame.save(f"{tmp}/{i:06d}.png")
        cmd = [
            ffmpeg_exe(),
            "-y",
            "-framerate",
            str(fps),
            "-i",
            f"{tmp}/%06d.png",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            output_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg encode başarısız: {proc.stderr[-2000:]}")
