"""Video düzenleme adapter'ı (Faz 8 adım 4): FFmpeg tabanlı kes/birleştir/hız/ses/format
+ kare bazlı büyütme (upscale, imgtool.py'nin spandrel/ESRGAN motorunu yeniden kullanır).

DİKKAT: Bu modül worker env'inde (envs/image ile paylaşılır — scheduler._ENV_ALIAS)
çalışır. 3.11+ sözdizimi KULLANMA.

**RIFE kare interpolasyonu (2026-09-18):** `rife-ncnn-vulkan-python` yerine (PyPI'da yalnız
kaynak dağıtım, Vulkan SDK + C++ zinciri) Practical-RIFE v4.25'in saf-PyTorch IFNet'i vendor
edildi (vendor/rife, MIT) — envs/image'ın torch'uyla çalışır, ek bağımlılık yok.

params:
- op: trim | concat | speed | add_audio | convert | upscale | interpolate (zorunlu)

interpolate (kare hızını artır):
- video_path · model_path (ikisi de zorunlu — models/video/RIFE-4.25/flownet.safetensors)
- multi: 2-8 (vars. 2; çıkış fps = kaynak fps × multi) · scale: 0.25|0.5|1.0|2.0 (vars. 1.0;
  4K için 0.5) · ses izi korunur

trim:
- video_path (zorunlu) · start (sn, vars. 0) · end (sn, opsiyonel — yoksa sona kadar)

concat:
- video_paths: [yol, ...] (zorunlu, en az 2) — hepsi aynı codec/çözünürlükte olmalı

speed:
- video_path (zorunlu) · factor: 0.25-4.0 (vars. 1.0'dan farklı olmalı; >1 hızlandırır)

add_audio:
- video_path · audio_path (ikisi de zorunlu) · mode: replace (vars.) | mix

convert:
- video_path (zorunlu) · format: mp4 (vars.) | mov | avi | gif

upscale:
- video_path · model_path (ikisi de zorunlu — Model Browser'dan bir ESRGAN modeli,
  imgtool.py'nin upscale'iyle aynı model havuzu) · her kare ayrı ayrı büyütülüp
  yeniden kodlanır (ses izi varsa olduğu gibi kopyalanır)

Tüm op'lar: output_path (zorunlu, format ile aynı uzantı)
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator
from typing import Any

from studio_inference.adapters._ffmpeg import ffmpeg_exe, ffprobe_exe, has_audio_stream


def _import_error(hint: str, exc: ImportError) -> RuntimeError:
    return RuntimeError(
        f"videotool worker ortamı eksik ({hint}). "
        "Kurulum reçetesi: docs/worker-envs.md  |  " + str(exc)
    )


def _run_ffmpeg(args: list[str]) -> None:
    cmd = [ffmpeg_exe(), "-y", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg başarısız: {proc.stderr[-2000:]}")


class VideoToolAdapter:
    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}  # "upscale:model_path" -> yüklü spandrel modeli

    def load(self, model_path: str, **kwargs: Any) -> None:
        pass  # model gerektirmez — saf FFmpeg subprocess çağrıları

    def _get_upscaler(self, model_path: str):
        # imgtool.py::_get_upscaler ile birebir aynı desen (spandrel — basicsr gerektirmez).
        key = f"upscale:{model_path}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        try:
            import torch
            from spandrel import ModelLoader
        except ImportError as exc:
            raise _import_error("spandrel bulunamadı", exc) from exc

        model = ModelLoader().load_from_file(model_path)
        model.eval()
        if torch.cuda.is_available():
            model.to("cuda")
        self._cache[key] = model
        return model

    def _get_rife(self, model_path: str):
        """Practical-RIFE v4.25 IFNet (vendor/rife) — Model Browser'daki
        `flownet.safetensors` (ya da upstream'in `flownet.pkl`'i) ile. fp32 tutulur:
        coarse-to-fine akış fp16'ya duyarlı (bkz. vendor/rife/NOTICE.md)."""
        key = f"rife:{model_path}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        import sys

        vendor = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "vendor", "rife"))
        if vendor not in sys.path:
            sys.path.insert(0, vendor)
        try:
            import torch
            from rife import IFNet
        except ImportError as exc:
            raise _import_error("RIFE (torch) yüklenemedi", exc) from exc

        path = model_path
        if os.path.isdir(path):
            for cand in ("flownet.safetensors", "flownet.pkl"):
                if os.path.isfile(os.path.join(path, cand)):
                    path = os.path.join(path, cand)
                    break
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file

            state = load_file(path)
        else:
            state = torch.load(path, map_location="cpu", weights_only=True)
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        net = IFNet()
        missing, unexpected = net.load_state_dict(state, strict=False)
        # upstream `strict=False` yükler: eğitimde kullanılan teacher/caltime blokları
        # çıkarım ağında yok; ama çıkarım bloklarından biri eksikse bu gerçek hatadır
        core_missing = [k for k in missing if k.startswith(("block", "encode"))]
        if core_missing:
            raise RuntimeError(f"RIFE ağırlıkları eksik: {core_missing[:5]} ({len(core_missing)})")
        net.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net.to(device)
        self._cache[key] = net
        return net

    def infer(self, params: dict) -> Iterator[dict]:
        op = params.get("op", "")
        output_path = params.get("output_path", "")
        if not output_path:
            raise RuntimeError("output_path gerekli")

        yield {"event": "progress", "value": 0.1, "message": f"{op} çalışıyor…"}

        if op == "trim":
            video_path = str(params.get("video_path", ""))
            if not video_path:
                raise RuntimeError("video_path gerekli")
            start = float(params.get("start", 0) or 0)
            args = ["-ss", str(start), "-i", video_path]
            end = params.get("end")
            if end is not None:
                duration = max(0.0, float(end) - start)
                args += ["-t", str(duration)]
            args += ["-c", "copy", output_path]
            _run_ffmpeg(args)

        elif op == "concat":
            paths = list(params.get("video_paths") or [])
            if len(paths) < 2:
                raise RuntimeError("concat için en az 2 video_paths gerekli")
            import os
            import tempfile

            with tempfile.NamedTemporaryFile(
                "w", suffix=".txt", delete=False, encoding="utf-8"
            ) as f:
                for p in paths:
                    escaped = str(p).replace("'", "'\\''")
                    f.write(f"file '{escaped}'\n")
                list_path = f.name
            try:
                _run_ffmpeg(
                    ["-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", output_path]
                )
            finally:
                os.unlink(list_path)

        elif op == "speed":
            video_path = str(params.get("video_path", ""))
            if not video_path:
                raise RuntimeError("video_path gerekli")
            factor = float(params.get("factor", 1.0))
            if factor <= 0 or factor == 1.0:
                raise RuntimeError("factor 0'dan büyük ve 1.0'dan farklı olmalı")
            # setpts video hızını ayarlar (ters oran); atempo ses hızını (0.5-2.0
            # aralığı dışı için zincirleme gerekir — şimdilik tek atempo ile sınırlı)
            if not (0.5 <= factor <= 2.0):
                raise RuntimeError("bu ilk sürümde factor 0.5-2.0 aralığıyla sınırlı")
            if has_audio_stream(video_path):
                _run_ffmpeg(
                    [
                        "-i",
                        video_path,
                        "-filter_complex",
                        f"[0:v]setpts={1 / factor}*PTS[v];[0:a]atempo={factor}[a]",
                        "-map",
                        "[v]",
                        "-map",
                        "[a]",
                        output_path,
                    ]
                )
            else:
                # Ses izi yok (ör. LTX-Video çıktısı) — yalnız video hızlandırılır.
                _run_ffmpeg(
                    ["-i", video_path, "-filter:v", f"setpts={1 / factor}*PTS", output_path]
                )

        elif op == "add_audio":
            video_path = str(params.get("video_path", ""))
            audio_path = str(params.get("audio_path", ""))
            if not video_path or not audio_path:
                raise RuntimeError("video_path ve audio_path gerekli")
            mode = str(params.get("mode", "replace") or "replace")
            if mode not in ("replace", "mix"):
                raise RuntimeError(f"bilinmeyen mode: {mode!r} (replace|mix)")
            if mode == "replace":
                _run_ffmpeg(
                    [
                        "-i",
                        video_path,
                        "-i",
                        audio_path,
                        "-map",
                        "0:v",
                        "-map",
                        "1:a",
                        "-c:v",
                        "copy",
                        "-shortest",
                        output_path,
                    ]
                )
            elif has_audio_stream(video_path):
                _run_ffmpeg(
                    [
                        "-i",
                        video_path,
                        "-i",
                        audio_path,
                        "-filter_complex",
                        "[0:a][1:a]amix=inputs=2:duration=shortest[a]",
                        "-map",
                        "0:v",
                        "-map",
                        "[a]",
                        "-c:v",
                        "copy",
                        output_path,
                    ]
                )
            else:
                # Kaynak videoda (ör. LTX-Video çıktısı) ses izi yok — karışacak
                # ikinci bir iz olmadığından replace ile aynı sonuca düşer.
                _run_ffmpeg(
                    [
                        "-i",
                        video_path,
                        "-i",
                        audio_path,
                        "-map",
                        "0:v",
                        "-map",
                        "1:a",
                        "-c:v",
                        "copy",
                        "-shortest",
                        output_path,
                    ]
                )

        elif op == "convert":
            video_path = str(params.get("video_path", ""))
            if not video_path:
                raise RuntimeError("video_path gerekli")
            fmt = str(params.get("format", "mp4") or "mp4")
            if fmt not in ("mp4", "mov", "avi", "gif"):
                raise RuntimeError(f"bilinmeyen format: {fmt!r} (mp4|mov|avi|gif)")
            if fmt == "gif":
                # Basit ama kaliteli GIF: palet üretip iki geçişte uygula.
                _run_ffmpeg(
                    [
                        "-i",
                        video_path,
                        "-vf",
                        "fps=12,scale=480:-1:flags=lanczos,split[s0][s1];"
                        "[s0]palettegen[p];[s1][p]paletteuse",
                        output_path,
                    ]
                )
            else:
                _run_ffmpeg(["-i", video_path, output_path])

        elif op == "upscale":
            video_path = str(params.get("video_path", ""))
            model_path = str(params.get("model_path", ""))
            if not video_path:
                raise RuntimeError("video_path gerekli")
            if not model_path:
                raise RuntimeError(
                    "upscale için model_path gerekli (Model Browser'dan bir ESRGAN modeli seçin)"
                )
            yield {"event": "progress", "value": 0.15, "message": "model yükleniyor…"}
            model = self._get_upscaler(model_path)

            import os
            import tempfile

            import torch
            import torchvision.transforms.functional as TF
            from PIL import Image

            with tempfile.TemporaryDirectory() as tmp:
                _run_ffmpeg(["-i", video_path, f"{tmp}/%06d.png"])
                frame_files = sorted(f for f in os.listdir(tmp) if f.endswith(".png"))
                if not frame_files:
                    raise RuntimeError("videodan kare çıkarılamadı")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                out_dir = os.path.join(tmp, "out")
                os.makedirs(out_dir, exist_ok=True)
                for i, fname in enumerate(frame_files):
                    img = Image.open(os.path.join(tmp, fname)).convert("RGB")
                    inp = TF.to_tensor(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        out_t = model(inp)
                    TF.to_pil_image(out_t.squeeze(0).clamp(0, 1).cpu()).save(
                        os.path.join(out_dir, fname)
                    )
                    yield {
                        "event": "progress",
                        "value": 0.2 + 0.7 * (i + 1) / len(frame_files),
                        "message": f"kare {i + 1}/{len(frame_files)} büyütülüyor…",
                    }

                yield {"event": "progress", "value": 0.95, "message": "yeniden kodlanıyor…"}
                fps_probe = subprocess.run(
                    [
                        ffprobe_exe(),
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=r_frame_rate",
                        "-of",
                        "csv=p=0",
                        video_path,
                    ],
                    capture_output=True,
                    text=True,
                )
                fps_str = fps_probe.stdout.strip() or "24/1"
                encode_args = ["-framerate", fps_str, "-i", f"{out_dir}/%06d.png"]
                if has_audio_stream(video_path):
                    encode_args += [
                        "-i",
                        video_path,
                        "-map",
                        "0:v",
                        "-map",
                        "1:a",
                        "-c:a",
                        "copy",
                    ]
                encode_args += [
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    output_path,
                ]
                _run_ffmpeg(encode_args)

        elif op == "interpolate":
            video_path = str(params.get("video_path", ""))
            model_path = str(params.get("model_path", ""))
            if not video_path:
                raise RuntimeError("video_path gerekli")
            if not model_path:
                raise RuntimeError(
                    "interpolate için model_path gerekli (Model Browser'dan RIFE-4.25 "
                    "flownet.safetensors seçin)"
                )
            multi = int(params.get("multi", 2))
            if multi < 2 or multi > 8:
                raise RuntimeError("multi 2-8 arası olmalı (2 = kare hızını ikiye katla)")
            scale = float(params.get("scale", 1.0))
            if scale not in (0.25, 0.5, 1.0, 2.0):
                raise RuntimeError("scale 0.25|0.5|1.0|2.0 olmalı (4K için 0.5)")

            yield {"event": "progress", "value": 0.12, "message": "RIFE modeli yükleniyor…"}
            net = self._get_rife(model_path)

            import os
            import tempfile

            import torch
            import torch.nn.functional as F
            import torchvision.transforms.functional as TF
            from PIL import Image

            with tempfile.TemporaryDirectory() as tmp:
                _run_ffmpeg(["-i", video_path, f"{tmp}/%06d.png"])
                frame_files = sorted(f for f in os.listdir(tmp) if f.endswith(".png"))
                if len(frame_files) < 2:
                    raise RuntimeError("interpolasyon için en az 2 kare gerekir")
                device = next(net.parameters()).device
                out_dir = os.path.join(tmp, "out")
                os.makedirs(out_dir, exist_ok=True)

                def load(fname: str) -> Any:
                    img = Image.open(os.path.join(tmp, fname)).convert("RGB")
                    return TF.to_tensor(img).unsqueeze(0).to(device)

                first = load(frame_files[0])
                h, w = first.shape[2], first.shape[3]
                # upstream inference_video.py: 128/scale katına doldur (5 kademeli piramit)
                unit = max(128, int(128 / scale))
                ph = ((h - 1) // unit + 1) * unit
                pw = ((w - 1) // unit + 1) * unit
                pad = (0, pw - w, 0, ph - h)
                scale_list = [16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale]

                def save(tensor: Any, idx: int) -> None:
                    TF.to_pil_image(tensor.squeeze(0)[:, :h, :w].clamp(0, 1).cpu()).save(
                        os.path.join(out_dir, f"{idx:06d}.png")
                    )

                out_idx = 1
                prev = F.pad(first, pad)
                save(prev, out_idx)
                out_idx += 1
                total_pairs = len(frame_files) - 1
                with torch.no_grad():
                    for i in range(1, len(frame_files)):
                        nxt = F.pad(load(frame_files[i]), pad)
                        imgs = torch.cat((prev, nxt), 1)
                        for k in range(1, multi):
                            _flow, _mask, merged = net(imgs, k / multi, scale_list)
                            save(merged[-1], out_idx)
                            out_idx += 1
                        save(nxt, out_idx)
                        out_idx += 1
                        prev = nxt
                        yield {
                            "event": "progress",
                            "value": 0.15 + 0.75 * i / total_pairs,
                            "message": f"kare çifti {i}/{total_pairs} → {multi}x",
                        }

                yield {"event": "progress", "value": 0.95, "message": "yeniden kodlanıyor…"}
                fps_probe = subprocess.run(
                    [
                        ffprobe_exe(),
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=r_frame_rate",
                        "-of",
                        "csv=p=0",
                        video_path,
                    ],
                    capture_output=True,
                    text=True,
                )
                num, _, den = (fps_probe.stdout.strip() or "24/1").partition("/")
                src_fps = float(num) / float(den or 1)
                out_fps = src_fps * multi
                encode_args = ["-framerate", f"{out_fps:.4f}", "-i", f"{out_dir}/%06d.png"]
                if has_audio_stream(video_path):
                    encode_args += ["-i", video_path, "-map", "0:v", "-map", "1:a", "-c:a", "copy"]
                encode_args += [
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    output_path,
                ]
                _run_ffmpeg(encode_args)
            yield {
                "event": "done",
                "result": {
                    "path": output_path,
                    "op": op,
                    "multi": multi,
                    "src_fps": src_fps,
                    "fps": out_fps,
                    "frames": out_idx - 1,
                },
            }
            return

        else:
            raise RuntimeError(
                f"bilinmeyen op: {op!r} (trim|concat|speed|add_audio|convert|upscale|interpolate)"
            )

        yield {"event": "done", "result": {"path": output_path, "op": op}}

    def unload(self) -> None:
        self._cache = {}
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def vram_estimate(self, params: dict) -> int:
        # upscale (spandrel/ESRGAN) ve interpolate (RIFE, fp32 IFNet) hariç yalnızca
        # CPU/FFmpeg subprocess kullanır
        return {"upscale": 2000, "interpolate": 3000}.get(str(params.get("op", "")), 0)
