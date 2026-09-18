"""Video adapter'ı (Faz 8): kademeli T2V/I2V modelleri (plan.md adım 1).

DİKKAT: Bu modül worker env'inde (envs/image ile paylaşılır — scheduler._ENV_ALIAS)
çalışır. 3.11+ sözdizimi KULLANMA. torch/diffusers importları metod içindedir.

Kademeli liste (LTX-Video → Wan 2.x → CogVideoX → Hunyuan Video → Mochi →
Pyramid Flow → SkyReels) — ilk 5'i diffusers'ın kendi pipeline sınıflarıyla
uygulandı (`family` parametresi seçer). Pyramid Flow ve SkyReels diffusers'ın
yerleşik pipeline'ları arasında YOK (özel/topluluk kod gerektiriyor) — bu
dilimde uygulanmadı, plan.md'ye not düşüldü.

**Diffusers pin güncellemesi:** `envs/image`'ın pini 0.32.2 → 0.33.1'e çıkarıldı
(Wan desteği için). `--no-deps` ile kuruldu, torch/transformers'a dokunulmadı;
SDXL/FLUX/SD3/LTX/ControlNet canlıda regresyon testinden geçti (bkz.
docs/worker-envs.md).

params:
- family: ltx (vars.) | wan | cogvideox | hunyuan | mochi
- model_path: Model Browser'da kayıtlı bir bileşen ağırlık dosyası (çok
  bileşenli bir diffusers pipeline'ı olduğundan hangi bileşen seçilirse
  seçilsin model_index.json köküne çözülür) ya da HF repo id; boşsa ailenin
  varsayılan repo'su kullanılır (ilk kullanımda büyük indirme yapabilir —
  bkz. docs/worker-envs.md boyut tablosu)
- mode: t2v (vars.) | i2v (yalnızca ltx/wan/cogvideox'ta var)
- init_image: kaynak görsel yolu (i2v için zorunlu)
- prompt / negative_prompt (hunyuan hariç tüm ailelerde) · width/height ·
  num_frames · fps (yalnızca encode'a etki eder; ltx'te ayrıca generation'a da
  frame_rate olarak geçer) · steps · cfg · seed · output_path (.mp4)
- vram_mode: max (vars.) | balanced (model offload) | low (bileşen bazlı
  device_map='auto' + max_memory ile GPU önce dolar, taşan kısım CPU RAM'e
  gider — tiling de eklenir) — wan/hunyuan/mochi 32 GB'lık bir GPU'da
  genellikle balanced/low ister; model boyutu sistem RAM'ini de aşıyorsa
  (ör. HunyuanVideo/Mochi bu makinede) yalnızca 'low' çalışır çünkü model asla
  CPU'da tam materialize olmaz (bkz. docs/worker-envs.md)
"""

from __future__ import annotations

import os
import queue
import random
import threading
from collections.abc import Iterator
from typing import Any

from studio_inference.adapters._ffmpeg import ensure_ffmpeg_on_path

# aile -> (t2v pipeline sınıfı, i2v pipeline sınıfı ya da None, vars. t2v repo,
#          vars. i2v repo, negative_prompt destekleniyor mu, LTX'in frame_rate'i)
_FAMILIES: dict[str, dict[str, Any]] = {
    "ltx": {
        "t2v_cls": "LTXPipeline",
        "i2v_cls": "LTXImageToVideoPipeline",
        "repo": "Lightricks/LTX-Video",
        "repo_i2v": "Lightricks/LTX-Video",
        "negative_prompt": True,
        "frame_rate_param": True,
        "defaults": {"width": 704, "height": 480, "num_frames": 121, "steps": 30, "cfg": 3.0},
    },
    "wan": {
        "t2v_cls": "WanPipeline",
        "i2v_cls": "WanImageToVideoPipeline",
        "repo": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "repo_i2v": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        "negative_prompt": True,
        "frame_rate_param": False,
        "defaults": {"width": 832, "height": 480, "num_frames": 81, "steps": 50, "cfg": 5.0},
    },
    "cogvideox": {
        "t2v_cls": "CogVideoXPipeline",
        "i2v_cls": "CogVideoXImageToVideoPipeline",
        "repo": "THUDM/CogVideoX-2b",
        "repo_i2v": "THUDM/CogVideoX-5b-I2V",
        "negative_prompt": True,
        "frame_rate_param": False,
        "defaults": {"width": 720, "height": 480, "num_frames": 49, "steps": 50, "cfg": 6.0},
    },
    "hunyuan": {
        "t2v_cls": "HunyuanVideoPipeline",
        "i2v_cls": None,
        "repo": "hunyuanvideo-community/HunyuanVideo",
        "repo_i2v": "",
        "negative_prompt": False,
        "frame_rate_param": False,
        "defaults": {"width": 960, "height": 544, "num_frames": 65, "steps": 30, "cfg": 6.0},
    },
    "mochi": {
        "t2v_cls": "MochiPipeline",
        "i2v_cls": None,
        "repo": "genmo/mochi-1-preview",
        "repo_i2v": "",
        "negative_prompt": True,
        "frame_rate_param": False,
        "defaults": {"width": 848, "height": 480, "num_frames": 19, "steps": 64, "cfg": 4.5},
    },
}


def _resolve_pipeline_root(path_or_repo: str) -> str:
    """Model Browser'da LTX-Video gibi çok-bileşenli bir diffusers pipeline'ının
    HERHANGİ bir bileşen ağırlık dosyası seçilmiş olabilir (ör. .../transformer/
    diffusion_pytorch_model-00001-of-00002.safetensors) — from_pretrained ise
    model_index.json'ın olduğu ÜST kökü ister. En fazla 4 üst klasöre kadar
    model_index.json aranır (Mochi/Hunyuan gibi derin yapılar için); bulunamazsa
    tek-bileşenli desene (config.json kardeşi — ör. ControlNet) düşer, o da
    yoksa yol/HF repo id olduğu gibi kullanılır."""
    p = str(path_or_repo)
    if not os.path.isfile(p):
        return p
    d = os.path.dirname(p)
    for _ in range(4):
        if not d or not os.path.isdir(d):
            break
        if os.path.exists(os.path.join(d, "model_index.json")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    parent = os.path.dirname(p)
    if parent and os.path.exists(os.path.join(parent, "config.json")):
        return parent
    return p


def _align_frames(value: Any, lo: int = 9, hi: int = 257) -> int:
    """Çoğu video VAE'si kare sayısının 8k+1 biçiminde olmasını ister (zamansal
    downsample oranı) — LTX/Wan/CogVideoX/Mochi hepsi bu kalıba uyar."""
    v = max(lo, min(hi, int(value)))
    k = round((v - 1) / 8)
    return max(lo, 8 * k + 1)


def _align(value: Any, multiple: int = 32, lo: int = 256, hi: int = 1280) -> int:
    v = max(lo, min(hi, int(value)))
    return max(lo, (v // multiple) * multiple)


# Aile başına ölçülen/tahmini adım hızı (adım/sn, varsayılan çözünürlükte) — Faz 8
# canlı ölçümlerinden (RTX 5090); tahmini süre göstergesi için kullanılır.
# ltx/cogvideox/wan gerçek canlı ölçüm (bkz. docs/worker-envs.md, küçültülmüş
# çözünürlükte ölçüldü); hunyuan/mochi bu makinede RAM yetersizliğinden canlı
# ölçülemedi, kaba tahmin olarak kaldı.
_STEPS_PER_SEC = {"ltx": 13.0, "wan": 5.0, "cogvideox": 4.5, "hunyuan": 0.5, "mochi": 1.0}


def estimate_seconds(family: str, steps: int) -> float:
    rate = _STEPS_PER_SEC.get(family, _STEPS_PER_SEC["wan"])
    return round(steps / rate, 1)


def recommend_settings(
    family: str, width: int, height: int, num_frames: int, vram_mode: str
) -> tuple[int, int, int, str]:
    """plan.md adım 3 — VRAM'e göre otomatik çözünürlük/frame önerisi: algılanan
    toplam VRAM, ailenin varsayılan ayarlardaki kaba tahmininin altındaysa
    çözünürlük/kare sayısı orantılı küçültülür ve vram_mode en az 'balanced'a
    yükseltilir. 32 GB+ GPU'larda (bu makine gibi) devreye girmez."""
    try:
        import torch

        if not torch.cuda.is_available():
            return width, height, num_frames, vram_mode
        total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    except Exception:
        return width, height, num_frames, vram_mode

    spec = _FAMILIES.get(family)
    if spec is None:
        return width, height, num_frames, vram_mode
    budget_mb = total_mb * 0.85  # sürücü/diğer süreçler için pay
    estimate_by_family = {
        "ltx": 14_000, "wan": 16_000, "cogvideox": 12_000, "hunyuan": 28_000, "mochi": 24_000,
    }
    estimate_mb = estimate_by_family.get(family, 14_000)
    if budget_mb >= estimate_mb:
        return width, height, num_frames, vram_mode

    scale = max(0.5, (budget_mb / estimate_mb) ** 0.5)  # VRAM ~ çözünürlük^2 kabaca
    new_width = _align(width * scale)
    new_height = _align(height * scale)
    new_frames = _align_frames(num_frames * scale)
    new_vram_mode = vram_mode if vram_mode == "low" else "balanced"
    return new_width, new_height, new_frames, new_vram_mode


def _load_pipeline_split(cls: Any, repo: str, dtype: Any, gpu_budget_gb: float, cpu_budget_gb: float) -> Any:
    """Bileşen bazlı yükleme (Faz 8, HunyuanVideo/Mochi RAM taşması sonrası eklendi
    — bkz. docs/worker-envs.md): pipeline'ın kendi from_pretrained'ı tüm modeli
    ÖNCE CPU RAM'e tam olarak yükleyip SONRA offload uyguluyordu
    (enable_sequential_cpu_offload) — model boyutu (HunyuanVideo ~39GB, Mochi
    ~37GB) sistem RAM'ini (bu makinede 15.7GB) aşınca segfault ile çöküyordu.
    Bunun yerine her ağır alt-modeli (transformer/text_encoder/vae) accelerate'in
    device_map='sequential' + max_memory mekanizmasıyla DOĞRUDAN hedef cihaza
    dağıtır: önce GPU'ya kadar dolan (gpu_budget_gb), taşan kısım CPU RAM'e
    (cpu_budget_gb) — model asla CPU'da tam olarak materialize olmaz.
    ('auto'/'balanced' yerine 'sequential' seçildi çünkü ikisi de transformers/
    diffusers'ın get_balanced_memory() adımını tetikliyor — bu fonksiyon
    max_memory'deki "disk" anahtarını torch.device(...) ile ayrıştırmaya
    çalışıp RuntimeError atıyor; 'sequential' bu adımı tamamen atlıyor, ayrıca
    GPU'yu önce doldurup taşanı sırayla CPU'ya vermesi zaten istediğimiz davranış.)
    GPU+CPU payı toplam model boyutunu karşılamazsa kalanı diske (offload_folder)
    taşır — yavaş ama çöküp süreci öldürmez."""
    import json
    import os as _os
    import tempfile

    import diffusers
    import torch

    try:
        import transformers
    except ImportError:
        transformers = None

    with open(_os.path.join(repo, "model_index.json"), encoding="utf-8") as f:
        index = json.load(f)

    max_memory = {0: f"{gpu_budget_gb:.1f}GiB", "cpu": f"{cpu_budget_gb:.1f}GiB", "disk": "200GiB"}
    offload_folder = _os.path.join(tempfile.gettempdir(), "studio_video_offload")
    _os.makedirs(offload_folder, exist_ok=True)
    components: dict[str, Any] = {}
    for name, entry in index.items():
        if name.startswith("_") or not isinstance(entry, list) or len(entry) != 2:
            continue
        library_name, class_name = entry
        if not class_name:
            components[name] = None
            continue
        module = diffusers if library_name == "diffusers" else transformers
        cls_obj = getattr(module, class_name, None) if module is not None else None
        if cls_obj is None:
            continue
        load_kwargs: dict[str, Any] = {"subfolder": name}
        if isinstance(cls_obj, type) and issubclass(cls_obj, torch.nn.Module):
            load_kwargs.update(
                torch_dtype=dtype,
                device_map="sequential",
                max_memory=dict(max_memory),
                low_cpu_mem_usage=True,
                offload_folder=_os.path.join(offload_folder, name),
                offload_state_dict=True,
            )
        components[name] = cls_obj.from_pretrained(repo, **load_kwargs)

    return cls(**components)


def _free_ram_gb() -> float:
    try:
        import psutil

        return psutil.virtual_memory().available / (1024**3)
    except Exception:
        return 4.0  # psutil yoksa ihtiyatlı bir varsayılan


class VideoAdapter:
    def __init__(self) -> None:
        self._pipe = None
        self._family = ""
        self._model_path = ""
        self._vram_mode = "max"

    def load(
        self, model_path: str, family: str = "ltx", vram_mode: str = "max", **kwargs: Any
    ) -> None:
        spec = _FAMILIES.get(family)
        if spec is None:
            raise RuntimeError(f"bilinmeyen family: {family!r} ({', '.join(_FAMILIES)})")
        try:
            import diffusers
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "video worker ortamı eksik (torch/diffusers bulunamadı). "
                "Kurulum reçetesi: docs/worker-envs.md  |  " + str(exc)
            ) from exc

        cls = getattr(diffusers, spec["t2v_cls"])
        use_cuda = torch.cuda.is_available()
        dtype = torch.bfloat16 if use_cuda else torch.float32
        repo = _resolve_pipeline_root(model_path) if model_path else spec["repo"]

        if use_cuda and vram_mode == "low":
            gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_budget_gb = max(4.0, gpu_total_gb * 0.9)
            cpu_budget_gb = max(2.0, _free_ram_gb() - 3.0)  # OS/diğer süreçler için pay
            pipe = _load_pipeline_split(cls, repo, dtype, gpu_budget_gb, cpu_budget_gb)
            try:
                pipe.vae.enable_tiling()
            except Exception:
                pass
        else:
            pipe = cls.from_pretrained(repo, torch_dtype=dtype)
            if use_cuda:
                if vram_mode == "balanced":
                    pipe.enable_model_cpu_offload()
                else:
                    pipe.to("cuda")

        self._pipe = pipe
        self._family = family
        self._model_path = str(model_path)
        self._vram_mode = vram_mode

    def infer(self, params: dict) -> Iterator[dict]:
        output_path = params.get("output_path", "")
        if not output_path:
            raise RuntimeError("output_path gerekli")
        family = str(params.get("family", "ltx") or "ltx")
        spec = _FAMILIES.get(family)
        if spec is None:
            raise RuntimeError(f"bilinmeyen family: {family!r} ({', '.join(_FAMILIES)})")
        mode = str(params.get("mode", "t2v") or "t2v")
        if mode not in ("t2v", "i2v"):
            raise RuntimeError(f"bilinmeyen mode: {mode!r} (t2v|i2v)")
        if mode == "i2v" and spec["i2v_cls"] is None:
            raise RuntimeError(f"{family} bu dilimde yalnızca t2v destekliyor (i2v yok)")
        vram_mode = str(params.get("vram_mode", "max") or "max")
        if vram_mode not in ("max", "balanced", "low"):
            raise RuntimeError(f"bilinmeyen vram_mode: {vram_mode!r} (max|balanced|low)")

        d = spec["defaults"]
        width = _align(params.get("width", d["width"]))
        height = _align(params.get("height", d["height"]))
        num_frames = _align_frames(params.get("num_frames", d["num_frames"]))
        if bool(params.get("auto_resolution", True)):
            width, height, num_frames, vram_mode = recommend_settings(
                family, width, height, num_frames, vram_mode
            )

        model_path = str(params.get("model_path", "") or spec["repo"])
        # Kalıcı worker: model komutlar arasında bellekte kalır; aile/yol/VRAM modu
        # değişirse takas.
        if self._pipe is not None and (
            self._family != family or self._model_path != model_path or self._vram_mode != vram_mode
        ):
            self.unload()
        if self._pipe is None:
            eta = estimate_seconds(family, int(params.get("steps", d["steps"])))
            yield {
                "event": "progress",
                "value": 0.0,
                "message": f"{family} modeli yükleniyor… (ilk kullanımda büyük indirme olabilir, "
                f"tahmini üretim süresi ~{eta:.0f}sn)",
            }
            self.load(model_path, family=family, vram_mode=vram_mode)

        import torch

        ensure_ffmpeg_on_path()

        fps = int(params.get("fps", 24))
        steps = int(params.get("steps", d["steps"]))
        cfg = float(params.get("cfg", d["cfg"]))
        seed = int(params.get("seed", -1))
        if seed < 0:
            seed = random.randrange(2**31)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device).manual_seed(seed)

        call: dict[str, Any] = {
            "prompt": params.get("prompt", ""),
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "num_inference_steps": steps,
            "guidance_scale": cfg,
            "generator": generator,
            "output_type": "pil",
        }
        if spec["negative_prompt"]:
            call["negative_prompt"] = params.get("negative_prompt", "") or None
        if spec["frame_rate_param"]:
            call["frame_rate"] = fps

        if mode == "i2v":
            import diffusers
            from PIL import Image

            init_path = str(params.get("init_image", ""))
            if not init_path:
                raise RuntimeError("i2v için init_image gerekli")
            call["image"] = Image.open(init_path).convert("RGB").resize((width, height))
            i2v_cls = getattr(diffusers, spec["i2v_cls"])
            work = i2v_cls.from_pipe(self._pipe)
        else:
            work = self._pipe

        events: queue.Queue = queue.Queue()
        outcome: dict = {}
        want_preview = bool(params.get("preview", True))
        # ~6 önizleme/üretim, çok sık pahalı VAE decode'a girmesin
        preview_every = max(1, steps // 6)

        def on_step(pipe_obj: Any, step: int, timestep: Any, cb_kwargs: dict) -> dict:
            ev: dict[str, Any] = {
                "event": "progress",
                "value": (step + 1) / max(1, steps),
                "message": f"adım {step + 1}/{steps}",
            }
            if want_preview and (step % preview_every == 0 or step + 1 == steps):
                b64 = _latent_preview_jpeg(
                    pipe_obj, cb_kwargs.get("latents"), family, num_frames, width, height
                )
                if b64:
                    ev["preview"] = b64
            events.put(ev)
            return cb_kwargs

        call["callback_on_step_end"] = on_step

        def run() -> None:
            try:
                frames = work(**call).frames[0]
                events.put({"event": "progress", "value": 0.97, "message": "video kodlanıyor…"})
                _encode_mp4(frames, output_path, fps)
                outcome["result"] = {
                    "path": output_path,
                    "family": family,
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
            self._family = ""
            self._model_path = ""
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def vram_estimate(self, params: dict) -> int:
        # Kaba tahminler (bf16 transformer + T5/UMT5 metin kodlayıcı) — Faz 14'te profillenecek
        return {
            "ltx": 14_000,
            "wan": 16_000,
            "cogvideox": 12_000,
            "hunyuan": 28_000,
            "mochi": 24_000,
        }.get(str(params.get("family", "ltx")), 14_000)


def _latent_preview_jpeg(
    pipe_obj: Any, latents: Any, family: str, num_frames: int, width: int, height: int
) -> str | None:
    """plan.md adım 2 — ara kare önizlemesi: mevcut latent'in İLK karesini
    pipeline'ın kendi VAE'siyle gerçekten decode edip küçük bir JPEG'e çevirir.
    5 aile de farklı bir video VAE/latent düzeni kullanıyor:
    - LTX transformer'ı latentleri düzleştirip (patchify) [B, S, D] dizisine
      paketler — VAE'ye vermeden önce pipeline'ın kendi `_unpack_latents` +
      `_denormalize_latents` yardımcılarıyla [B, C, F, H, W]'ye geri çevrilmesi
      GEREKİR (canlı testte yakalandı: aksi halde `ValueError: not enough
      values to unpack`).
    - Wan/CogVideoX/Hunyuan/Mochi latentleri zaten 5D [B, C, F, H, W] tutar
      (DiT tarzı transformer, ayrı bir paketleme yok) — bunlarda doğrudan
      VAE decode denenir.
    Bu decode ucuz değildir (SDXL'in doğrusal yaklaşımının aksine) — bu yüzden
    yalnızca birkaç adımda bir çağrılır (bkz. infer()'daki preview_every).
    Herhangi bir aile/şekil uyuşmazlığında sessizce None döner — önizleme asla
    üretimi düşürmesin."""
    try:
        import torch

        if latents is None:
            return None
        vae = pipe_obj.vae
        with torch.no_grad():
            lat = latents
            if family == "ltx" and lat.ndim == 3:
                lf = (num_frames - 1) // pipe_obj.vae_temporal_compression_ratio + 1
                lh = height // pipe_obj.vae_spatial_compression_ratio
                lw = width // pipe_obj.vae_spatial_compression_ratio
                lat = type(pipe_obj)._unpack_latents(
                    lat, lf, lh, lw,
                    pipe_obj.transformer_spatial_patch_size,
                    pipe_obj.transformer_temporal_patch_size,
                )
                lat = type(pipe_obj)._denormalize_latents(
                    lat, vae.latents_mean, vae.latents_std, vae.config.scaling_factor
                )
                lat = lat[:, :, :1]  # yalnızca ilk kare
            elif lat.ndim == 5:  # [B, C, T, H, W] — yalnızca ilk kareyi al
                scaling = getattr(vae.config, "scaling_factor", 1.0) or 1.0
                lat = lat[:, :, :1] / scaling
            else:
                return None
            decoded = vae.decode(lat.to(vae.dtype)).sample
            if decoded.ndim == 5:
                decoded = decoded[:, :, 0]
            frame = decoded[0]

        import base64
        import io

        from PIL import Image

        arr = ((frame.float() + 1.0) / 2.0).clamp(0, 1).mul(255).byte()
        arr = arr.permute(1, 2, 0).cpu().numpy()
        img = Image.fromarray(arr)
        img.thumbnail((192, 192))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=65)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


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
