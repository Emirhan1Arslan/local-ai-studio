"""Mesh (3D) adapter'ı (Faz 9): Image-to-3D — kademeli liste (plan.md adım 1).
TripoSR (hafif, hızlı) ve Stable Fast 3D (SF3D, dokulu/tekstürlü çıktı)
şimdilik desteklenen iki aile; InstantMesh/TRELLIS/Hunyuan3D-2/OpenLRM/CRM
sonraki dilimlerde eklenecek (ağırlıkları zaten `models/3d/` altında,
docs/worker-envs.md'de durum notu var).

DİKKAT: Bu modül worker env'inde (envs/3d) çalışır. torch/tsr/sf3d/rembg/
trimesh importları metod içindedir; API süreci bu modülü hiç yüklemez.

Her iki ailenin de kendi çıkarım kodu (`tsr`/`sf3d` paketleri) vendor edildi
(`vendor/triposr`, `vendor/stablefast3d`) — bkz. ilgili NOTICE.md dosyaları.

**LİSANS UYARISI (yalnızca stable-fast-3d):** Stability AI Community
License — araştırma/kişisel kullanım serbest, yıllık geliri 1M USD'yi aşan
kuruluşlar için ayrı Enterprise License gerekir (XTTS-v2 ile aynı uyarı
deseni, bkz. audio.py). TripoSR MIT.

params:
- family: triposr (vars.) | sf3d
- image_path: girdi görsel (zorunlu)
- model_path: Model Browser'dan seçilen ağırlık dosyası ya da HF repo id
  (boşsa ailenin varsayılan HF repo'su kullanılır — ilk kullanımda ağdan iner)
- remove_background: bool (vars. True) — rembg ile arkaplan temizliği
- foreground_ratio: 0-1 (vars. 0.85) — yalnızca remove_background=true iken

triposr'a özgü:
- mc_resolution: marching cubes ızgara çözünürlüğü (vars. 256)
- chunk_size: render/yüzey-çıkarım chunk boyutu (vars. 8192, 0=chunksız)

sf3d'ye özgü:
- texture_resolution: doku atlası çözünürlüğü (vars. 1024)
- remesh: none (vars.) | triangle | quad
- target_vertex_count: hedef vertex sayısı (vars. -1 = küçültme yok)

output_path: .glb/.obj/.stl/.ply (zorunlu — uzantıya göre trimesh formatı seçer)
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Iterator
from typing import Any

_VENDOR_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "vendor"))
_FAMILY_VENDOR_DIR = {
    "triposr": os.path.join(_VENDOR_BASE, "triposr"),
    "sf3d": os.path.join(_VENDOR_BASE, "stablefast3d"),
}
_FAMILY_DEFAULT_REPO = {
    "triposr": "stabilityai/TripoSR",
    "sf3d": "stabilityai/stable-fast-3d",
}

# Tek-adımlı bir ileri geçiş (diffusion değil, adım sayısı yok) — RTX 5090'da
# canlı ölçüldü: triposr 9.8sn (chair.png, mc_resolution=256), sf3d 14.7sn
# (chair.png, texture_resolution=1024 — dokulu çıktı, remesh dahil).
_ESTIMATE_SECONDS = {"triposr": 10.0, "sf3d": 15.0}


def estimate_seconds(family: str) -> float:
    return _ESTIMATE_SECONDS.get(family, 10.0)


def _ensure_vendor_on_path(family: str) -> None:
    vendor = _FAMILY_VENDOR_DIR.get(family)
    if vendor and vendor not in sys.path:
        sys.path.insert(0, vendor)


_texture_baker_patched = False


def _patch_texture_baker_for_cpu_kernel() -> None:
    """`texture_baker` uzantısı bu makinede CPU çekirdeği olarak derlendi (NVIDIA
    CUDA Toolkit yok, bkz. docs/worker-envs.md Faz 9 — torchmcubes'la aynı sebep)
    ama SF3D modeli GPU'da çalışıyor, `rasterize`/`interpolate`'e CUDA tensörleri
    geliyor -> "Could not run 'texture_baker_cpp::rasterize' ... CUDA backend"
    hatası. `TextureBaker`'ı, girdileri CPU'ya taşıyıp sonucu orijinal cihaza
    geri döndürecek şekilde bir kez monkeypatch'liyoruz — SF3D'nin geri kalanı
    (sinir ağı çıkarımı, VRAM'in çoğunu kullanan kısım) yine GPU'da kalır;
    yalnızca bu rasterizasyon adımı CPU'da çalışır."""
    global _texture_baker_patched
    if _texture_baker_patched:
        return
    from texture_baker.baker import TextureBaker

    orig_rasterize = TextureBaker.rasterize
    orig_interpolate = TextureBaker.interpolate

    def rasterize(self, uv, face_indices, bake_resolution):
        device = uv.device
        rast = orig_rasterize(self, uv.cpu(), face_indices.cpu(), bake_resolution)
        return rast.to(device)

    def interpolate(self, attr, rast, face_indices):
        device = attr.device
        out = orig_interpolate(self, attr.cpu(), rast.cpu(), face_indices.cpu())
        return out.to(device)

    TextureBaker.rasterize = rasterize
    TextureBaker.interpolate = interpolate
    _texture_baker_patched = True


def _resolve_repo(model_path: str, family: str) -> str:
    repo = model_path or _FAMILY_DEFAULT_REPO[family]
    # Model Browser tekil bir ağırlık dosyası verebilir (config.yaml/.ckpt/
    # .safetensors) — from_pretrained klasör bekliyor.
    if os.path.isfile(repo):
        repo = os.path.dirname(repo)
    return repo


class MeshAdapter:
    def __init__(self) -> None:
        self._model = None
        self._family = ""
        self._model_path = ""
        self._device = "cpu"

    def load(self, model_path: str, family: str = "triposr", **kwargs: Any) -> None:
        if family not in _FAMILY_VENDOR_DIR:
            raise RuntimeError(
                f"bilinmeyen family: {family!r} ({', '.join(_FAMILY_VENDOR_DIR)})"
            )
        _ensure_vendor_on_path(family)
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "3d worker ortamı eksik (torch bulunamadı). "
                "Kurulum reçetesi: docs/worker-envs.md  |  " + str(exc)
            ) from exc

        repo = _resolve_repo(model_path, family)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if family == "triposr":
            from tsr.system import TSR

            model = TSR.from_pretrained(repo, config_name="config.yaml", weight_name="model.ckpt")
            chunk_size = int(kwargs.get("chunk_size", 8192) or 0)
            model.renderer.set_chunk_size(chunk_size)
            model.to(device)
        elif family == "sf3d":
            from sf3d.system import SF3D

            _patch_texture_baker_for_cpu_kernel()
            model = SF3D.from_pretrained(
                repo, config_name="config.yaml", weight_name="model.safetensors"
            )
            model.to(device)
            model.eval()
        else:
            raise AssertionError("unreachable")

        self._model = model
        self._family = family
        self._model_path = str(model_path)
        self._device = device

    def infer(self, params: dict) -> Iterator[dict]:
        output_path = params.get("output_path", "")
        if not output_path:
            raise RuntimeError("output_path gerekli")
        image_path = params.get("image_path", "")
        if not image_path:
            raise RuntimeError("image_path gerekli")
        family = str(params.get("family", "triposr") or "triposr")
        model_path = str(params.get("model_path", "") or "")
        remove_bg = bool(params.get("remove_background", True))
        foreground_ratio = float(params.get("foreground_ratio", 0.85))

        if self._model is None or self._family != family or self._model_path != model_path:
            self.load(model_path, family=family, chunk_size=params.get("chunk_size", 8192))

        import numpy as np
        from PIL import Image

        t0 = time.time()

        def _progress(msg: str) -> dict:
            return {"event": "progress", "message": f"[{time.time()-t0:6.1f}s] {msg}"}

        yield _progress("görsel hazırlanıyor…")
        # sf3d her zaman RGBA bekliyor (remove_bg=False ise alfa=255, tüm
        # görsel önplan sayılır); triposr düz RGB kabul ediyor.
        image = Image.open(image_path).convert("RGBA" if family == "sf3d" else "RGB")
        if remove_bg:
            import rembg

            if family == "triposr":
                from tsr.utils import remove_background, resize_foreground
            else:
                from sf3d.utils import remove_background, resize_foreground

            image = remove_background(Image.open(image_path).convert("RGBA"), rembg.new_session())
            image = resize_foreground(image, foreground_ratio)
            if family == "triposr":
                # TripoSR gri arkaplana compose edilmiş düz RGB bekliyor.
                arr = np.array(image).astype(np.float32) / 255.0
                arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
                image = Image.fromarray((arr * 255.0).astype(np.uint8))
            # sf3d: prepare_image() ham RGBA bekliyor (alfa kanalını kendi
            # işliyor) — compose ETMİYORUZ, image RGBA olarak kalıyor.

        if family == "triposr":
            mc_resolution = int(params.get("mc_resolution", 256))
            yield _progress("model çıkarımı çalışıyor…")
            import torch

            with torch.no_grad():
                scene_codes = self._model([image], device=self._device)
            yield _progress(f"mesh çıkarılıyor (çözünürlük {mc_resolution})…")
            meshes = self._model.extract_mesh(scene_codes, True, resolution=mc_resolution)
            mesh = meshes[0]
        else:  # sf3d
            import torch

            texture_resolution = int(params.get("texture_resolution", 1024))
            remesh = str(params.get("remesh", "none") or "none")
            target_vertex_count = int(params.get("target_vertex_count", -1))
            yield _progress("model çıkarımı + doku pişirme çalışıyor…")
            with torch.no_grad():
                autocast = (
                    torch.autocast(device_type=self._device, dtype=torch.bfloat16)
                    if self._device == "cuda"
                    else _nullcontext()
                )
                with autocast:
                    mesh, _glob = self._model.run_image(
                        image,
                        bake_resolution=texture_resolution,
                        remesh=remesh,
                        vertex_count=target_vertex_count,
                    )

        yield _progress("dışa aktarılıyor…")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        if family == "sf3d":
            mesh.export(output_path, include_normals=True)
        else:
            mesh.export(output_path)

        result = {
            "path": output_path,
            "family": family,
            "vertices": int(mesh.vertices.shape[0]),
            "faces": int(mesh.faces.shape[0]),
        }
        yield {"event": "done", "result": result}

    def unload(self) -> None:
        self._model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def vram_estimate(self, params: dict) -> int:
        family = str(params.get("family", "triposr") or "triposr")
        # Her ikisi de küçük modeller (~700MB-2GB ağırlık) — kaba, güvenli tahmin.
        return 6_000 if family == "sf3d" else 4_000


def _nullcontext():
    from contextlib import nullcontext

    return nullcontext()
