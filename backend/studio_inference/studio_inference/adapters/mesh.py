"""Mesh (3D) adapter'ı (Faz 9): Image-to-3D — kademeli liste (plan.md adım 1).
TripoSR (hafif, hızlı) ilk aile; Stable Fast 3D → InstantMesh → TRELLIS →
Hunyuan3D-2 → OpenLRM/CRM sonraki dilimlerde eklenecek.

DİKKAT: Bu modül worker env'inde (envs/3d) çalışır. torch/tsr/rembg/trimesh
importları metod içindedir; API süreci bu modülü hiç yüklemez.

TripoSR'ın kendi çıkarım kodu (`tsr` paketi) vendor edildi
(`vendor/triposr/tsr`, MIT lisans — bkz. `vendor/triposr/NOTICE.md`).
`torchmcubes` (CUDA marching cubes) yerine `scikit-image`'in CPU
marching_cubes'üne patchlendi: bu makinede NVIDIA CUDA Toolkit (nvcc) kurulu
değil, torchmcubes'un CUDA eklentisi derlenemiyor (bkz. docs/worker-envs.md,
Faz 9). Sinir ağı çıkarımı (VRAM'in çoğunu kullanan kısım) yine GPU'da
kalır; yalnızca son geometri-çıkarım adımı CPU'da çalışır.

params:
- family: triposr (vars., tek aile şimdilik)
- image_path: girdi görsel (zorunlu)
- model_path: Model Browser'dan seçilen ağırlık dosyası ya da HF repo id
  (vars. "stabilityai/TripoSR" — ilk kullanımda ağdan iner)
- remove_background: bool (vars. True) — rembg ile arkaplan temizliği;
  false verilirse girdi zaten gri arkaplanlı/düzgün kırpılmış olmalı
- foreground_ratio: 0-1 (vars. 0.85) — yalnızca remove_background=true iken
- mc_resolution: marching cubes ızgara çözünürlüğü (vars. 256)
- chunk_size: render/yüzey-çıkarım chunk boyutu (vars. 8192, 0=chunksız)
- output_path: .glb veya .obj (zorunlu — uzantıya göre trimesh formatı seçer)
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Iterator
from typing import Any

_VENDOR_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "vendor", "triposr")
)


def _ensure_vendor_on_path() -> None:
    if _VENDOR_DIR not in sys.path:
        sys.path.insert(0, _VENDOR_DIR)


# TripoSR tek-adımlı bir ileri geçiş (diffusion değil, adım sayısı yok) —
# RTX 5090'da canlı ölçüldü: 9.8sn toplam (chair.png, mc_resolution=256:
# ~0.5sn görsel hazırlığı + ~1sn model çıkarımı + ~2sn mesh + ~6sn dışa aktarım).
_ESTIMATE_SECONDS = {"triposr": 10.0}


def estimate_seconds(family: str) -> float:
    return _ESTIMATE_SECONDS.get(family, 10.0)


class MeshAdapter:
    def __init__(self) -> None:
        self._model = None
        self._family = ""
        self._model_path = ""
        self._device = "cpu"

    def load(self, model_path: str, family: str = "triposr", **kwargs: Any) -> None:
        if family != "triposr":
            raise RuntimeError(f"bilinmeyen family: {family!r} (yalnızca 'triposr' var)")
        _ensure_vendor_on_path()
        try:
            import torch
            from tsr.system import TSR
        except ImportError as exc:
            raise RuntimeError(
                "3d worker ortamı eksik (torch/tsr bulunamadı). "
                "Kurulum reçetesi: docs/worker-envs.md  |  " + str(exc)
            ) from exc

        repo = model_path or "stabilityai/TripoSR"
        # Model Browser tekil bir ağırlık dosyası (config.yaml ya da model.ckpt)
        # verebilir — TSR.from_pretrained klasör bekliyor.
        if os.path.isfile(repo):
            repo = os.path.dirname(repo)

        model = TSR.from_pretrained(repo, config_name="config.yaml", weight_name="model.ckpt")
        chunk_size = int(kwargs.get("chunk_size", 8192) or 0)
        model.renderer.set_chunk_size(chunk_size)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

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
        mc_resolution = int(params.get("mc_resolution", 256))
        chunk_size = int(params.get("chunk_size", 8192) or 0)

        if self._model is None or self._family != family or self._model_path != model_path:
            self.load(model_path, family=family, chunk_size=chunk_size)

        import numpy as np
        import torch
        from PIL import Image

        from tsr.utils import remove_background, resize_foreground

        t0 = time.time()

        def _progress(msg: str) -> dict:
            return {"event": "progress", "message": f"[{time.time()-t0:6.1f}s] {msg}"}

        yield _progress("görsel hazırlanıyor…")
        image = Image.open(image_path).convert("RGB")
        if remove_bg:
            import rembg

            image = remove_background(Image.open(image_path), rembg.new_session())
            image = resize_foreground(image, foreground_ratio)
            arr = np.array(image).astype(np.float32) / 255.0
            arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
            image = Image.fromarray((arr * 255.0).astype(np.uint8))

        yield _progress("model çıkarımı çalışıyor…")
        with torch.no_grad():
            scene_codes = self._model([image], device=self._device)

        yield _progress(f"mesh çıkarılıyor (çözünürlük {mc_resolution})…")
        meshes = self._model.extract_mesh(scene_codes, True, resolution=mc_resolution)
        mesh = meshes[0]

        yield _progress("dışa aktarılıyor…")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
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
        # TripoSR küçük bir model (~700MB ağırlık) — kaba, güvenli tahmin.
        return 4_000
