"""Mesh (3D) adapter'ı (Faz 9): Image-to-3D — kademeli liste (plan.md adım 1).
TripoSR (hafif, hızlı), Stable Fast 3D (SF3D, dokulu/tekstürlü çıktı) ve
OpenLRM (vertex-renkli, EG3D tarzı saf-PyTorch NeRF renderer) desteklenen
üç aile. InstantMesh/TRELLIS/CRM eklenMEDİ — üçü de mesh çıkarımı için
`nvdiffrast`'ın `RasterizeCudaContext`'ini zorunlu kılıyor, bu da (torchmcubes/
texture_baker'ın aksine) ÇALIŞMA ZAMANINDA JIT CUDA derlemesi gerektiriyor —
bu makinede NVIDIA CUDA Toolkit kurulu değil, bu üçü için de tek gerçek
çözüm CUDA Toolkit kurmak (büyük, sistem geneli bir değişiklik — kullanıcı
onayı olmadan yapılmadı). Ayrıntılar docs/worker-envs.md'de.
Hunyuan3D-2 (shape+paint) henüz eklenmedi.

DİKKAT: Bu modül worker env'inde (envs/3d) çalışır. torch/tsr/sf3d/openlrm/
rembg/trimesh importları metod içindedir; API süreci bu modülü hiç yüklemez.

Üç ailenin de kendi çıkarım kodu vendor edildi (`vendor/triposr`,
`vendor/stablefast3d`, `vendor/openlrm`) — bkz. ilgili NOTICE.md dosyaları.

**LİSANS UYARISI:** stable-fast-3d → Stability AI Community License
(araştırma/kişisel serbest, yıllık geliri 1M USD'yi aşan kuruluşlar için
Enterprise License gerekir — XTTS-v2 ile aynı desen, bkz. audio.py).
openlrm → CC-BY-NC-4.0 (yalnızca araştırma/kişisel, TİCARİ KULLANIM YOK).
TripoSR MIT.

params:
- family: triposr (vars.) | sf3d | openlrm
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

openlrm'e özgü:
- mesh_size: marching cubes ızgara çözünürlüğü (vars. 384)
- mesh_thres: isosurface eşiği (vars. 3.0)
- source_cam_dist: kaynak kamera uzaklığı (vars. 2.0)

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
    "openlrm": os.path.join(_VENDOR_BASE, "openlrm"),
}
_FAMILY_DEFAULT_REPO = {
    "triposr": "stabilityai/TripoSR",
    "sf3d": "stabilityai/stable-fast-3d",
    "openlrm": "zxhezexin/openlrm-mix-base-1.1",
}

# Tek-adımlı bir ileri geçiş (diffusion değil, adım sayısı yok) — RTX 5090'da
# canlı ölçüldü: triposr 9.8sn (chair.png, mc_resolution=256), sf3d 14.7sn
# (chair.png, texture_resolution=1024 — dokulu çıktı, remesh dahil), openlrm
# 7.3sn (portre fotoğrafı, mesh_size=192 — ilk kullanımda ayrıca ~330MB'lık
# bir DINOv2 kodlayıcı iner, bu tahmine dahil değil).
_ESTIMATE_SECONDS = {"triposr": 10.0, "sf3d": 15.0, "openlrm": 8.0}


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


def _remove_background(image, rembg_session):
    """TripoSR/SF3D/OpenLRM'in üçünde de aynı algoritma (Stability AI'nin
    ortak soyundan geliyor) — tek, aileden bağımsız bir kopya burada tutulup
    çapraz-aile vendor import'undan kaçınılıyor (openlrm yüklüyken sf3d'nin
    vendor dizini sys.path'te olmayabilir)."""
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        return image  # zaten gerçek bir alfa maskesi var
    import rembg

    return rembg.remove(image, session=rembg_session)


def _resize_foreground(image, ratio: float):
    import numpy as np
    from PIL import Image

    arr = np.array(image)
    alpha = np.where(arr[..., 3] > 0)
    y1, y2, x1, x2 = alpha[0].min(), alpha[0].max(), alpha[1].min(), alpha[1].max()
    fg = arr[y1:y2, x1:x2]
    size = max(fg.shape[0], fg.shape[1])
    ph0, pw0 = (size - fg.shape[0]) // 2, (size - fg.shape[1]) // 2
    ph1, pw1 = size - fg.shape[0] - ph0, size - fg.shape[1] - pw0
    new_image = np.pad(fg, ((ph0, ph1), (pw0, pw1), (0, 0)), mode="constant")
    new_size = int(new_image.shape[0] / ratio)
    ph0, pw0 = (new_size - size) // 2, (new_size - size) // 2
    ph1, pw1 = new_size - size - ph0, new_size - size - pw0
    new_image = np.pad(new_image, ((ph0, ph1), (pw0, pw1), (0, 0)), mode="constant")
    return Image.fromarray(new_image)


def _build_openlrm_inferrer(repo: str):
    """LRMInferrer, kendi kurucusunda ortam değişkenlerinden (APP_MODEL_NAME/
    APP_INFER) okuyan bir CLI/Hydra-tarzı yapılandırma kullanıyor
    (openlrm/runners/infer/lrm.py::parse_configs) — gradio app.py'nin kendisi
    de aynı desenle çağırıyor, biz de programatik olarak taklit ediyoruz.
    repo adında/yolunda 'small'/'large' geçiyorsa ilgili config, yoksa
    varsayılan olarak 'base' config'i kullanılır (indirilen tek model
    openlrm-mix-base-1.1)."""
    name_lower = repo.lower()
    size = "s" if "small" in name_lower else "l" if "large" in name_lower else "b"
    config_path = os.path.join(_FAMILY_VENDOR_DIR["openlrm"], "configs", f"infer-{size}.yaml")

    os.environ["APP_ENABLED"] = "1"
    os.environ["APP_MODEL_NAME"] = repo
    os.environ["APP_INFER"] = config_path

    from openlrm.runners.infer.lrm import LRMInferrer

    return LRMInferrer()


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
        elif family == "openlrm":
            model = _build_openlrm_inferrer(repo)
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
        image = Image.open(image_path).convert("RGBA")
        if remove_bg:
            import rembg

            image = _remove_background(image, rembg.new_session())
            image = _resize_foreground(image, foreground_ratio)
        if family == "triposr":
            # TripoSR gri arkaplana compose edilmiş düz RGB bekliyor.
            arr = np.array(image).astype(np.float32) / 255.0
            arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
            image = Image.fromarray((arr * 255.0).astype(np.uint8))
        # sf3d: prepare_image() ham RGBA bekliyor (alfa kanalını kendi işliyor).
        # openlrm: infer_single() dosyadan okuyup kendi RGBA->RGB compose'unu
        # (beyaz arkaplan) kendi yapıyor — ikisi de RGBA olarak kalıyor.

        if family == "triposr":
            mc_resolution = int(params.get("mc_resolution", 256))
            yield _progress("model çıkarımı çalışıyor…")
            import torch

            with torch.no_grad():
                scene_codes = self._model([image], device=self._device)
            yield _progress(f"mesh çıkarılıyor (çözünürlük {mc_resolution})…")
            meshes = self._model.extract_mesh(scene_codes, True, resolution=mc_resolution)
            mesh = meshes[0]
        elif family == "sf3d":
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
        else:  # openlrm
            import tempfile

            import trimesh

            mesh_size = int(params.get("mesh_size", 384))
            mesh_thres = float(params.get("mesh_thres", 3.0))
            source_cam_dist = params.get("source_cam_dist")
            yield _progress("model çıkarımı çalışıyor…")
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_image_path = tmp.name
            try:
                image.save(tmp_image_path)
                self._model.cfg.mesh_size = mesh_size
                self._model.cfg.mesh_thres = mesh_thres
                # infer_mesh() dosyaya export ediyor ama mesh nesnesini geri
                # döndürmüyor — dışa aktarımdan sonra vertex/yüz sayısı için
                # trimesh ile tekrar okunuyor (aşağıda).
                self._model.infer_single(
                    tmp_image_path,
                    source_cam_dist=float(source_cam_dist) if source_cam_dist else None,
                    export_video=False,
                    export_mesh=True,
                    dump_video_path="",
                    dump_mesh_path=output_path,
                )
            finally:
                os.unlink(tmp_image_path)
            yield _progress("dışa aktarılıyor…")
            mesh = trimesh.load(output_path, force="mesh")
            result = {
                "path": output_path,
                "family": family,
                "vertices": int(mesh.vertices.shape[0]),
                "faces": int(mesh.faces.shape[0]),
            }
            yield {"event": "done", "result": result}
            return

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
        # Üçü de küçük modeller (~700MB-2GB ağırlık) — kaba, güvenli tahmin.
        if family == "sf3d":
            return 6_000
        return 4_000


def _nullcontext():
    from contextlib import nullcontext

    return nullcontext()
