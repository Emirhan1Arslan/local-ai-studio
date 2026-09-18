"""Mesh (3D) adapter'ı (Faz 9): Image-to-3D — kademeli liste (plan.md adım 1).
TripoSR (hafif, hızlı), Stable Fast 3D (SF3D, dokulu/tekstürlü çıktı),
OpenLRM (vertex-renkli, EG3D tarzı saf-PyTorch NeRF renderer),
Hunyuan3D-2 (shape + opsiyonel Paint dokusu — bkz. aşağı), InstantMesh (LRM+FlexiCubes,
zero123plus çoklu-görünüm difüzyonu üzerine), TRELLIS (sparse-voxel
yapı + structured latent, gerçek dokulu çıktı) ve CRM (Convolutional
Reconstruction Model, iki aşamalı çoklu-görünüm difüzyonu + nvdiffrast
UV pişirme) desteklenen yedi aile — kademeli listenin TAMAMI eklendi.

NVIDIA CUDA Toolkit 12.8 bu makineye kuruldu (2026-07-20, kullanıcı elle
çalıştırdı, Display.Driver bileşeni HARİÇ — sürücü 576.88'de değişmeden
kaldı). Bu, önceki "nvdiffrast/CUDA Toolkit eksik" engelini KALDIRDI —
`nvdiffrast` ve `custom_rasterizer_kernel` artık bu makinede gerçekten
derleniyor ve çalışıyor (canlı doğrulandı). InstantMesh/TRELLIS/CRM bu
sayede eklendi.

CRM'in kendi `imagedream` (vendored, diffusers'a bağımsız ldm-tarzı)
UNet'i `BasicTransformerBlock.__init__`'te KOŞULSUZ `assert XFORMERS_
IS_AVAILBLE` içeriyordu — TRELLIS'te bulunan aynı xformers/sm_120
uyumsuzluğu (bkz. aşağı). q/k/v zaten `[B*heads, N, dim_head]` biçiminde
olduğu için TRELLIS'in paketlenmiş-diziyi elle doldurma gerektiren
düzeltmesinden DAHA BASİT bir sdpa köprüsü yeterliydi — sdpa öndeki
boyutu zaten toplu (batch) boyutu olarak ele alıyor. Ayrıca `envs/3d`'ye
kurulan `kiui` paketinin KENDİSİNDE gerçek bir bug bulundu (`kiui/
typing.py` boştu, `Union`/`Literal` gibi adları hiç export etmiyordu —
`kiui/op.py` bunlara güveniyordu) — yerel site-packages kopyası
düzeltildi (vendor kodu değil). Ayrıntılar vendor/crm/NOTICE.md'de.

TRELLIS'in `flash_attn`/`xformers` gereksinimi bu makinede İKİSİ de
ÇALIŞMIYOR (flash_attn'in resmi Windows wheel'i yok; xformers'ın derlenmiş
çekirdekleri RTX 5090/sm_120'yi henüz desteklemiyor — "too new" hatası) —
TRELLIS'in kendi dense-attention modülünde zaten var olan 'sdpa' seçeneğini
sparse (voxel) attention modülüne de PORTLADIK (aynı matematik, farklı
çekirdek — model ağırlığı/davranışıyla ilgisi yok, yalnızca hangi PyTorch
fonksiyonunun çağrıldığıyla ilgili). `kaolin` de gerekli DEĞİL — TRELLIS
paketinin kendisi hiç kullanmıyor, yalnızca vendored FlexiCubes alt
modülünün bir şekil-doğrulama yardımcısı (`check_tensor`) için import
ediliyordu, yerel bir eşdeğerle değiştirildi. Ayrıntılar
vendor/trellis/NOTICE.md'de.

Hunyuan3D-2: shape (`hy3dgen.shapegen`) + opsiyonel Paint dokusu
(`texture=True`, `hy3dgen.texgen`). Paint uzun süre "upstream checkpoint/kod
uyumsuzluğu" diye engelli sayıldı: `hunyuan3d-paint-v2-0-turbo` UNet
checkpoint'i, `modules.py`'nin oluşturmadığı 36 IP-Adapter anahtarı
(`unet.image_proj_model.*`, `*.attn2.to_{k,v}_ip.weight`) taşıyor ve upstream
`strict=True` ile yüklüyor. 2026-09-18'de yeniden incelendi: yayınlanan
çıkarım kodu bu ağırlıkları HİÇ ÇAĞIRMIYOR — `HunyuanPaintPipeline.__init__`
`image_encoder`'ı kayıt bile etmiyor, `Multiview_Diffusion_Net.__call__`
`ip_adapter_image` geçmiyor, dolayısıyla `added_cond_kwargs=None` ve
`to_k_ip/to_v_ip` yolu ölü kod. Yani bu anahtarları ayıklamak eğitilmiş bir
koşullamayı "sessizce düşürmek" değil, upstream'in kendi çıkarım yoluyla
birebir aynı ağı kurmaktır. `_load_hunyuan3d_paint()` UNet'i `strict=False`
yükler ama DENETİMLİ: eksik anahtar sıfır olmalı ve beklenmeyen anahtarların
TAMAMI bu bilinen IP-Adapter desenine uymalı — aksi hâlde hata (bkz.
`classify_paint_unexpected_keys`). Ayrıntılar vendor/hunyuan3d/NOTICE.md'de.

DİKKAT: Bu modül worker env'inde (envs/3d) çalışır. torch/tsr/sf3d/openlrm/
hy3dgen/src(instantmesh)/trellis/crm/rembg/trimesh importları metod
içindedir; API süreci bu modülü hiç yüklemez.

Yedi ailenin de kendi çıkarım kodu vendor edildi (`vendor/triposr`,
`vendor/stablefast3d`, `vendor/openlrm`, `vendor/hunyuan3d`,
`vendor/instantmesh`, `vendor/trellis`, `vendor/crm`) — bkz. ilgili
NOTICE.md dosyaları.

**LİSANS UYARISI:** stable-fast-3d → Stability AI Community License
(araştırma/kişisel serbest, yıllık geliri 1M USD'yi aşan kuruluşlar için
Enterprise License gerekir — XTTS-v2 ile aynı desen, bkz. audio.py).
openlrm → CC-BY-NC-4.0 (yalnızca araştırma/kişisel, TİCARİ KULLANIM YOK).
hunyuan3d → Tencent Hunyuan 3D 2.0 Community License (AB/UK/Güney Kore
hariç her yerde serbest; aylık 1M aktif kullanıcıyı aşan ürünler için
Tencent'ten ayrı lisans gerekir). trellis'in doku pişirme bileşimi
(diff-gaussian-rasterization) Inria/MPII NC lisansı (TİCARİ KULLANIM YOK).
TripoSR/instantmesh/trellis(çekirdek)/crm MIT/Apache-2.0 (özel uyarı
gerekmez).

params:
- family: triposr (vars.) | sf3d | openlrm | hunyuan3d | instantmesh | trellis | crm
- image_path: girdi görsel (zorunlu)
- model_path: Model Browser'dan seçilen ağırlık dosyası ya da HF repo id
  (boşsa ailenin varsayılan HF repo'su kullanılır — ilk kullanımda ağdan iner)
- remove_background: bool (vars. True) — rembg ile arkaplan temizliği
  (trellis kendi ön işlemesini kendi yapar, bkz. aşağı — bu bayrağı okumaz)
- foreground_ratio: 0-1 (vars. 0.85) — yalnızca remove_background=true iken
  (trellis DIŞINDAKİ aileler için)

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

hunyuan3d'ye özgü:
- num_inference_steps: diffusion adım sayısı (vars. 30)
- octree_resolution: hacim kod çözme ızgara çözünürlüğü (vars. 256 —
  upstream varsayılanı 384'tür ama ~584K vertex üretiyor, 256 daha hafif)
- texture: bool (vars. False) — True ise Hunyuan3D-Paint (delight +
  6 görünümlü difüzyon + UV pişirme) ile 2048² dokulu mesh üretir. Model
  Manager kökünde (`.../Hunyuan3D-2/`) `hunyuan3d-delight-v2-0` ve
  `hunyuan3d-paint-v2-0-turbo` klasörleri gerekir; yalnız CUDA.
- texture_max_faces: doku öncesi yüz indirgeme hedefi (vars. 40000 —
  upstream demo akışı: FloaterRemover + DegenerateFaceRemover + FaceReducer)

instantmesh'e özgü:
- num_inference_steps: zero123plus çoklu-görünüm difüzyon adım sayısı
  (vars. 50 — upstream CLI varsayılanı 75, UI için düşürüldü)
- view: 4 | 6 (vars. 6) — zero123plus'ın ürettiği görünüm sayısı
- scale: obje ölçeği (vars. 1.0)
- export_texmap: bool (vars. False) — True ise xatlas+nvdiffrast ile gerçek
  UV dokusu üretir (yavaş); False ise vertex-renkli mesh (hızlı, upstream
  varsayılanı)

trellis'e özgü:
- num_inference_steps: hem sparse-structure hem structured-latent
  örnekleyicisi için adım sayısı (vars. 25, upstream varsayılanı)
- seed: rastgelelik tohumu (vars. 42)
- simplify: mesh basitleştirmede kaldırılacak yüz oranı (vars. 0.95 —
  upstream varsayılanı; ham çıktı çok yoğun)
- texture_size: doku atlası çözünürlüğü (vars. 1024)

crm'e özgü:
- num_inference_steps: iki aşamalı (pixel + CCM/XYZ) çoklu-görünüm
  difüzyonunun HER İKİSİ için de adım sayısı (vars. 30 — upstream
  varsayılanı 50, UI için düşürüldü)
- scale: classifier-free guidance ölçeği (vars. 5.0, upstream varsayılanı)
- remove_background: True ise rembg ile arkaplan kaldırılır; False ise
  upstream'in "Alpha as mask" modu (yalnızca girdi görselin alfa kanalı
  maske olarak kullanılır, ayrıca kaldırma yapılmaz)

output_path: .glb/.obj/.stl/.ply (zorunlu — uzantıya göre trimesh formatı seçer)
"""

from __future__ import annotations

import os
import re
import sys
import time
from collections.abc import Iterator
from typing import Any

_VENDOR_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "vendor"))
_FAMILY_VENDOR_DIR = {
    "triposr": os.path.join(_VENDOR_BASE, "triposr"),
    "sf3d": os.path.join(_VENDOR_BASE, "stablefast3d"),
    "openlrm": os.path.join(_VENDOR_BASE, "openlrm"),
    "hunyuan3d": os.path.join(_VENDOR_BASE, "hunyuan3d"),
    "instantmesh": os.path.join(_VENDOR_BASE, "instantmesh"),
    "trellis": os.path.join(_VENDOR_BASE, "trellis"),
    "crm": os.path.join(_VENDOR_BASE, "crm"),
}
_FAMILY_DEFAULT_REPO = {
    "triposr": "stabilityai/TripoSR",
    "sf3d": "stabilityai/stable-fast-3d",
    "openlrm": "zxhezexin/openlrm-mix-base-1.1",
    "hunyuan3d": "tencent/Hunyuan3D-2",
    "instantmesh": "TencentARC/InstantMesh",
    "trellis": "microsoft/TRELLIS-image-large",
    "crm": "Zhengyi/CRM",
}

# Tek-adımlı bir ileri geçiş (diffusion değil, adım sayısı yok) — RTX 5090'da
# canlı ölçüldü: triposr 9.8sn (chair.png, mc_resolution=256), sf3d 14.7sn
# (chair.png, texture_resolution=1024 — dokulu çıktı, remesh dahil), openlrm
# 7.3sn (portre fotoğrafı, mesh_size=192 — ilk kullanımda ayrıca ~330MB'lık
# bir DINOv2 kodlayıcı iner, bu tahmine dahil değil). hunyuan3d 9.0sn
# (num_inference_steps=30, octree_resolution=256 varsayılanlarıyla; 20 adım/
# 256 çözünürlükte canlı ölçüm 7.8sn sürdü — diffusion örneklemesi ~1.9sn,
# hacim kod çözme ~4.7sn, adım sayısından bağımsız kısım baskın).
_ESTIMATE_SECONDS = {
    "triposr": 10.0,
    "sf3d": 15.0,
    "openlrm": 8.0,
    "hunyuan3d": 9.0,
    # canlı ölçüldü (yüklü model, adaLN/dino ilk-kullanım indirmesi hariç):
    # num_inference_steps=30, export_texmap=False ile 7.5sn; export_texmap=
    # True ile (nvdiffrast+xatlas UV pişirme dahil) 29.2sn. UI varsayımı
    # export_texmap=False + num_inference_steps=50 (upstream CLI'nin 75'i
    # yerine) — diffusion adım sayısıyla ~doğrusal ölçeklenen kısma göre
    # ekstrapole edildi.
    "instantmesh": 12.0,
    # canlı ölçüldü (yüklü model): num_inference_steps=25, formats=[mesh,
    # gaussian], texture_size=1024 ile 27.8sn — çoğu süre 100 görünümlük
    # gaussian render + doku pişirmede (to_glb), yalnızca sparse-structure/
    # slat örneklemesi ~7sn.
    "trellis": 28.0,
    # canlı ölçüldü (yüklü model): num_inference_steps=30, scale=5.0 ile
    # çoklu-görünüm difüzyonu (iki aşama BİRLİKTE) 11sn, mesh çıkarımı +
    # nvdiffrast UV pişirme 29sn.
    "crm": 40.0,
}


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


def _load_hunyuan3d_pipeline(repo: str, device: str):
    """`smart_load_model()`'ın kendi önbellek-yolu kuralı
    (`~/.cache/hy3dgen/<repo>/<subfolder>`) burada bilerek atlanıyor — Model
    Manager'ın gerçek indirme yolu (`.../Hunyuan3D-2/hunyuan3d-dit-v2-0-fast/`)
    zaten `config.yaml` + `model.fp16.safetensors` içeriyor, doğrudan
    `from_single_file` ile okunuyor. safetensors'taki tek checkpoint `model.*`,
    `vae.*` VE `conditioner.*` ağırlıklarının hepsini birlikte barındırıyor
    (bkz. vendor/hunyuan3d/NOTICE.md) — ayrıca indirilen `hunyuan3d-vae-v2-0`
    klasörü bu yüzden kullanılmıyor. `repo` yerel bir klasör değilse (model_path
    boş bırakılıp varsayılan HF repo id'si kullanıldıysa) upstream'in kendi
    `from_pretrained`/`smart_load_model` akışına düşülüyor."""
    import torch
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    dtype = torch.float16 if device == "cuda" else torch.float32
    config_path = os.path.join(repo, "config.yaml")
    ckpt_path = os.path.join(repo, "model.fp16.safetensors")
    if not os.path.isfile(ckpt_path):
        ckpt_path = os.path.join(repo, "model.safetensors")
    if os.path.isfile(config_path) and os.path.isfile(ckpt_path):
        return Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
            ckpt_path, config_path, device=device, dtype=dtype, use_safetensors=True
        )
    return Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        repo,
        subfolder="hunyuan3d-dit-v2-0-fast",
        device=device,
        dtype=dtype,
        use_safetensors=True,
    )


def _load_instantmesh(repo: str, device: str) -> dict:
    """`repo`, Model Manager'daki `instant_mesh_base.ckpt`'in klasörü
    (`_resolve_repo` dosya->dirname çeviriyor) — aynı klasörde upstream'in
    "white-background" ince ayarlı UNet'i (`diffusion_pytorch_model.bin`) de
    bekleniyor. zero123plus'ın kendi temel ağırlıkları (~4.3GB, ayrı bir HF
    repo'su — `sudo-ai/zero123plus-v1.2`) `models/3d/` altında KARDEŞ bir
    klasörde aranıyor (`InstantMesh/`'in yanında `zero123plus-v1.2/`);
    yoksa upstream'in `from_pretrained("sudo-ai/zero123plus-v1.2")` HF
    indirme akışına düşülüyor. `custom_pipeline`, upstream'in bare-string
    `"zero123plus"` deseni (diffusers'ın HF community pipelines'tan ağdan
    çektiği) yerine vendor edilmiş yerel klasöre işaret ediyor — diğer
    ailelerle aynı "yerel kod, ağ çağrısı yok" kuralı."""
    import torch
    from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

    vendor_dir = _FAMILY_VENDOR_DIR["instantmesh"]
    dtype = torch.float16 if device == "cuda" else torch.float32

    zero123plus_dir = os.path.join(os.path.dirname(repo), "zero123plus-v1.2")
    zero123plus_repo = (
        zero123plus_dir if os.path.isdir(zero123plus_dir) else "sudo-ai/zero123plus-v1.2"
    )
    pipeline = DiffusionPipeline.from_pretrained(
        zero123plus_repo,
        custom_pipeline=os.path.join(vendor_dir, "zero123plus"),
        torch_dtype=dtype,
    )
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config, timestep_spacing="trailing"
    )
    unet_ckpt_path = os.path.join(repo, "diffusion_pytorch_model.bin")
    unet_state_dict = torch.load(unet_ckpt_path, map_location="cpu")
    pipeline.unet.load_state_dict(unet_state_dict, strict=True)
    pipeline = pipeline.to(device)

    from omegaconf import OmegaConf
    from src.utils.train_util import instantiate_from_config

    config = OmegaConf.load(os.path.join(vendor_dir, "configs", "instant-mesh-base.yaml"))
    recon_model = instantiate_from_config(config.model_config)
    model_ckpt_path = os.path.join(repo, "instant_mesh_base.ckpt")
    model_state_dict = torch.load(model_ckpt_path, map_location="cpu")["state_dict"]
    model_state_dict = {
        k[14:]: v for k, v in model_state_dict.items() if k.startswith("lrm_generator.")
    }
    recon_model.load_state_dict(model_state_dict, strict=True)
    recon_model = recon_model.to(device)
    recon_model.init_flexicubes_geometry(device, fovy=30.0)
    recon_model = recon_model.eval()

    return {"pipeline": pipeline, "model": recon_model, "infer_config": config.infer_config}


def _load_trellis(repo: str, device: str):
    """`repo`, Model Manager'daki bir ckpt dosyasının klasörüdür
    (`_resolve_repo` dosya->dirname çeviriyor) — TRELLIS'in 8 bileşen
    checkpoint'i `<TRELLIS-image-large>/ckpts/*.safetensors` altında, ama
    `pipeline.json` (from_pretrained'in aradığı) bir üst klasörde. Model
    Manager her checkpoint'i ayrı bir dosya olarak kaydettiği için
    `_resolve_repo` bizi `ckpts/`'e getirir — burada bir üst seviyeye
    çıkıyoruz.

    ATTN_BACKEND/SPCONV_ALGO env değişkenleri `trellis` paketinin İLK
    import'undan ÖNCE ayarlanmalı (modül seviyesinde okunuyorlar, bkz.
    vendor/trellis/trellis/modules/sparse/__init__.py) — flash_attn'in
    Windows wheel'i yok, xformers'ın derlenmiş çekirdekleri RTX 5090/sm_120'yi
    henüz desteklemiyor, bu yüzden 'sdpa' zorunlu (yerel patch, bkz.
    vendor/trellis/NOTICE.md)."""
    if os.path.basename(os.path.normpath(repo)) == "ckpts":
        repo = os.path.dirname(os.path.normpath(repo))

    os.environ.setdefault("SPCONV_ALGO", "native")
    os.environ["ATTN_BACKEND"] = "sdpa"

    from trellis.pipelines import TrellisImageTo3DPipeline

    pipeline = TrellisImageTo3DPipeline.from_pretrained(repo)
    if device == "cuda":
        pipeline.cuda()
    return pipeline


def _load_crm(repo: str, device: str) -> dict:
    """`repo`, Model Manager'daki `CRM.pth`'in klasörüdür (`_resolve_repo`
    dosya->dirname çeviriyor) — `pixel-diffusion.pth`/`ccm-diffusion.pth`
    aynı klasörde KARDEŞ dosyalar olarak bekleniyor. `vendor/crm`'in kendi
    `configs/specs_objaverse_total.json` + iki `.yaml` konfigürasyonu model
    mimarisini/örnekleyiciyi tanımlıyor (ağırlık indirmiyor)."""
    import json

    import torch
    from omegaconf import OmegaConf

    vendor_dir = _FAMILY_VENDOR_DIR["crm"]
    from model import CRM

    specs = json.load(open(os.path.join(vendor_dir, "configs", "specs_objaverse_total.json")))
    model = CRM(specs).to(device)
    model.load_state_dict(
        torch.load(os.path.join(repo, "CRM.pth"), map_location=device), strict=False
    )

    from pipelines import TwoStagePipeline

    stage1_config = OmegaConf.load(
        os.path.join(vendor_dir, "configs", "nf7_v3_SNR_rd_size_stroke.yaml")
    ).config
    stage2_config = OmegaConf.load(os.path.join(vendor_dir, "configs", "stage2-v2-snr.yaml")).config
    stage1_config.models.resume = os.path.join(repo, "pixel-diffusion.pth")
    stage2_config.models.resume = os.path.join(repo, "ccm-diffusion.pth")

    # stage{1,2}_config.models.config (nf7_v3_SNR_rd_size_stroke.yaml /
    # stage2-v2-snr.yaml içinde tanımlı) upstream'de "imagedream/configs/
    # ...yaml" gibi vendor-köküne-göreli bir yol — TwoStagePipeline bunu
    # OmegaConf.load ile CWD'ye göre çözüyor, worker'ın gerçek cwd'si
    # vendor dizini olmadığı için scope'lu bir chdir gerekiyor.
    prev_cwd = os.getcwd()
    try:
        os.chdir(vendor_dir)
        pipeline = TwoStagePipeline(
            stage1_config.models,
            stage2_config.models,
            stage1_config.sampler,
            stage2_config.sampler,
            device=device,
        )
    finally:
        os.chdir(prev_cwd)
    return {"model": model, "pipeline": pipeline}


_PAINT_SUBFOLDER = "hunyuan3d-paint-v2-0-turbo"
_DELIGHT_SUBFOLDER = "hunyuan3d-delight-v2-0"
# Yayınlanan paint checkpoint'inde bulunan ama çıkarım kodunun hiç kullanmadığı
# IP-Adapter anahtarları (modül docstring'i) — strict=False yüklemede yalnız
# bunlara göz yumulur.
_PAINT_IP_ADAPTER_KEY_RE = re.compile(r"(^|\.)image_proj_model\.|\.attn2\.to_[kv]_ip\.weight$")


def classify_paint_unexpected_keys(unexpected: list[str]) -> list[str]:
    """Bilinen IP-Adapter deseni DIŞINDA kalan beklenmeyen anahtarları döner
    (boş liste = yükleme güvenli)."""
    return [k for k in unexpected if not _PAINT_IP_ADAPTER_KEY_RE.search(k)]


def paint_root_for(repo: str) -> str | None:
    """Shape modelinin klasöründen (`.../Hunyuan3D-2/hunyuan3d-dit-v2-0-fast`)
    Paint bileşenlerini barındıran Model Manager kökünü çözer; delight ve paint
    klasörlerinden biri yoksa None."""
    for root in (repo, os.path.dirname(repo.rstrip("/\\"))):
        if os.path.isdir(os.path.join(root, _DELIGHT_SUBFOLDER)) and os.path.isdir(
            os.path.join(root, _PAINT_SUBFOLDER)
        ):
            return root
    return None


def _load_hunyuan3d_paint(root: str, device: str):
    """Hunyuan3D-Paint'i, UNet'i kendimiz denetimli yükleyerek kurar (modül
    docstring'indeki IP-Adapter anahtarı açıklaması). Delight (InstructPix2Pix)
    ve pipeline'ın kalan bileşenleri upstream akışıyla yüklenir."""
    import json

    import torch
    from diffusers import UNet2DConditionModel
    from hy3dgen.texgen.hunyuanpaint.unet.modules import UNet2p5DConditionModel
    from hy3dgen.texgen.pipelines import Hunyuan3DPaintPipeline, Hunyuan3DTexGenConfig

    if device != "cuda":
        raise RuntimeError("Hunyuan3D-Paint dokusu için CUDA gerekli (custom_rasterizer)")

    unet_dir = os.path.join(root, _PAINT_SUBFOLDER, "unet")
    with open(os.path.join(unet_dir, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    unet = UNet2p5DConditionModel(UNet2DConditionModel(**config))

    st_path = os.path.join(unet_dir, "diffusion_pytorch_model.safetensors")
    if os.path.isfile(st_path):
        from safetensors.torch import load_file

        state = load_file(st_path)
    else:
        state = torch.load(
            os.path.join(unet_dir, "diffusion_pytorch_model.bin"),
            map_location="cpu",
            weights_only=True,
        )
    missing, unexpected = unet.load_state_dict(state, strict=False)
    bad = classify_paint_unexpected_keys(list(unexpected))
    if missing or bad:
        raise RuntimeError(
            "Hunyuan3D-Paint UNet checkpoint'i beklenen mimariyle uyuşmuyor — "
            f"eksik: {list(missing)[:5]} ({len(missing)}), "
            f"bilinmeyen fazla: {bad[:5]} ({len(bad)})"
        )
    del state
    unet = unet.to(torch.float16)

    tex_config = Hunyuan3DTexGenConfig(
        os.path.join(root, _DELIGHT_SUBFOLDER),
        os.path.join(root, _PAINT_SUBFOLDER),
        _PAINT_SUBFOLDER,
    )
    tex_config.device = device
    tex_config.unet = unet  # vendor multiview_utils.py yaması bunu okur
    return Hunyuan3DPaintPipeline(tex_config)


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
        self._paint = None
        self._paint_root = ""

    def load(self, model_path: str, family: str = "triposr", **kwargs: Any) -> None:
        if family not in _FAMILY_VENDOR_DIR:
            raise RuntimeError(f"bilinmeyen family: {family!r} ({', '.join(_FAMILY_VENDOR_DIR)})")
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
        elif family == "hunyuan3d":
            model = _load_hunyuan3d_pipeline(repo, device)
        elif family == "instantmesh":
            model = _load_instantmesh(repo, device)
        elif family == "trellis":
            model = _load_trellis(repo, device)
        elif family == "crm":
            model = _load_crm(repo, device)
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
            return {"event": "progress", "message": f"[{time.time() - t0:6.1f}s] {msg}"}

        yield _progress("görsel hazırlanıyor…")
        image = Image.open(image_path).convert("RGBA")
        if remove_bg and family not in ("trellis", "crm"):
            import rembg

            image = _remove_background(image, rembg.new_session())
            image = _resize_foreground(image, foreground_ratio)
        # trellis: kendi preprocess_image()'ı rembg + DINOv2 için gereken
        # özel 518x518 bbox-crop'u BİRLİKTE yapıyor — paylaşılan resize_
        # foreground'la karıştırmak bu formatı bozar, bkz. aşağıdaki
        # pipeline.run(..., preprocess_image=True) çağrısı.
        # crm: kendi ön işlemesi de kendine özgü (kare genişletme + sabit
        # gri arkaplan doldurma, RGB'ye çeviriliyor) — aşağıdaki crm dalında
        # elle yapılıyor.
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
        elif family == "openlrm":
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
        elif family == "hunyuan3d":
            num_inference_steps = int(params.get("num_inference_steps", 30))
            octree_resolution = int(params.get("octree_resolution", 256))
            yield _progress(
                f"diffusion örneklemesi ({num_inference_steps} adım) + hacim kod çözme…"
            )
            mesh = self._model(
                image=image,
                num_inference_steps=num_inference_steps,
                octree_resolution=octree_resolution,
            )[0]
            if bool(params.get("texture", False)):
                from hy3dgen.shapegen import DegenerateFaceRemover, FaceReducer, FloaterRemover

                root = paint_root_for(_resolve_repo(model_path, family))
                if root is None:
                    raise RuntimeError(
                        "texture=true için Model Manager'daki Hunyuan3D-2 kökünde "
                        f"'{_DELIGHT_SUBFOLDER}' ve '{_PAINT_SUBFOLDER}' klasörleri gerekli — "
                        "Model Browser'dan tencent/Hunyuan3D-2 snapshot'ını bu alt "
                        "klasörlerle indirin"
                    )
                if self._paint is None or self._paint_root != root:
                    self._paint = None
                    yield _progress("doku modeli yükleniyor (delight + çoklu-görünüm UNet)…")
                    self._paint = _load_hunyuan3d_paint(root, self._device)
                    self._paint_root = root
                max_faces = int(params.get("texture_max_faces", 40000))
                yield _progress(f"doku öncesi mesh temizliği + {max_faces} yüze indirgeme…")
                mesh = FloaterRemover()(mesh)
                mesh = DegenerateFaceRemover()(mesh)
                mesh = FaceReducer()(mesh, max_facenum=max_faces)
                yield _progress("doku üretimi: delight + 6 görünüm difüzyonu + UV pişirme…")
                mesh = self._paint(mesh, image)
        elif family == "instantmesh":
            import numpy as np
            import torch
            import trimesh
            from einops import rearrange
            from src.utils.camera_util import get_zero123plus_input_cameras
            from torchvision.transforms import v2

            num_inference_steps = int(params.get("num_inference_steps", 50))
            view = int(params.get("view", 6))
            scale = float(params.get("scale", 1.0))
            export_texmap = bool(params.get("export_texmap", False))
            pipeline = self._model["pipeline"]
            recon_model = self._model["model"]
            infer_config = self._model["infer_config"]

            yield _progress(f"zero123plus çoklu-görünüm difüzyonu ({num_inference_steps} adım)…")
            mv_image = pipeline(image, num_inference_steps=num_inference_steps).images[0]

            arr = np.asarray(mv_image, dtype=np.float32) / 255.0
            imgs = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float()
            imgs = rearrange(imgs, "c (n h) (m w) -> (n m) c h w", n=3, m=2)
            imgs = imgs.unsqueeze(0).to(self._device)
            imgs = v2.functional.resize(imgs, 320, interpolation=3, antialias=True).clamp(0, 1)

            input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0 * scale).to(
                self._device
            )
            if view == 4:
                indices = torch.tensor([0, 2, 4, 5]).long().to(self._device)
                imgs = imgs[:, indices]
                input_cameras = input_cameras[:, indices]

            yield _progress("LRM triplane çıkarımı çalışıyor…")
            with torch.no_grad():
                planes = recon_model.forward_planes(imgs, input_cameras)
                yield _progress("mesh çıkarılıyor (FlexiCubes)…")
                mesh_out = recon_model.extract_mesh(
                    planes, use_texture_map=export_texmap, **infer_config
                )

            if export_texmap:
                import tempfile

                from src.utils.mesh_util import save_obj_with_mtl

                vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
                tmp_dir = tempfile.mkdtemp()
                tmp_obj = os.path.join(tmp_dir, "mesh.obj")
                save_obj_with_mtl(
                    vertices.data.cpu().numpy(),
                    uvs.data.cpu().numpy(),
                    faces.data.cpu().numpy(),
                    mesh_tex_idx.data.cpu().numpy(),
                    tex_map.permute(1, 2, 0).data.cpu().numpy(),
                    tmp_obj,
                )
                mesh = trimesh.load(tmp_obj, force="mesh")
            else:
                vertices, faces, vertex_colors = mesh_out
                # save_glb'nin (src/utils/mesh_util.py) eksen dönüşümüyle aynı —
                # upstream'in kendi koordinat düzeltmesi, GLB/diğer formatlar
                # için de tutarlı.
                vertices = vertices @ np.array(
                    [[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32
                )
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors)
        elif family == "trellis":
            from trellis.utils import postprocessing_utils

            num_inference_steps = int(params.get("num_inference_steps", 25))
            seed = int(params.get("seed", 42))
            simplify = float(params.get("simplify", 0.95))
            texture_size = int(params.get("texture_size", 1024))

            yield _progress(
                f"sparse yapı + structured latent örneklemesi ({num_inference_steps} adım)…"
            )
            outputs = self._model.run(
                image,
                seed=seed,
                sparse_structure_sampler_params={"steps": num_inference_steps},
                slat_sampler_params={"steps": num_inference_steps},
                formats=["mesh", "gaussian"],
                preprocess_image=True,
            )
            yield _progress("mesh dışa aktarılıyor (gaussian'dan doku pişirme)…")
            mesh = postprocessing_utils.to_glb(
                outputs["gaussian"][0],
                outputs["mesh"][0],
                simplify=simplify,
                texture_size=texture_size,
                verbose=False,
            )
        elif family == "crm":
            import trimesh
            from inference import generate3d

            num_inference_steps = int(params.get("num_inference_steps", 30))
            scale = float(params.get("scale", 5.0))

            # CRM'in kendi preprocess_image()'ı: rembg YA DA alfa-maske,
            # sonra kare genişletme + sabit gri arkaplan doldurma, RGB'ye
            # çevirme (upstream run.py --bg_choice "Auto Remove background"
            # ile birebir aynı akış).
            if remove_bg:
                import rembg

                img = rembg.remove(image, session=rembg.new_session())
            else:
                bg = Image.new("RGBA", image.size, (0, 0, 0, 0))
                img = Image.alpha_composite(bg, image)
            w, h = img.size
            size = max(w, h)
            sq = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            sq.paste(img, ((size - w) // 2, (size - h) // 2))
            gray_bg = Image.new("RGBA", sq.size, (127, 127, 127))
            img = Image.alpha_composite(gray_bg, sq).convert("RGB")

            pipeline = self._model["pipeline"]
            recon_model = self._model["model"]

            yield _progress(f"iki aşamalı çoklu-görünüm difüzyonu ({num_inference_steps} adım)…")
            rt_dict = pipeline(img, scale=scale, step=num_inference_steps)
            np_imgs = np.concatenate(rt_dict["stage1_images"], 1)
            np_xyzs = np.concatenate(rt_dict["stage2_images"], 1)

            yield _progress("mesh çıkarılıyor + UV pişiriliyor (nvdiffrast)…")
            glb_path, _obj_zip_path = generate3d(recon_model, np_imgs, np_xyzs, self._device)
            mesh = trimesh.load(glb_path, force="mesh")
        else:
            raise AssertionError("unreachable")

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
        self._paint = None
        self._paint_root = ""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def vram_estimate(self, params: dict) -> int:
        family = str(params.get("family", "triposr") or "triposr")
        # triposr/sf3d/openlrm: küçük modeller (~700MB-2GB ağırlık) — kaba,
        # güvenli tahmin. hunyuan3d: canlı ölçüldü — num_inference_steps=30,
        # octree_resolution=384 (upstream varsayılanı) ile tepe VRAM 5983MB;
        # UI varsayılanı 256 daha hafif ama kullanıcı 384'e çıkarabilir, pay
        # bırakılıyor.
        if family == "hunyuan3d" and params.get("texture"):
            # Canlı ölçüldü (2026-09-18): shape 20 adım/256 + delight + dual-stream
            # paint UNet (1.86B param fp16) + 2048² render/pişirme ile tepe VRAM
            # 14872MB, toplam 79sn (paint ~46sn, ~14sn'si model yükleme).
            return 16_000
        if family in ("sf3d", "hunyuan3d"):
            return 6_500
        if family == "instantmesh":
            # Canlı ölçüldü — num_inference_steps=50, view=6, export_texmap=
            # False ile tepe VRAM 15590MB. Diğer dört aileden BELİRGİN daha
            # ağır: zero123plus tam bir SD1.5-boyutlu UNet+VAE+CLIP+vision_
            # encoder taşıyor, LRM triplane/FlexiCubes bunun üstüne biniyor.
            return 16_000
        if family == "trellis":
            # Canlı ölçüldü — num_inference_steps=25, formats=[mesh,gaussian],
            # texture_size=1024 ile tepe VRAM 11583MB (100 görünümlük
            # gaussian render + doku pişirme dahil).
            return 12_000
        if family == "crm":
            # Canlı ölçüldü — num_inference_steps=30, scale=5.0 ile (ısınmış
            # model, ikinci çağrı) tepe VRAM 13556MB — yedi aile arasında
            # TRELLIS'ten sonra ikinci en ağır: iki büyük difüzyon UNet'i
            # (pixel + CCM/XYZ, fp16) + CRM reconstruction modeli aynı anda
            # GPU'da.
            return 14_000
        return 4_000


def _nullcontext():
    from contextlib import nullcontext

    return nullcontext()
