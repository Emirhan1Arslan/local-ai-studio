"""Kenar-eşleşmeli (seamless / tileable) doku yardımcıları — tüm görsel adapter'ları paylaşır.

İki strateji:
- `set_circular_padding`: UNet + VAE'nin Conv2d katmanlarını `padding_mode="circular"`
  yapar (A1111'in "Tiling" seçeneği) — evrişimsel modellerde (SDXL/SD/FLUX'un VAE'si
  ve SDXL UNet'i) gerçek anlamda döngüsel üretim: sol kenar sağ kenarın devamıdır.
  Duck-typed (sınıf ADI Conv2d) — API süreci/testler torch'suz çalıştırabilsin.
- `make_tileable`: model-bağımsız son işlem (wrap-blend). Görsel yarı yarıya
  kaydırılır (ImageChops.offset = döngüsel kaydırma) — dış kenarlar artık orijinalin
  komşu orta sütunları olduğu için kesintisizdir; ortaya taşınan eski dikiş ise
  orijinalin (orada sürekli olan) pikselleriyle doğrusal bir bantta karıştırılarak
  kapatılır. DiT tabanlı modellerde (HiDream piksel-DiT, Z-Image) evrişimsel
  döngüsellik yoktur; bu yüzden orada bu fallback kullanılır. Bant genişliği kadar
  yumuşak bir geçiş bölgesi bırakır (oyun dokusu pratiğinde kabul gören "offset
  + blend" tekniği).
"""

from __future__ import annotations

from typing import Any


def set_circular_padding(module: Any, enabled: bool) -> int:
    """`module` altındaki tüm Conv2d katmanlarının padding_mode'unu ayarlar; değişen
    katman sayısını döner. torch Conv2d ileri geçişte `padding_mode`'u dinamik okur
    (zeros dışında F.pad + padding'siz conv), bu yüzden iş başına açıp kapatmak güvenli."""
    if module is None:
        return 0
    mode = "circular" if enabled else "zeros"
    count = 0
    for sub in module.modules():
        if sub.__class__.__name__ == "Conv2d" and hasattr(sub, "padding_mode"):
            if sub.padding_mode != mode:
                sub.padding_mode = mode
            count += 1
    return count


def _seam_mask_1d(length: int, band_px: int) -> list[int]:
    """Ortada 255, ±band_px/2 uzaklıkta 0'a inen üçgen profil."""
    center = length / 2.0
    half = max(1.0, band_px / 2.0)
    return [
        max(0, min(255, int(round(255 * (1.0 - abs(i + 0.5 - center) / half)))))
        for i in range(length)
    ]


def _wrap_blend_axis(rgb: Any, band: float, horizontal: bool) -> Any:
    """Tek eksende kaydır + orta bandı orijinalle karıştır. İki eksen tek maskeyle
    yapılamaz: yatay bant tam genişlikte orijinali kullanırsa o satırlarda sol/sağ
    kenar yine kopar — bu yüzden eksenler ardışık uygulanır."""
    from PIL import Image, ImageChops

    w, h = rgb.size
    if horizontal:
        rolled = ImageChops.offset(rgb, w // 2, 0)
        mask = Image.new("L", (w, 1))
        mask.putdata(_seam_mask_1d(w, int(band * w)))
    else:
        rolled = ImageChops.offset(rgb, 0, h // 2)
        mask = Image.new("L", (1, h))
        mask.putdata(_seam_mask_1d(h, int(band * h)))
    mask = mask.resize((w, h), Image.NEAREST)
    # mask 255 → orijinal (bu eksende ortası sürekli), 0 → kaydırılmış (kenarları sürekli)
    return Image.composite(rgb, rolled, mask)


def make_tileable(image: Any, band: float = 0.12) -> Any:
    """Görseli wrap-blend ile kenar-eşleşmeli hale getirir (bkz. modül docstring'i).
    `band`: karıştırma bandının kenar uzunluğuna oranı (0.02-0.5)."""
    rgb = image.convert("RGB")
    band = min(0.5, max(0.02, float(band)))
    step1 = _wrap_blend_axis(rgb, band, horizontal=True)
    return _wrap_blend_axis(step1, band, horizontal=False)
