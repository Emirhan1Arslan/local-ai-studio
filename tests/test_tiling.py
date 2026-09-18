"""Kenar-eşleşmeli doku yardımcıları (adapters/_tiling.py) — torch'suz, saf PIL testleri."""

from __future__ import annotations

from PIL import Image
from studio_inference.adapters._tiling import make_tileable, set_circular_padding


class _FakeConv2d:
    def __init__(self):
        self.padding_mode = "zeros"


class _FakeLinear:
    pass


class _FakeModule:
    def __init__(self, children):
        self._children = children

    def modules(self):
        return [self, *self._children]


Conv2d = type("Conv2d", (_FakeConv2d,), {})  # sınıf adı torch'unkiyle aynı olmalı


def test_set_circular_padding_toggles_only_conv2d():
    convs = [Conv2d(), Conv2d()]
    module = _FakeModule([*convs, _FakeLinear()])

    assert set_circular_padding(module, True) == 2
    assert all(c.padding_mode == "circular" for c in convs)
    assert set_circular_padding(module, False) == 2
    assert all(c.padding_mode == "zeros" for c in convs)
    assert set_circular_padding(None, True) == 0


def _edge_diff(img: Image.Image) -> tuple[int, int]:
    """(sol-sağ kenar, üst-alt kenar) maksimum kanal farkı — döşendiğinde dikiş ölçüsü."""
    w, h = img.size
    px = img.load()
    lr = max(abs(px[0, y][c] - px[w - 1, y][c]) for y in range(h) for c in range(3))
    tb = max(abs(px[x, 0][c] - px[x, h - 1][c]) for x in range(w) for c in range(3))
    return lr, tb


def test_make_tileable_removes_edge_seam():
    # Yatay + dikey gradyan: kenarlar (0 ↔ 255) belirgin dikiş, ortası pürüzsüz.
    w = h = 128
    src = Image.new("RGB", (w, h))
    src.putdata(
        [(int(255 * x / (w - 1)), int(255 * y / (h - 1)), 128) for y in range(h) for x in range(w)]
    )
    assert _edge_diff(src)[0] == 255

    out = make_tileable(src, band=0.2)

    assert out.size == (w, h)
    lr, tb = _edge_diff(out)
    assert lr <= 4 and tb <= 4  # döşenince kenarlar birbirinin devamı
    # eski dikişin taşındığı orta çizgi de karıştırmayla sürekli kalmalı
    px = out.load()
    mid = w // 2
    assert max(abs(px[mid - 1, y][0] - px[mid, y][0]) for y in range(h)) <= 6


def test_make_tileable_clamps_band():
    src = Image.new("RGB", (32, 32), (10, 20, 30))
    assert make_tileable(src, band=5.0).size == (32, 32)
    assert make_tileable(src, band=0.0).size == (32, 32)
