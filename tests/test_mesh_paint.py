"""Faz 9 — Hunyuan3D-Paint bağlama: denetimli UNet yükleme yardımcıları
(GPU'suz saf fonksiyonlar; gerçek doku üretimi GPU smoke testinde)."""

from __future__ import annotations

from studio_inference.adapters.mesh import classify_paint_unexpected_keys, paint_root_for

IP_KEYS = [
    "unet.image_proj_model.norm.weight",
    "unet.image_proj_model.proj.bias",
    "unet.down_blocks.0.attentions.0.transformer_blocks.0.transformer.attn2.to_k_ip.weight",
    "unet.up_blocks.3.attentions.2.transformer_blocks.0.transformer.attn2.to_v_ip.weight",
    "unet_dual.mid_block.attentions.0.transformer_blocks.0.transformer.attn2.to_k_ip.weight",
]


def test_known_ip_adapter_keys_are_tolerated():
    assert classify_paint_unexpected_keys(IP_KEYS) == []
    assert classify_paint_unexpected_keys([]) == []


def test_other_unexpected_keys_are_flagged():
    bad = [
        "unet.conv_in.weight",  # gerçek mimari anahtarı fazla gelirse yükleme güvenli değil
        "unet.down_blocks.0.attentions.0.transformer_blocks.0.transformer.attn2.to_k.weight",
        "unet.image_proj_model_extra.weight",
    ]
    assert classify_paint_unexpected_keys(IP_KEYS + bad) == bad


def test_paint_root_resolves_from_dit_folder(tmp_path):
    root = tmp_path / "Hunyuan3D-2"
    (root / "hunyuan3d-delight-v2-0").mkdir(parents=True)
    (root / "hunyuan3d-paint-v2-0-turbo").mkdir()
    dit = root / "hunyuan3d-dit-v2-0-fast"
    dit.mkdir()

    assert paint_root_for(str(dit)) == str(root)
    assert paint_root_for(str(root)) == str(root)


def test_paint_root_missing_components_is_none(tmp_path):
    root = tmp_path / "Hunyuan3D-2"
    (root / "hunyuan3d-dit-v2-0-fast").mkdir(parents=True)
    (root / "hunyuan3d-delight-v2-0").mkdir()  # paint klasörü yok

    assert paint_root_for(str(root / "hunyuan3d-dit-v2-0-fast")) is None
    assert paint_root_for("tencent/Hunyuan3D-2") is None
