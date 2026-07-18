# ADR-0009: Görüntü/Video Pipeline — Diffusers Adapter Katmanı

**Durum:** accepted · 2026-07-18

**Karar:** Diffusers tabanlı adapter katmanı; her model ailesi (SDXL, SD3, FLUX, LTX, Wan…)
bir adapter'dır.

**Gerekçe:** En geniş model kapsamı; adapter deseni gelecekteki modelleri plugin ile eklemeyi
mümkün kılar (engine-agnostic ilke, bkz. önemli.md).

**Sonuçlar:** ComfyUI dahil harici motorlar da aynı adapter arayüzünden bağlanabilir; çekirdek
hiçbir motora kilitlenmez.
