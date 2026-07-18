# ADR-0013: Paketleme — Slim Çekirdek + Akıllı İlk Kurulum

**Durum:** accepted · 2026-07-18

**Karar:** Tauri bundler + python-build-standalone gömülü Python; torch varyantları
(CUDA/ROCm/CPU) ilk açılışta uv ile kurulur. Tam çevrimdışı kurulum için ayrı "full offline
bundle" varyantı.

**Gerekçe:** Tek dev installer yerine indirme boyutu yönetilebilir kalır (torch + CUDA
GB'larca).
