# ADR-0001: Desktop Framework — Tauri 2 + React 18 + TypeScript (Vite)

**Durum:** accepted · 2026-07-18

**Bağlam:** Masaüstü kabuk; ağır iş Python tarafında, kabuk hafif olmalı.

**Karar:** Tauri 2 + React 18 + TS (Vite). Electron yalnızca Tauri'de aşılamayan bir platform
engeli çıkarsa B planıdır.

**Gerekçe:** Electron'a göre ~10x küçük binary, çok daha düşük RAM.

**Sonuçlar:** Rust toolchain geliştirme bağımlılığı olur; UI, kabuk hazır olana kadar Vite dev
modunda tarayıcıda da çalışabilir (aynı REST/WS yüzeyi — ADR-0003).
