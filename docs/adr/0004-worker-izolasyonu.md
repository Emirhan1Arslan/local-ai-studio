# ADR-0004: Inference İzolasyonu — Ayrı Worker Süreçleri

**Durum:** accepted · 2026-07-18

**Karar:** Her ağır iş ayrı worker sürecinde çalışır (scheduler → subprocess, JSON-lines IPC).

**Gerekçe:** CUDA OOM/çökme ana uygulamayı düşürmez; zorla iptal = süreç kill → VRAM'in
gerçekten geri verilmesi garanti olur.

**Sonuçlar:** IPC ve model yükleme maliyeti; LRU model cache ve idle timeout ile dengelenir
(Faz 3).
