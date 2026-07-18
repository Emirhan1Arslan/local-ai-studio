# ADR-0005: Python Ortam Stratejisi — uv + İzole Env'ler

**Durum:** accepted · 2026-07-18

**Karar:** uv ile yönetilen çekirdek env; çakışan model aileleri ve plugin'ler için izole
env'ler (`~/LocalAIStudio/envs`).

**Gerekçe:** diffusers/transformers sürüm cehennemi projenin 1 numaralı teknik riskidir
(plan.md §8); izolasyon şart.

**Sonuçlar:** Worker'lar farklı env'lerden başlatılabilmeli; env yöneticisi Faz 3'te
scheduler'a bağlanır.
