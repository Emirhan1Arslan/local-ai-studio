# Mimari Özet

Kaynaklar: [önemli.md](../önemli.md) (ürün felsefesi) · [plan.md](../plan.md) (faz planı §2 mimari şeması).

## Orkestrasyon Katmanı (ürünün merkezi)

Platformun merkezinde modeller veya node'lar değil, kullanıcının **amacı (Goal)** bulunur:

```
Kullanıcı / MCP istemcisi (Claude, Cursor…)
        → Goal (amaç)
        → Planner        (niyet çözümleme, model/çözünürlük/araç seçimi — Faz 11)
        → Task Graph     (alt görevler + bağımlılıklar)
        → Scheduler      (öncelik kuyruğu + VRAM bütçesi — Faz 3)
        → Workers        (image / llm / video / audio / 3d — ayrı süreçler)
```

- **Planner** planlar, **Scheduler** yalnızca yürütür — iki sorumluluk tamamen ayrıdır.
- UI hiçbir zaman doğrudan model çalıştırmaz; her istek bir **Job** olur.
- Worker olayları WS üzerinden UI, SDK ve MCP istemcilerine **aynı formatta** akar.

## Engine Agnostic Adapter Katmanı

ComfyUI, Ollama, Whisper, XTTS vb. platformun parçası değil, **değiştirilebilir motorlardır**.
Tüm motorlar ortak adapter sözleşmesi üzerinden bağlanır:

```python
class EngineAdapter(Protocol):
    def load(...) ...
    def infer(...) -> AsyncIterator[Event]   # stream
    def unload(...) ...
    def vram_estimate(...) -> int
```

Adapter aileleri: Image · Video · Audio · 3D · LLM · Tool. Yeni motor eklemek = yeni adapter,
sistemin geri kalanı etkilenmez.

## Asset Graph

Üretilen her varlık tek dosya değil, üretim ağının düğümüdür. Karakter → portrait, icon,
sprite, mesh, voice, animation… ilişkileri DB'de tutulur; ana varlık değişince türevler
yeniden üretilebilir. (Şema Faz 4'te `studio_core.models` içinde genişletilecek.)

## MCP-first

MCP, dış istemciler için sonradan eklenmiş bir API değil, **birincil entegrasyon yüzeyidir**.
REST `/v1`, WS olayları ve MCP araçları aynı application-service katmanını çağırır (tek yüzey,
çift iş yok — ADR-0003).

## Süreç Modeli

```
Tauri (Rust kabuk) ──sidecar──▶ studio-api (FastAPI, headless de çalışır)
                                   │ REST /v1 + WS /v1/ws (token'lı)
                                   └─ Scheduler ──▶ worker süreçleri (crash-izole)
```

Kullanıcı verisi repo dışında: `~/LocalAIStudio/{models,projects,cache,logs,plugins,envs}`.
