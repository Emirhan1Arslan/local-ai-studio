# Local AI Studio

Yerel, çevrimdışı çalışan açık kaynak **AI Orchestration Platform**. Amaç tek bir araç değil;
ComfyUI, llama.cpp/Ollama, Whisper, XTTS gibi motorları **tek akıllı orkestrasyon katmanı**
altında birleştirip kullanıcının *hedef odaklı* (Goal → Plan → Execution) çalışmasını sağlamak.

> Ayrıntılı ürün felsefesi: [önemli.md](önemli.md) · Faz faz uygulama planı: [plan.md](plan.md)
> Mimari özet: [docs/architecture.md](docs/architecture.md) · Kararlar: [docs/adr/](docs/adr/)

## Temel İlkeler

- **Goal → Planner → Task Graph → Scheduler → Workers** — kullanıcı amacını verir, sistem planlar.
- **Engine Agnostic** — her üretim motoru (görüntü, video, ses, 3D, LLM) değiştirilebilir bir adapter'dır.
- **Asset Graph** — üretilen her varlık türevleriyle (portrait, icon, sprite, mesh, voice…) ilişkili bir ağın parçasıdır.
- **MCP-first** — MCP, dış istemciler için ek bir API değil, platformun birincil entegrasyon yüzeyidir.
- **UI hiçbir zaman doğrudan model çalıştırmaz** — her istek bir Job olur; Scheduler VRAM bütçesine göre worker'a atar.

## Depo Yapısı

```
apps/desktop/       Tauri + React + TS masaüstü kabuğu (şimdilik Vite dev modu)
backend/            Python paketleri (uv workspace)
  studio_core/      Domain + config + DB + olay şemaları + donanım algılama
  studio_api/       FastAPI: REST /v1 + WebSocket olay veriyolu
  studio_inference/ Scheduler + worker'lar + model adapter'ları   (Faz 3)
  studio_models/    Model Manager + Download Manager              (Faz 2)
  studio_workflows/ Tipli graph executor + node kütüphanesi       (Faz 10)
  studio_agents/    Agent planner + tool registry + otomasyon     (Faz 11)
  studio_mcp/       MCP server (stdio + streamable HTTP)          (Faz 12)
  studio_plugins/   Plugin runtime + manifest + izinler           (Faz 13)
  studio_rag/       Embeddings + vektör deposu + ingest           (Faz 6)
sdk/python/         `localaistudio` PyPI paketi                   (Faz 12)
docs/               Dokümantasyon + ADR'ler
```

## Geliştirme

Gereksinimler: [uv](https://docs.astral.sh/uv/) · Node 20+ · pnpm 9+ · (masaüstü kabuk için) Rust.

```bash
# Backend
uv sync
uv run studio-api            # http://127.0.0.1:8710 (–port 0 → boş port)

# Frontend (ayrı terminal)
pnpm install
pnpm --filter desktop dev    # http://localhost:5173

# Testler
uv run pytest
```

## Durum

Faz 0–1 (walking skeleton) sürüyor. Yol haritası ve kilometre taşları için [plan.md](plan.md).

## Lisans

[Apache-2.0](LICENSE). Modeller uygulamayla dağıtılmaz; lisansları metadata olarak taşınır.
