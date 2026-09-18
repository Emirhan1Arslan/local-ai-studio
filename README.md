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

## MCP ile kullanım / uzaktan erişim

Studio çalışırken `http://127.0.0.1:8710/mcp` (streamable HTTP) ve `uv run studio-mcp`
(stdio köprüsü) aynı araç setini sunar: `generate_image`, `generate_video`, `generate_3d`,
`chat`, `transcribe_audio`, `synthesize_voice`, görüntü araçları, model indirme, workflow
çalıştırma. Claude Desktop / Claude Code / Cursor / VS Code / Windsurf tarifleri ve **LAN
üzerinden uzak makineden kullanım** için [docs/mcp.md](docs/mcp.md).

```bash
uv run studio-api --host 0.0.0.0 --token <gizli>   # LAN modu (token zorunlu, uyarı basar)
STUDIO_API_URL=http://<ip>:8710 STUDIO_API_TOKEN=<gizli> studio-mcp   # uzak istemcide köprü
```

## Durum

- **Faz 0–5 tamam:** monorepo/ADR'ler, FastAPI `/v1` + WS, Model Manager + indirme motoru,
  Job/Scheduler/crash-izole worker'lar, SDXL/SD3/FLUX görsel üretimi (img2img/inpaint/outpaint,
  ControlNet, IP-Adapter, LoRA), görüntü araç seti (BiRefNet, ESRGAN, Florence-2, SAM, OCR).
- **Faz 6–9 çekirdeği tamam:** llama.cpp sohbet (RAG/Ollama köprüsü ertelendi), STT/TTS
  (faster-whisper, Kokoro, XTTS-v2), video (LTX/Wan/CogVideoX/Hunyuan/Mochi + FFmpeg araçları),
  3D (yedi aile, viewer, GLB/OBJ/STL/PLY/FBX export).
- **Faz 10 DoD karşılandı:** tipli graph executor, node cache'i, React Flow editörü, `/v1/workflows`.
- **Faz 12 MCP tamam:** `/mcp` mount + `studio-mcp` stdio köprüsü, LAN modu token kapısı,
  istemci tarifleri ([docs/mcp.md](docs/mcp.md)). Python SDK sırada.
- **Yeni nesil görsel modeller (`imagex`):** HiDream-O1-Image (MIT, 8B, ~4 MP) ve
  Z-Image-Turbo (Apache-2.0, 6B) ayrı `envs/imagex` worker env'inde; Model Manager aileyi
  klasörden tanır, Üret paneli ve MCP otomatik yönlendirir (`docs/worker-envs.md`).

Ayrıntılı faz durumları ve ertelenen kalanlar için [plan.md](plan.md).

## Lisans

[Apache-2.0](LICENSE). Modeller uygulamayla dağıtılmaz; lisansları metadata olarak taşınır.
