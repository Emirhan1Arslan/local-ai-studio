# Local AI Studio — Faz Faz Uygulama Planı

**Sürüm:** 1.0 · **Tarih:** 7 Temmuz 2026
**Kapsam:** Kaynak prompt'taki tüm gereksinimler korunmuştur; hiçbir özellik kesilmemiş, uygulanabilir bir sıraya ve fazlara bölünmüştür.

---

## 0. Yönetici Özeti

Hedef: Higgsfield AI'ın yerel alternatifi olacak; ComfyUI + Automatic1111 + LM Studio + Ollama + Flowise + n8n yeteneklerini tek bir çevrimdışı masaüstü uygulamasında birleştiren, açık kaynak bir **AI Creation Platform** inşa etmek.

Bu ölçekte bir ürün tek seferde yazılamaz; "her şeyi aynı anda" yaklaşımı bu tür projelerin en sık başarısızlık nedenidir. Plan bu yüzden şu ilkeye dayanır:

> **Önce çekirdek platform (kabuk + model yönetimi + iş kuyruğu), sonra modaliteler tek tek, en sonda platformlaşma (workflow, agent, API, plugin).** Her faz kendi başına çalışan, test edilebilir ve yayınlanabilir bir ürün durumu üretir.

Plan **17 fazdan (Faz 0–16)** oluşur ve 6 kilometre taşına bağlanır:

| Kilometre Taşı | Fazlar | Sonuç |
|---|---|---|
| **M1 — Walking Skeleton** | 0–3 | Açılan, donanımı tanıyan, model indiren, iş kuyruklayan iskelet |
| **M2 — Görüntü MVP (Alfa 0.1)** | 4–5 | FLUX/SDXL üretimi + görüntü araç seti + proje sistemi |
| **M3 — Çok Modlu Platform (Beta'ya doğru)** | 6–9 | LLM/Chat/RAG, ses, video, 3D |
| **M4 — Platformlaşma** | 10–13 | Node editör, agent/otomasyon, REST/MCP/SDK, Plugin SDK + Marketplace |
| **M5 — Üretim Sertleştirme** | 14–15 | Güvenlik, performans, paketleme, auto-update, Docker |
| **M6 — v1.0** | 16 | Beta programı, dokümantasyon finali, yayın |

**Tahmini takvim:** 5–8 kişilik çekirdek ekip ve kısmi paralelleştirme ile **16–22 ay** (v1.0). 1–2 kişilik ekiple aynı sıra geçerlidir ancak süre 3–4 katına çıkar; bu durumda Faz 8 (video), Faz 9 (3D) ve Faz 13 (marketplace) v1.0 sonrasına ertelenebilir.

---

## 1. Temel Mimari Kararlar

Bu kararlar Faz 0'da birer ADR (Architecture Decision Record) olarak resmileştirilecek. Öneriler ve gerekçeleri:

| # | Konu | Karar | Gerekçe |
|---|---|---|---|
| 1 | Desktop framework | **Tauri 2 + React 18 + TypeScript (Vite)** | Electron'a göre ~10x küçük binary, çok daha düşük RAM; ağır iş zaten Python tarafında olduğundan kabuk hafif olmalı. Electron yalnızca Tauri'de aşılamayan bir platform engeli çıkarsa B planıdır. |
| 2 | Backend süreç modeli | Python backend'i Tauri tarafından yönetilen **sidecar süreç** | Çökme izolasyonu, bağımsız güncellenebilirlik, headless (Docker) modda aynı backend'in tek başına çalışabilmesi. |
| 3 | UI ↔ Backend iletişimi | localhost üzerinde **REST (kontrol) + WebSocket (olay/stream)**, tek kullanımlık token ile | Aynı API hem UI hem dış istemciler (SDK, MCP) tarafından kullanılır → tek yüzey, çift iş yok. |
| 4 | Inference izolasyonu | Her ağır iş **ayrı worker sürecinde** (scheduler → subprocess) | CUDA OOM/çökme ana uygulamayı düşürmez; zorla iptal ve VRAM'in gerçekten geri verilmesi garanti olur. |
| 5 | Python ortam stratejisi | **uv** ile yönetilen çekirdek env + çakışan model aileleri için **izole env'ler** | diffusers/transformers sürüm cehennemi bu projenin 1 numaralı teknik riskidir; izolasyon şart. |
| 6 | Veritabanı | **SQLite + SQLAlchemy 2 + Alembic**; vektörler için **sqlite-vec** (büyük korpus: LanceDB) | Sıfır kurulum, tek dosya, taşınabilir proje yedekleri. |
| 7 | Model formatları | **safetensors varsayılan**, GGUF, ONNX, TensorRT engine cache; pickle (.ckpt/.pt) karantina + açık uyarı ile | Güvenlik (pickle RCE riski) + performans. |
| 8 | LLM runtime | **llama.cpp (llama-cpp-python)** birincil; Transformers; opsiyonel vLLM (Linux/CUDA); **Ollama köprüsü** | GGUF her platformda çalışır; Ollama köprüsü mevcut kullanıcı kurulumlarını yeniden indirmeden kullanır. |
| 9 | Görüntü/video pipeline | **Diffusers** tabanlı adapter katmanı | En geniş model kapsamı; adapter deseni gelecekteki modelleri plugin ile eklemeyi mümkün kılar. |
| 10 | Node editör | Frontend: **React Flow (xyflow)** · Backend: özel tip güvenli graph executor | ComfyUI benzeri deneyim; yürütme backend'de olduğundan API/MCP'den de workflow çalıştırılabilir. |
| 11 | 3D görüntüleyici | **three.js** (React Three Fiber) | GLB/OBJ önizleme, orbit kontrol, malzeme incelemesi. |
| 12 | Dockable UI | **dockview** + Tailwind + Radix/shadcn | ComfyUI/Blender tarzı sürüklenebilir, kaydedilebilir yerleşimler. |
| 13 | Paketleme | Tauri bundler + **python-build-standalone** gömülü Python; torch varyantları (CUDA/ROCm/CPU) ilk açılışta uv ile kurulur | Tek dev installer yerine "slim çekirdek + akıllı ilk kurulum" → indirme boyutu yönetilebilir kalır. |
| 14 | GPU destek matrisi | CUDA (birincil) · ROCm (Linux birincil) · MPS (macOS) · CPU fallback · DirectML (Windows AMD/Intel, opsiyonel) | Windows+ROCm resmi olarak zayıf; dürüst bir destek matrisi yayınlanır. |
| 15 | Lisans | Çekirdek: **Apache-2.0**; modeller uygulamayla dağıtılmaz, lisansları metadata olarak taşınır | FLUX-dev gibi ticari kısıtlı model lisansları kullanıcıya UI'da açıkça gösterilir. |

---

## 2. Yüksek Seviyeli Mimari

```
┌──────────────────────────────────────────────────────────────────┐
│  DESKTOP SHELL (Tauri / Rust)                                    │
│  pencere · tray · dosya diyalogları · auto-update · sidecar mgmt │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  UI (React + TS): Dock Layout · Asset/Model Browser ·      │  │
│  │  Workflow Editor · Chat · Queue · VRAM Monitor · Settings  │  │
│  └───────────────▲────────────────────────────────────────────┘  │
└──────────────────┼───────────────────────────────────────────────┘
        REST /v1   │   WebSocket (olaylar, stream)      + token
┌──────────────────▼───────────────────────────────────────────────┐
│  BACKEND (Python · FastAPI · sidecar, headless de çalışır)       │
│  ┌ API Gateway: REST · WS · MCP Server · OpenAPI                 │
│  ┌ Application: use-case servisleri (CQRS-lite) · Agent katmanı  │
│  ┌ Domain: Project · Asset · ModelRecord · Job · Workflow ·      │
│  │         Plugin · Download                                     │
│  ┌ Infrastructure: SQLite repo'ları · Downloader · FS · Telemetri│
│  ┌ Workflow Engine: tipli graph executor · node cache            │
│  ┌ Plugin Runtime: izole süreç + izin köprüsü                    │
└──────────────────┬───────────────────────────────────────────────┘
                   │  Scheduler (öncelik kuyruğu · VRAM bütçesi)
     ┌─────────────┼─────────────┬──────────────┬─────────────┐
┌────▼────┐  ┌─────▼─────┐ ┌─────▼─────┐  ┌─────▼─────┐  ┌────▼────┐
│ Worker  │  │ Worker    │ │ Worker    │  │ Worker    │  │ Worker  │
│ Image   │  │ LLM       │ │ Video     │  │ Audio     │  │ 3D/Tool │
│(Diffus.)│  │(llama.cpp)│ │(Diffusers)│  │(whisper..)│  │(SAM2..) │
└─────────┘  └───────────┘ └───────────┘  └───────────┘  └─────────┘
   PyTorch · ONNX Runtime · TensorRT · CUDA/ROCm/MPS/CPU
```

Kritik akış: UI hiçbir zaman doğrudan model çalıştırmaz. Her istek bir **Job** olur → Scheduler VRAM bütçesine ve önceliğe göre uygun worker'a atar → worker olayları WS üzerinden UI'a, SDK'ya ve MCP istemcilerine aynı formatta akar.

---

## 3. Monorepo Klasör Yapısı

```
local-ai-studio/
├─ apps/
│  └─ desktop/                  # Tauri + React + TS
│     ├─ src/                   # React UI (panels/, features/, stores/)
│     └─ src-tauri/             # Rust kabuk, sidecar lifecycle, updater
├─ backend/
│  ├─ studio_core/              # Domain + application (framework bağımsız)
│  ├─ studio_api/               # FastAPI: REST + WS + auth
│  ├─ studio_inference/         # Scheduler, worker'lar, model adapter'ları
│  │  └─ adapters/              # torch/, onnx/, tensorrt/, llamacpp/
│  ├─ studio_models/            # Model Manager + Download Manager
│  ├─ studio_workflows/         # Graph engine + node kütüphanesi
│  ├─ studio_agents/            # Agent planner + tool registry + otomasyon
│  ├─ studio_mcp/               # MCP server (stdio + streamable HTTP)
│  ├─ studio_plugins/           # Plugin runtime, manifest, izinler, imza
│  └─ studio_rag/               # Embeddings, vektör deposu, ingest
├─ sdk/
│  └─ python/                   # `localaistudio` PyPI paketi (sync+async)
├─ plugins/                     # Resmi örnek plugin'ler + şablon
├─ docs/                        # Dokümantasyon sitesi (MkDocs Material)
├─ installers/                  # NSIS/MSI, dmg, AppImage/deb/rpm/Flatpak
├─ docker/                      # Headless API+MCP imajları (cuda/rocm/cpu)
├─ scripts/                     # dev, benchmark, release yardımcıları
├─ tests/                       # e2e/, gpu/, benchmarks/
└─ .github/workflows/           # CI/CD (lint, test, 3-OS build, release)
```

Kullanıcı verisi (uygulama dizini, repo dışı): `~/LocalAIStudio/{models,projects,cache,logs,plugins,envs}` — `models/` altı prompt'taki şemaya uyar (`image/ video/ 3d/ tts/ stt/ ocr/ embeddings/ llm/ lora/ controlnet/ checkpoints/ vae/`).

---

## 4. Zaman Çizelgesi Özeti

Süreler 5–8 kişilik ekip içindir. "Paralel" sütunu, önceki fazlar bitmeden başlatılabilecek işleri gösterir.

| Faz | Başlık | Süre | Paralellik notu |
|---|---|---|---|
| 0 | Mimari, kararlar, proje kurulumu | 2–3 hafta | — |
| 1 | Çekirdek altyapı (shell + backend) | 4–6 hafta | UX tasarımı paralel |
| 2 | Model Manager + Download Manager | 3–4 hafta | Faz 3 ile kısmen paralel |
| 3 | Inference Runtime + GPU Scheduler | 4–6 hafta | Faz 2 ile kısmen paralel |
| 4 | Görüntü üretimi + Proje sistemi → **Alfa 0.1** | 6–8 hafta | — |
| 5 | Görüntü araç seti + OCR | 4–6 hafta | Faz 6 ile paralel |
| 6 | LLM: Chat, Embeddings, RAG | 5–7 hafta | Faz 5 ile paralel |
| 7 | Ses: STT, TTS, müzik | 4–6 hafta | Faz 8 ile paralel |
| 8 | Video üretimi + düzenleme | 6–8 hafta | Faz 7 ile paralel |
| 9 | 3D üretim + viewer + export | 4–6 hafta | Faz 10 ile paralel |
| 10 | Node Workflow Editörü | 8–10 hafta | Faz 9/11 ile paralel |
| 11 | Agent sistemi + otomasyon | 4–6 hafta | Faz 12 ile paralel |
| 12 | REST · WS · MCP Server · Python SDK | 5–7 hafta | Faz 11 ile paralel |
| 13 | Plugin SDK + Marketplace | 6–8 hafta | — |
| 14 | Güvenlik + performans sertleştirme | 4–6 hafta | Sürekli izlerin finali |
| 15 | Paketleme, dağıtım, auto-update, Docker | 4–6 hafta | Faz 14 ile paralel |
| 16 | Beta, dokümantasyon finali, **v1.0** | 6–8 hafta | — |

Kritik yol: 0 → 1 → 3 → 4 → 10 → 12/13 → 15 → 16.

---

## 5. Faz Detayları

Her fazda: **Amaç → Adımlar → Çıktılar → Kabul Kriterleri (DoD)**. Testler ve dokümantasyon her fazın içindedir (bkz. Bölüm 6), ayrı bir "sona bırakılmış" faz değildir.

---

### FAZ 0 — Mimari, Teknoloji Kararları ve Proje Kurulumu (2–3 hafta)

**Amaç:** Kod yazmaya başlamadan önce tüm yapısal kararları vermek, iskeleti ve OSS altyapısını kurmak.

**Adımlar**
1. Kaynak prompt'u bir **özellik matrisine** dök: özellik × platform (Win/Linux/macOS) × GPU backend (CUDA/ROCm/MPS/CPU) × faz numarası. Bu matris projenin tek kapsam kaynağı olur.
2. **ADR süreci** kur; Bölüm 1'deki 15 kararı birer ADR olarak yaz ve tartışmaya kapat.
3. Monorepo iskeletini oluştur: pnpm workspaces (frontend) + uv workspace (Python); ruff, mypy, eslint, prettier, pre-commit.
4. **CI iskeleti** (GitHub Actions): lint + unit test + 3 OS'te CPU-only build matrisi. Self-hosted GPU runner planını yaz (gecelik GPU testleri için).
5. OSS hijyeni: LICENSE (Apache-2.0), CODE_OF_CONDUCT, CONTRIBUTING, issue/PR şablonları, RFC süreci, semver + değişiklik günlüğü politikası.
6. **UX temeli:** bilgi mimarisi, ana panellerin wireframe'leri (Asset Browser, Model Browser, Üretim, Queue, Workflow, Chat, Settings), dark theme design token'ları.
7. Tehdit modeli taslağı (pickle modeller, plugin'ler, workflow'daki Python/HTTP node'ları, yerel API erişimi).
8. Faz 1 backlog'unu sprint görevlerine kır; "walking skeleton" tanımını yaz.

**Çıktılar:** ADR seti · derlenen boş monorepo · yeşil CI · wireframe'ler · özellik matrisi · risk/tehdit dokümanı.
**DoD:** Üç OS'te "boş" uygulama derleniyor; her PR lint+test kapısından geçiyor.

---

### FAZ 1 — Çekirdek Altyapı: Desktop Shell + Backend Çekirdeği (4–6 hafta)

**Amaç:** Tüm platformun üzerine oturacağı kabuğu ve backend omurgasını çalışır hâle getirmek.

**Adımlar**
1. **Tauri kabuğu:** pencere yönetimi, menü/tray, tema; React uygulama iskeleti (router, Zustand store'ları, hata sınırları).
2. **Sidecar lifecycle:** Tauri, gömülü Python runtime'ını başlatır/izler/kapatır; boş port + tek kullanımlık token pazarlığı; çökmede otomatik yeniden başlatma; uygulama kapanınca süreç sızıntısı olmaması.
3. **FastAPI çekirdeği:** `/v1` sürümleme, `/health`, `/system/info`, standart hata modeli, structured logging (structlog + rotasyon), OpenAPI otomatik dokümanı.
4. **WebSocket olay veriyolu:** `job.*`, `download.*`, `system.*`, `log` olay şemaları — UI, SDK ve MCP aynı olayları dinleyecek.
5. **SQLite katmanı:** SQLAlchemy modelleri (settings, jobs, models, projects iskeleti) + Alembic migration akışı.
6. **Donanım algılama servisi:** CUDA/ROCm/MPS/CPU tespiti, VRAM/RAM/disk; UI'da Sistem paneli.
7. **Ayarlar sistemi:** şema tabanlı (Pydantic), UI Settings paneli; yol yapılandırması (models dizini, cache, projeler).
8. **Dockable layout:** dockview entegrasyonu, yerleşim kaydet/yükle; **Console paneli** (WS'ten canlı log).
9. Yerel çökme raporu: crash dump + log paketleme (gönderim yok, kullanıcı elle paylaşır).

**Çıktılar:** Açılan, sidecar'ı yöneten, sistem bilgisi ve canlı log gösteren uygulama.
**DoD:** 3 OS'te sidecar otomatik başlıyor/kapanıyor; backend çökertilince UI hayatta kalıp yeniden bağlanıyor; WS'ten log akıyor.

---

### FAZ 2 — Model Manager + Download Manager (3–4 hafta)

**Amaç:** Prompt'taki model dizin şemasını canlı tarayan, indiren, doğrulayan tam bir model yönetimi.

**Adımlar**
1. Model dizin şemasını uygula; **watchdog** ile canlı izleme → yeni model **restart'sız** algılanır.
2. **Metadata çıkarımı:** safetensors header, GGUF metadata, `config.json`; tür sınıflandırma (checkpoint/LoRA/VAE/ControlNet/LLM…); blake3/sha256 hash; boyut.
3. **Model registry (DB):** aile, sürüm, lisans, etiketler, favoriler, kullanım sayacı.
4. **İndirme motoru:** Hugging Face Hub (resume destekli), CivitAI API (token + sürüm/hash metadata), GitHub Releases, yerel arşiv içe aktarma; sha256 doğrulama; eşzamanlılık ve hız limiti; disk alanı ön kontrolü.
5. **Download Manager UI:** kuyruk, ilerleme, duraklat/devam/iptal, hata + yeniden dene.
6. **Model Browser UI:** filtre/arama/detay, sil/taşı, favori, lisans rozeti.
7. Lisans farkındalığı: ticari kısıtlı lisanslarda (ör. FLUX-dev) belirgin uyarı + onay.

**DoD:** SDXL checkpoint'i HF'den indirilir, doğrulanır, otomatik listelenir; yarıda kesilen indirme kaldığı yerden sürer; klasöre elle atılan GGUF 5 sn içinde görünür.

---

### FAZ 3 — Inference Runtime + GPU Scheduler (4–6 hafta)

**Amaç:** Tüm modalitelerin paylaşacağı iş kuyruğu, worker izolasyonu ve VRAM yönetimi. **Projenin kalbi.**

**Adımlar**
1. **Job modeli:** durum makinesi (queued → running → done | failed | canceled | paused), öncelik, sahiplik (proje/API istemcisi), tahmini süre alanı.
2. **Scheduler:** öncelikli kuyruk; multi-GPU cihaz ataması; model başına VRAM tahmini + kapasite kontrolü; pause/resume/cancel.
3. **Worker mimarisi:** iş türü başına ayrı süreç; JSON-lines IPC; crash izolasyonu; zorla iptal = süreç kill → VRAM garanti geri döner.
4. **Model yaşam döngüsü:** yükle/boşalt API'si, LRU cache, idle timeout, VRAM baskısında otomatik boşaltma (prompt'taki auto load/unload).
5. **Backend adapter sözleşmesi:** `load() / infer(stream) / unload() / vram_estimate()` — TorchAdapter, OnnxAdapter, LlamaCppAdapter tam; TensorRTAdapter stub (Faz 14'te doldurulur).
6. **Telemetri:** pynvml / rocm-smi / psutil → WS ile UI'a saniyelik VRAM/GPU util akışı.
7. **Generation Queue UI + VRAM Monitor UI** (sıra değiştirme, iptal, öncelik).
8. GPU smoke-test altyapısı: self-hosted runner'da gecelik "yükle → üret → boşalt → VRAM sıfırlandı mı" testleri.

**DoD:** 10 sahte iş öncelik sırasıyla işlenir; biri çalışırken iptal edilir ve VRAM geri gelir; worker'ı `kill -9` ile öldürmek uygulamayı düşürmez, iş "failed" olarak işaretlenir.

---

### FAZ 4 — Görüntü Üretimi + Proje Sistemi v1 → **ALFA 0.1** (6–8 hafta)

**Amaç:** İlk gerçek modalite. Kullanıcının "değer anı": temiz kurulumdan ilk görsele.

**Adımlar**
1. **Diffusers adapter'ları:** SDXL (+ Juggernaut/RealVisXL/Pony gibi fine-tune'lar otomatik uyumlu) → SD 3.x → FLUX (schnell/dev); sampler/scheduler seçimi; seed determinizmi.
2. **VRAM stratejileri:** fp16/bf16, attention slicing, CPU offload, VAE tiling; düşük VRAM için GGUF quantized FLUX (opsiyonel yol).
3. **LoRA:** çoklu LoRA yükleme + ağırlıklandırma; DreamBooth checkpoint'leri checkpoint olarak destek.
4. **ControlNet** (canny/depth/pose + ön işlemciler), **IPAdapter**, **InstantID**.
5. **img2img · inpaint** (maske fırçası) · **outpaint** (tuval genişletme).
6. **Üretim UI:** prompt/negatif, parametre paneli, adım adım canlı önizleme, sonuç galerisi; üretim metadata'sının PNG'ye gömülmesi (A1111 uyumlu format).
7. **Proje sistemi v1:** proje oluştur; asset kaydı; üretim geçmişi (prompt/negatif/seed/model/parametre); tek tıkla **"yeniden üret"**.
8. **Batch üretim** ve seed varyasyonları; **Asset Browser v1** (grid, önizleme, metadata paneli).
9. Alfa 0.1 paketi: 3 OS için imzasız geliştirici build'i, dar çevre testi.

**DoD:** Temiz kurulumda kullanıcı Model Browser'dan SDXL indirir ve ilk görselini üretir; 8 GB VRAM'de FLUX-schnell offload ile çalışır; aynı seed + parametre → aynı görsel.

---

### FAZ 5 — Görüntü Araç Seti + OCR (4–6 hafta)

**Amaç:** Prompt'taki tüm görüntü araçlarını tek tuvalde birleştirmek.

**Adımlar**
1. **Arka plan kaldırma:** RMBG-2.0, BiRefNet.
2. **Segmentasyon:** SAM2 (nokta/kutu ile), Grounding DINO ile "metinden segment" (open-vocabulary).
3. **Caption:** Florence-2 (kısa/detaylı caption, nesne tespiti) — otomatik etiketleme Asset Browser'ı besler.
4. **Upscale:** RealESRGAN + genel ESRGAN model desteği; SUPIR (ağır — "gelişmiş" bayrağıyla).
5. **Yüz restorasyonu:** GFPGAN, CodeFormer.
6. **OCR:** PaddleOCR, EasyOCR, Surya, Florence; dil paketi yönetimi; çıktılar txt/json/hOCR.
7. **Görüntü Editörü tuvali:** maske, kırpma, temel ayarlar; araç zincirleme ("seç → kaldır → inpaint ile doldur", "upscale → face restore").
8. Tüm araçların Job/Queue sistemi üzerinden çalışması (Faz 3 mimarisi burada kendini kanıtlar).

**DoD:** Ürün fotoğrafında tek tıkla şeffaf PNG; "kırmızı araba" yazınca ilgili bölge segmentlenir; taranmış belge OCR'ı referans sette hedef doğruluğu geçer.

---

### FAZ 6 — LLM Katmanı: Chat, Embeddings, RAG (5–7 hafta)

**Amaç:** LM Studio + temel Flowise yeteneği: yerel sohbet, embedding ve belge tabanlı soru-cevap.

**Adımlar**
1. **llama.cpp entegrasyonu:** GGUF yükleme, context/KV ayarları, GPU offload katman sayısı otomatik önerisi (Qwen, Llama, Gemma, Mistral, DeepSeek, Phi, EXAONE — hepsi GGUF üzerinden tek adapter'la gelir).
2. **Transformers backend** (safetensors LLM'ler); **opsiyonel vLLM** (Linux/CUDA, güç kullanıcıları); **Ollama köprüsü** (kuruluysa modellerini yeniden indirmeden kullan).
3. **Chat servisi:** token streaming (WS), oturum geçmişi (DB), sistem prompt şablonları, otomatik chat-template tespiti.
4. **Chat UI:** oturum listesi, model seçici, parametreler, durdur/yeniden üret, markdown + kod bloğu render.
5. **Embedding servisi:** bge / gte / nomic-embed; toplu embedding job'u.
6. **Vektör deposu:** sqlite-vec varsayılan; büyük koleksiyonlar için LanceDB.
7. **RAG pipeline:** klasör/dosya ingest (pdf, docx, md, html, txt), chunking stratejileri, retriever (+ opsiyonel reranker), cevapta kaynak gösterimi.
8. **Tool-use temeli:** function-calling arayüzü — Faz 11'deki agent'ların ön koşulu.

**DoD:** Qwen GGUF referans donanımda akıcı stream eder; 100 PDF'lik korpusa kaynak göstererek cevap verir; Ollama'daki mevcut model listede görünür ve çalışır.

**Durum (2026-07-19):** Adım 1, 3, 4 tamamlandı ve RTX 5090'da canlı doğrulandı
(llama.cpp + GGUF, token streaming, kalıcı sohbet geçmişi, Sohbet UI paneli —
commit `2c5eb11`). **Ertelenen kalanlar** (kullanıcı isteğiyle bilinçli olarak
bu fazın sonuna bırakıldı, ileride "Faz 6 kalanları" olarak sürdürülecek):
- Adım 2: Transformers backend (safetensors LLM'ler), opsiyonel vLLM, Ollama köprüsü
- Adım 5: Embedding servisi (bge/gte/nomic-embed)
- Adım 6: Vektör deposu (sqlite-vec/LanceDB)
- Adım 7: RAG pipeline (belge ingest, chunking, retriever, kaynak gösterimi)
- Adım 8: Tool-use temeli (function-calling — Faz 11 agent'larının ön koşulu)

DoD'nin "100 PDF'lik korpusa kaynak göstererek cevap verir" ve "Ollama'daki
model listede görünür" maddeleri bu yüzden henüz karşılanmadı — RAG/Ollama
köprüsü tamamlanınca geçerli olacak.

---

### FAZ 7 — Ses: STT, TTS, Müzik (4–6 hafta)

**Adımlar**
1. **STT:** faster-whisper (CTranslate2) varsayılan; WhisperX (kelime hizalama + diarization); Parakeet (NeMo, opsiyonel).
2. Mikrofon kaydı + dosya transkripsiyonu; srt/vtt/json zaman damgalı çıktı; toplu klasör transkripsiyonu.
3. **TTS:** Kokoro (hafif varsayılan), XTTS-v2 (ses klonlama), F5-TTS, CosyVoice, Bark; ses profili yönetimi (kayıtlı klon profilleri).
4. Ses klonlamada **etik onay akışı** (kullanıcı, sesin sahibinden izinli olduğunu beyan eder; çıktılara opsiyonel işaret).
5. **Müzik/ses üretimi:** MusicGen, Stable Audio Open.
6. **Ses UI:** kayıt, dalga formu, oynatıcı, kırpma; çıktılar WAV/FLAC/MP3.

**DoD:** 10 dk'lık ses referans GPU'da gerçek zamanın altında transkribe edilir; 20 sn örnekten klonlanan sesle cümle sentezlenir; 30 sn müzik klibi üretilir.

**Durum (2026-07-19):** Adım 1 (yalnız faster-whisper — WhisperX/Parakeet
ertelendi), 2 (dosya transkripsiyonu + mikrofon kaydı + srt/vtt/json; toplu
klasör transkripsiyonu ertelendi), 3 (Kokoro + **XTTS-v2 ses klonlama, çok
dilli — F5-TTS/CosyVoice/Bark ve kayıtlı klon profili yönetimi ertelendi**),
4 (etik onay akışı — UI'da zorunlu onay kutusu + backend `consent` guard'ı),
6 tamamlandı ve RTX 5090'da canlı doğrulandı: Kokoro TTS→STT round-trip
birebir eşleşti; XTTS-v2 ile Türkçe metin + referans sesten klonlanmış Türkçe
konuşma üretildi (model yükleme 9.4sn, sentez 3.1sn), faster-whisper
`language=tr` ile geri transkribe edilip dil algılama %100 "tr" doğrulandı.
**Ertelenen kalanlar:**
- Adım 1 kalan: WhisperX (kelime hizalama + diarization), Parakeet (NeMo)
- Adım 2 kalan: toplu klasör transkripsiyonu
- Adım 3 kalan: F5-TTS, CosyVoice, Bark, kayıtlı ses klon profili yönetimi
- Adım 5: müzik/ses üretimi (MusicGen, Stable Audio Open)

Bilinen sınır: Kokoro'nun varsayılan sesleri yalnızca İngilizce fonemleştirme
yapıyor — Türkçe metin İngilizce okunuş kurallarıyla telaffuz ediliyor (bkz.
docs/worker-envs.md). **Çözüldü:** XTTS-v2 motoru (Ses panelinde seçilebilir)
çok dilli/Türkçe seslendirmeyi ve ses klonlamayı sağlıyor. **Lisans notu:**
XTTS-v2 model ağırlıkları Coqui Public Model License (CPML) ile geliyor —
yalnızca ticari olmayan kullanım; bu UI'da XTTS motoru seçilince kullanıcıya
gösteriliyor. XTTS-v2 kurulumu ayrıca sistemde bulunmayan bir FFmpeg paylaşımlı
kütüphane bağımlılığı (torchcodec üzerinden) ortaya çıkardı — çözümü ve tam
bağımlılık zinciri docs/worker-envs.md'de belgelendi.

---

### FAZ 8 — Video Üretimi ve Düzenleme (6–8 hafta)

**Amaç:** Higgsfield'in çekirdek vaadi — yerelde. En VRAM-yoğun faz; kademeli model desteği şart.

**Adımlar**
1. **T2V/I2V adapter'ları — kademeli:** LTX-Video (hızlı, düşük VRAM — varsayılan) → Wan 2.x → CogVideoX → Hunyuan Video → Mochi → Pyramid Flow → SkyReels.
2. **Uzun iş yönetimi:** düşük öncelik sınıfı, tahmini süre, ara kare önizlemesi, mümkün olan modellerde checkpoint/sürdürme.
3. **Bellek stratejileri:** sequential CPU offload, tiled VAE decode, fp8 (destekleyen modellerde), çözünürlük/frame önerileri (VRAM'e göre otomatik öneri).
4. **ffmpeg tabanlı düzenleme:** kes, birleştir, hız, ses ekleme, format dönüştürme (MP4/MOV/AVI/GIF); frame interpolasyonu (RIFE) ve video upscale (opsiyonel).
5. **Video UI:** basit timeline, kare kare önizleme, kuyruk entegrasyonu, "görüntüden video" akışı (Faz 4 çıktılarıyla köprü).

**DoD:** 5 sn 720p klip LTX ile üretilir, iptal/duraklat çalışır ve VRAM geri döner; ffmpeg zinciri (kes+birleştir+ses) UI'dan kurulup çalışır.

---

### FAZ 9 — 3D Üretim (4–6 hafta)

**Adımlar**
1. **Image-to-3D — kademeli:** TripoSR (hafif, hızlı) → Stable Fast 3D → InstantMesh → TRELLIS → Hunyuan3D-2 → OpenLRM/CRM.
2. **Mesh işleme:** trimesh/pymeshlab — decimate, normal düzeltme, texture aktarımı, watertight kontrolü (3D baskı için).
3. **Export:** GLB/GLTF/OBJ/PLY/STL doğrudan; FBX için assimp veya headless Blender köprüsü.
4. **3D viewer:** three.js (R3F) — orbit, wireframe, malzeme/ışık, turntable önizleme videosu.
5. Görüntü → 3D köprüsü: Faz 4 çıktısı tek tıkla 3D job'una gönderilir; 3D varlıklar proje sistemine girer.

**DoD:** Tek görselden GLB üretilip viewer'da incelenir; STL bir slicer'da hatasız açılır; FBX Blender'da import edilir.

---

### FAZ 10 — Node Workflow Editörü (8–10 hafta)

**Amaç:** ComfyUI eşdeğeri görsel pipeline; tüm modaliteleri (Faz 4–9) node olarak birleştirir. **M4'ün temeli.**

**Adımlar**
1. **Graph çekirdeği (backend):** tipli port sistemi (image, latent, text, audio, video, mesh, tensor, any), topolojik yürütme, node bazlı çıktı cache'i, hata yayılımı, deterministik seed aktarımı.
2. **Kontrol akışı node'ları:** Condition, Loop (map/for-each), Variables, Memory, SubGraph (workflow içinde workflow).
3. **Node kütüphanesi:** Prompt, LLM, Image Gen, ControlNet, LoRA, IPAdapter, Upscale, BG Remove, Segment, Caption, OCR, TTS, STT, Video, 3D, Save/Export, Python (sandbox — Faz 14'te sıkılaştırılır), HTTP/REST Request, MCP Tool, Custom Node arayüzü.
4. **Editor UI (React Flow):** pan/zoom, grup/not, node arama paleti, tip uyumlu bağlantı doğrulama (yanlış bağlantı daha çizerken engellenir), node üstünde canlı progress ve hata rozetleri.
5. **Çalıştırma UX:** kısmi çalıştırma ("bu node'a kadar"), cache göstergesi (değişmeyen dallar yeniden koşmaz), kuyruk entegrasyonu.
6. **Workflow formatı:** sürümlü JSON şema; import/export; şablon galerisi (10+ hazır workflow).
7. **ComfyUI import (best-effort dönüştürücü):** desteklenen node'ları eşle, desteklenmeyenleri raporla — topluluk benimsemesi için kritik kaldıraç.
8. Workflow'ların proje sistemi ve (Faz 12'de) `workflow_run` API'siyle bağlanması.

**DoD:** "Prompt → FLUX → Upscale → BG Remove → Save" grafiği kurulur ve çalışır; ikinci çalıştırmada değişmeyen dallar cache'ten gelir; bozuk workflow yüklenince anlaşılır doğrulama hatası verir.

---

### FAZ 11 — Agent Sistemi + Otomasyon (4–6 hafta)

**Adımlar**
1. **Agent çekirdeği:** planner (yerel LLM, Faz 6 tool-use altyapısı) + **tool registry** (tüm iç servisler araç olarak) + yürütücü; adım limiti, süre bütçesi, kritik eylemlerde kullanıcı onayı.
2. **Kaynak-farkında karar katmanı:** hangi model / workflow / çözünürlük / GPU — kural tabanlı çekirdek (VRAM uygunluğu, model yetenek matrisi) + LLM ile niyet çözümleme (hibrit; saf LLM kararına güvenilmez).
3. **Otomasyon tetikleyicileri:** klasör izleme (hot folder), zamanlama (cron), yerel webhook.
4. **Pipeline kayıtları + denetim izi:** her agent adımı ve aracın girdi/çıktısı loglanır, UI'dan incelenir.
5. **Agent UI:** görev tarifi → plan önizleme → onay → adım adım yürütme izleme → sonuç raporu.

**DoD:** "Bu klasöre düşen her görselin arka planını kaldır, 2x upscale et, çıktı klasörüne yaz" otomasyonu kurulur ve dosya düştükçe çalışır; agent, 6 GB VRAM'li makinede FLUX-dev yerine uygun modeli seçtiğini gerekçesiyle raporlar.

---

### FAZ 12 — Dış API'ler: REST, WebSocket, MCP Server, Python SDK (5–7 hafta)

**Amaç:** İç API'yi dış dünyaya açmak. UI zaten aynı API'yi kullandığı için bu faz büyük ölçüde stabilizasyon + sözleşme işidir.

**Adımlar**
1. **REST yüzeyini stabilize et:** tüm özellikler `/v1` altında; OpenAPI temizliği; API token yönetimi (oluştur/iptal/kapsam); localhost-varsayılan + opsiyonel LAN modu uyarılarla.
2. **WS streaming sözleşmeleri:** üretim adım olayları, LLM token stream'i, iş/indirme olayları — kararlı, sürümlü şema.
3. **MCP Server:** stdio + streamable HTTP transport. Araçlar: `generate_image, generate_video, generate_3d, chat, transcribe_audio, synthesize_voice, remove_background, upscale, caption, ocr, segment, download_model, install_model, list_models, workflow_run, workflow_create, workflow_save, workflow_export`. Uzun işler için **job-id + poll/subscribe** deseni; yıkıcı/pahalı araçlarda izin onayı.
4. Entegrasyon tarifleri + örnek konfigürasyonlar: Claude Desktop, Cursor, VS Code, Windsurf, özel MCP istemcileri.
5. **Python SDK (`localaistudio`):** OpenAPI'den üretilen çekirdek + elle yazılmış ergonomik katman (`studio.generate_image()`, `studio.generate_video()`, `studio.chat()`, `studio.generate_3d()`, `studio.remove_background()` …); sync + async; job bekleme/stream yardımcıları; PyPI yayını.
6. API bölümü dokümantasyonu + uçtan uca kod örnekleri.

**DoD:** Claude Desktop, MCP üzerinden görsel üretir ve dosya yolunu döndürür; 10 satırlık SDK script'i bir video işini kuyruklar, ilerlemesini stream eder, sonucu indirir.

---

### FAZ 13 — Plugin SDK + Marketplace (6–8 hafta)

**Amaç:** Üçüncü tarafların çekirdeğe dokunmadan model, node, panel, exporter ve API eklemesi.

**Adımlar**
1. **Plugin manifesti (`plugin.toml`):** kimlik, sürüm, min. uygulama sürümü, **izinler**, giriş noktaları (model adapter / node / UI panel / exporter / API route / MCP tool).
2. **Python plugin runtime:** ayrı süreçte çalışır, çekirdeğe kısıtlı RPC köprüsüyle erişir; bağımlılık çakışmasına karşı **plugin başına opsiyonel izole venv**.
3. **UI eklentileri:** sandboxed webview panel + tip güvenli postMessage köprüsü (plugin, ana DOM'a dokunamaz).
4. **İzin sistemi:** dosya kapsamları, ağ erişimi (varsayılan **kapalı**), model/GPU erişimi; kurulumda izin onay ekranı.
5. **İmzalama:** minisign/sigstore tabanlı imza + doğrulama; imzasız plugin'de belirgin uyarı ve varsayılan engel.
6. **Plugin Manager UI:** kur/kaldır/güncelle/devre dışı bırak; çöken plugin'in izole raporu (uygulama ayakta kalır).
7. **Marketplace v1:** git-repo tabanlı imzalı statik indeks; arama, kategori, sürüm uyumluluk kontrolü; UI'da Marketplace paneli.
8. Örnek plugin'ler (yeni bir upscaler modeli, özel bir node, bir exporter) + cookiecutter şablonu + Plugin SDK dokümantasyonu.

**DoD:** Örnek üçüncü taraf plugin, çekirdek koduna dokunmadan yeni bir node ve panel ekler; ağ izni olmayan plugin'in soket açması engellenir ve raporlanır; çöken plugin uygulamayı düşürmez.

---

### FAZ 14 — Güvenlik ve Performans Sertleştirme (4–6 hafta)

**Güvenlik adımları**
1. **Workflow doğrulama:** şema doğrulaması + tehlikeli node'lar (Python, HTTP) için içe aktarımda izin kapısı ("bu workflow kod çalıştırıyor" onayı).
2. Python node sandbox sıkılaştırma: süreç izolasyonu, kaynak limitleri (CPU/RAM/süre), dosya kapsamı; ileri seviye için container opsiyonu.
3. **Yerel kimlik doğrulama:** API token rotasyonu; opsiyonel uygulama parola kilidi.
4. **Şifreleme:** API anahtarları/hassas ayarlar OS keychain'de (keyring); dosya bazlı şifreli ayar deposu.
5. **safetensors-only varsayılanı;** pickle (.ckpt/.pt) yüklemede karantina + açık kullanıcı onayı.
6. Bağımlılık taraması (pip-audit, npm audit) CI'da; SBOM üretimi; bütçe varsa üçüncü taraf güvenlik incelemesi.

**Performans adımları**
7. **ONNX dönüşüm hattı:** seçili araç modelleri (upscale, BG removal, OCR) için ONNX Runtime yolu.
8. **TensorRT engine cache:** SDXL/FLUX için opsiyonel derleme akışı (TensorRTAdapter'ın doldurulması) — ilk derleme uzun, sonrası hızlı; UI'da açık beklenti yönetimi.
9. torch.compile ve quantization (int8/fp8) bayrakları; model bazlı hız/kalite profilleri.
10. **Benchmark suite + regresyon takibi:** CI'da (self-hosted GPU) gecelik; soğuk başlangıç optimizasyonu (lazy import), büyük galerilerde UI sanallaştırma.

**DoD:** Tanımlı performans bütçeleri tutuyor (ör. uygulama açılışı < 3 sn; SDXL 1024² referans GPU'da hedef sürede); pickle model uyarısız yüklenemiyor; benchmark regresyonu PR'ı kırmızıya boyuyor.

---

### FAZ 15 — Paketleme, Dağıtım, Otomatik Güncelleme, Docker (4–6 hafta)

**Adımlar**
1. Gömülü Python stratejisinin finali: python-build-standalone + ilk açılışta uv ile donanıma uygun torch varyantının (CUDA/ROCm/CPU) kurulumu; tamamen çevrimdışı kurulum için "full offline bundle" varyantı.
2. **Windows:** NSIS/MSI + code signing (SmartScreen itibarı için EV sertifika değerlendirmesi).
3. **macOS:** .app + dmg, notarization, universal2 (Apple Silicon + Intel).
4. **Linux:** AppImage (birincil), .deb/.rpm, Flatpak.
5. **Auto-update:** Tauri updater + imzalı manifest; stable/beta/nightly kanalları; backend bağımlılık güncellemelerinin ayrık yönetimi.
6. **Docker:** headless sunucu imajı (REST + MCP, UI'sız) — `cuda / rocm / cpu` tag'leri; docker-compose örneği.
7. **Release pipeline (CI/CD finali):** tag → 3 OS build → imza/notarize → GitHub Releases → update manifest → Docker push.
8. **İlk çalıştırma sihirbazı:** donanım tespiti, model dizini seçimi, VRAM'e göre başlangıç modeli önerisi, örnek proje.

**DoD:** Üç OS'te temiz makinede kurulum → ilk görsel akışı sorunsuz; beta kanalındaki kullanıcı auto-update ile yeni sürüme geçer; Docker imajı UI'sız sunucuda API + MCP servis eder.

---

### FAZ 16 — Beta Programı, Dokümantasyon Finali ve v1.0 (6–8 hafta)

**Adımlar**
1. **Kapalı beta → açık beta:** geri bildirim döngüsü; opt-in, gönderilmeden önce kullanıcıya gösterilen çökme raporu.
2. **Dokümantasyon sitesinin tamamlanması:** mimari, kurulum, kullanıcı kılavuzu, API referansı, Python SDK, Plugin SDK, Workflow API, geliştirme rehberi, dağıtım, katkı rehberi; 5+ kısa eğitim videosu/GIF.
3. Erişilebilirlik geçişi (klavye navigasyonu, kontrast) + i18n temeli (EN + TR ile başla).
4. Haftalık RC ritmi: bilinen hata triage'ı, kararlılık sprintleri, "sıfır bloklayıcı" hedefi.
5. **v1.0 go/no-go listesi:** özellik matrisi %100 mü, destek matrisi dürüst mü, güvenlik maddeleri kapalı mı, benchmark'lar yeşil mi.
6. **v1.0 yayını** + yayın sonrası plan: semver politikası, hotfix süreci, v1.1+ yol haritası (topluluk RFC'leriyle).

**DoD:** Go/no-go listesi %100; kritik/bloklayıcı hata: 0; dokümantasyon sitesi tüm modülleri kapsıyor.

---

## 6. Sürekli (Cross-Cutting) İş Parçacıkları

Bu izler faz değildir; **her fazın tanımına gömülüdür** ve fazın DoD'sine dahildir.

- **Test:** her fazda birim + entegrasyon testi zorunlu. GPU testleri self-hosted runner'da gecelik. E2E (Playwright + tauri-driver) Faz 4'ten itibaren. Üretim determinizmi için golden-image benzerlik testleri (pHash/LPIPS eşiği). Faz 14'te stres/performans testleri eklenir.
- **CI/CD:** Faz 0'da doğar, her fazla büyür; Faz 15'te release pipeline ile tamamlanır.
- **Dokümantasyon:** "Faz, dokümanı yazılmadan bitmez" kuralı. Faz 16 yalnızca derleme ve cilalamadır, sıfırdan yazım değil.
- **Topluluk/OSS:** Alfa'dan itibaren haftalık dev-log, "good first issue" etiketleri, RFC süreci; Beta'dan itibaren Discord/forum.

## 7. Ekip ve Roller (öneri)

| Rol | Kişi | Ağırlıklı fazlar |
|---|---|---|
| Tech Lead / Mimar | 1 | 0–3, 10, 13, 14 |
| ML / Inference mühendisi | 2 | 3–9, 14 |
| Backend mühendisi | 1–2 | 1–3, 10–13 |
| Frontend / Desktop mühendisi | 1–2 | 1, 4–10, 13 |
| DevOps | 1 (yarı zamanlı olabilir) | 0, 15, sürekli CI |
| UX tasarımcı | 1 (ön yüklü) | 0–5, 10 |
| QA | rotasyonla | sürekli |

## 8. Risk Kaydı ve Önlemler

| Risk | Etki | Önlem |
|---|---|---|
| Python bağımlılık çakışmaları (diffusers/transformers sürüm cehennemi) | Yüksek | İzole env stratejisi (ADR-5), sürüm sabitleme, adapter soyutlaması, plugin-başına venv |
| VRAM çeşitliliği (4 GB laptop ↔ 24 GB masaüstü) | Yüksek | Donanım profilleri (low/med/high), otomatik offload/quantization, model önerileri |
| ROCm'un Windows'ta zayıflığı | Orta | ROCm'u Linux'ta birincil destekle; Windows AMD için DirectML opsiyonu; **dürüst destek matrisi** yayınla |
| macOS'ta CUDA yok | Orta | MPS + GGUF yolu; video modelleri için "sınırlı destek" beyanı |
| Model lisansları (FLUX-dev vb. ticari kısıt) | Orta | Lisans metadata + UI uyarısı; modelleri uygulamayla dağıtma |
| Kapsam patlaması | Yüksek | "Çekirdek mi, plugin mi?" karar ağacı; yeni model talepleri varsayılan olarak plugin'e |
| Paket boyutu (torch + CUDA GB'larca) | Orta | Slim installer + ilk açılışta varyant kurulumu; offline bundle ayrı |
| Pickle/plugin güvenliği | Yüksek | safetensors varsayılan, imzalı plugin, izin sistemi, süreç izolasyonu (Faz 13–14) |
| Uzun video işlerinin kesilmesi | Orta | Checkpoint/sürdürme, düşük öncelik sınıfı, ara önizleme |
| Tek kişiye bağımlılık (bus factor) | Orta | Modül sahiplikleri, ADR'ler, dokümante API sınırları |

## 9. MVP Tanımı ve Kesme Çizgileri

- **Alfa 0.1 (MVP):** Faz 0–4 → model yöneten, kuyruklayan, görüntü üreten kararlı masaüstü uygulaması.
- **Beta:** + Faz 5, 6, 10, 12 → araç seti, LLM/RAG, workflow editörü, API/MCP/SDK.
- **v1.0:** tüm fazlar. Kaynak isterlerdeki hiçbir madde silinmez; yalnızca ekip küçükse Faz 8, 9, 13 v1.0 sonrasına kaydırılabilir (bu karar Faz 6 sonunda gözden geçirilir).

## 10. İlk 30 Gün — Somut Başlangıç Backlog'u

1. Özellik matrisini çıkar ve dondur (hafta 1).
2. ADR-1…15'i yaz, tartış, kapat (hafta 1–2).
3. Monorepo + lint/format + pre-commit iskeleti (hafta 1).
4. CI: 3 OS CPU build + test matrisi yeşil (hafta 2).
5. Wireframe seti + design token'lar (hafta 2–3).
6. Tauri boş kabuk + React iskeleti (hafta 2).
7. FastAPI `/health` + `/system/info` + structured logging (hafta 2–3).
8. Sidecar lifecycle: başlat/izle/kapat + token pazarlığı (hafta 3–4).
9. SQLite + Alembic ilk migration (hafta 3).
10. Donanım algılama servisi + UI sistem paneli (hafta 4).
11. WS olay veriyolu + Console paneli (hafta 4).
12. dockview yerleşimi kaydet/yükle (hafta 4).
13. Tehdit modeli v0 + güvenlik ilkeleri dokümanı (hafta 3–4).
14. Faz 2–3 backlog'unun sprint'lere kırılması (hafta 4).

---

*Bu plan yaşayan bir dokümandır: her faz sonunda retrospektifle güncellenir; kapsam değişiklikleri yalnızca özellik matrisi + ADR üzerinden yapılır.*