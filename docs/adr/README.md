# Architecture Decision Records

Format: her karar tek dosya — **Bağlam → Karar → Gerekçe → Sonuçlar**. Durumlar: `accepted` / `superseded` / `deprecated`.
Kapsam değişiklikleri yalnızca özellik matrisi + ADR üzerinden yapılır (bkz. plan.md §10).

| # | Karar | Durum |
|---|---|---|
| [0001](0001-desktop-framework.md) | Tauri 2 + React 18 + TypeScript (Vite) | accepted |
| [0002](0002-backend-sidecar.md) | Python backend sidecar süreç | accepted |
| [0003](0003-ui-backend-iletisim.md) | localhost REST + WebSocket, tek kullanımlık token | accepted |
| [0004](0004-worker-izolasyonu.md) | Her ağır iş ayrı worker sürecinde | accepted |
| [0005](0005-python-ortam-stratejisi.md) | uv çekirdek env + izole model env'leri | accepted |
| [0006](0006-veritabani.md) | SQLite + SQLAlchemy 2 + Alembic + sqlite-vec | accepted |
| [0007](0007-model-formatlari.md) | safetensors varsayılan; pickle karantina | accepted |
| [0008](0008-llm-runtime.md) | llama.cpp birincil + Transformers + Ollama köprüsü | accepted |
| [0009](0009-goruntu-video-pipeline.md) | Diffusers tabanlı adapter katmanı | accepted |
| [0010](0010-node-editor.md) | React Flow + backend tipli graph executor | accepted |
| [0011](0011-3d-viewer.md) | three.js (React Three Fiber) | accepted |
| [0012](0012-dockable-ui.md) | dockview + Tailwind + Radix/shadcn | accepted |
| [0013](0013-paketleme.md) | Tauri bundler + python-build-standalone + ilk açılışta torch varyantı | accepted |
| [0014](0014-gpu-destek-matrisi.md) | CUDA birincil · ROCm Linux · MPS · CPU fallback · DirectML ops. | accepted |
| [0015](0015-lisans.md) | Çekirdek Apache-2.0; modeller dağıtılmaz | accepted |
