Vendored from https://github.com/Stability-AI/stable-fast-3d (MIT-style repo
structure, but see `LICENSE.md` — **Stability AI Community License**, not
MIT/Apache). Only the `sf3d/` inference package and `load/tets/160_tets.npz`
(required data file for marching-tetrahedra isosurface extraction) are
vendored — not `run.py`/`comfyui/` demo code.

**LİSANS UYARISI:** Hem kod hem ağırlıklar Stability AI Community License
ile geliyor — araştırma/kişisel kullanım serbest, yıllık geliri 1M USD'yi
aşan kuruluşlar için ticari kullanım ayrı bir Enterprise License gerektirir.
UI, bu motoru seçmeden önce bunu kullanıcıya XTTS-v2 ile aynı desende
göstermeli (adapters/audio.py'deki emsal).

No local patches needed: SF3D uses marching TETRAHEDRA (pure PyTorch, the
precomputed `160_tets.npz` grid), not marching cubes — unlike TripoSR it
never needed `torchmcubes`, so there was nothing to patch. Its two compiled
extensions (`texture_baker`, `uv_unwrapper`) built cleanly as CPU (MSVC)
extensions on this machine (no CUDA Toolkit installed, same constraint as
documented in docs/worker-envs.md Faz 9 — but unlike torchmcubes these two
gracefully fall back to `CppExtension` instead of hard-requiring CUDA).
