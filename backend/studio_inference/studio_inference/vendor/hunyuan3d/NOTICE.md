# Hunyuan3D-2 (vendored subset)

Source: https://github.com/Tencent-Hunyuan/Hunyuan3D-2
Commit: f8db63096c8282cb27354314d896feba5ba6ff8a (2025-10-28)
License: TENCENT HUNYUAN 3D 2.0 COMMUNITY LICENSE AGREEMENT (see `LICENSE` in this
directory). Non-standard license, NOT MIT/Apache. Key points:
- Territory-restricted: does NOT apply in the EU, UK, or South Korea.
- Free for use with <1M monthly active users; beyond that a separate license must be
  requested from Tencent (hunyuan3d@tencent.com).
- Redistribution of the works (or Model Derivatives) must be accompanied by the
  required "Notice" text and must not misrepresent Tencent affiliation/endorsement.
- Output usage must comply with Tencent's Acceptable Use Policy (Exhibit A of LICENSE).

## What's included

`hy3dgen/shapegen/` (image → untextured mesh, diffusion + VAE decode), the top-level
`hy3dgen/__init__.py` stub, and `hy3dgen/texgen/` + `hy3dgen/rembg.py` (the
paint/texture pipeline — see "Texture/paint status" below for why it's vendored but
NOT wired into `adapters/mesh.py` yet).

## CUDA Toolkit (installed 2026-07-20)

The NVIDIA CUDA Toolkit 12.8 (`nvcc`) was installed on this machine (user-run silent
install, `Display.Driver` component excluded — driver stayed at 576.88 throughout).
This unblocked `nvdiffrast` (confirmed working: `RasterizeCudaContext` + `rasterize()`
JIT-compile and run) and `custom_rasterizer_kernel` (below) — see docs/worker-envs.md
"Faz 9 — CUDA Toolkit kuruldu" for the install command and verification steps.

### Local patches: `texgen/custom_rasterizer` (two files)

Building `custom_rasterizer_kernel` initially failed with `error C2872: 'std':
belirsiz simge` (ambiguous symbol) inside `torch/csrc/dynamo/compiled_autograd.h`,
reproduced with a MINIMAL `.cu` file containing only `#include <torch/extension.h>`
— confirmed **systemic**: any `.cu` file compiled via nvcc (MSVC 14.44.35207 host
compiler + CUDA 12.8's bundled libcu++) that pulls in `torch/extension.h` hits this.
`nvdiffrast` avoids it entirely: its own `.cu` files stay narrow (no torch headers),
and the only file needing `torch/extension.h` (`csrc/torch/torch_bindings.cpp`, for
the `PYBIND11_MODULE` macro + pybind casters) is a plain `.cpp`, compiled directly by
`cl.exe` — never passed through nvcc's host-pass. Applied the same split here:
- `lib/custom_rasterizer_kernel/rasterizer.h`: `#include <torch/extension.h>` →
  `#include <ATen/ATen.h>` + `#include <torch/types.h>` (still provides
  `torch::Tensor`/`torch::TensorOptions`, without dynamo/autograd headers).
- `lib/custom_rasterizer_kernel/rasterizer.cpp` (has the `PYBIND11_MODULE`, compiled
  by plain `cl.exe` — immune to the bug): added back an explicit
  `#include <torch/extension.h>` at the top, since it lost the one `rasterizer.h` used
  to provide transitively.

Separately: building from a deeply-nested path failed with `LINK : fatal error
LNK1104: ... .lib dosyası açılamıyor` — the `/IMPLIB:` output path was 283 characters,
over Windows' MAX_PATH. Fix: copy the extension source to a short path (e.g.
`C:\cr_build\`) before `pip install --no-build-isolation --no-deps .`; the built
`.pyd` lands in site-packages regardless of build-dir location, so this is a
build-time-only workaround, not a permanent vendor-path requirement.

`differentiable_renderer`'s `mesh_processor` (pybind11, no CUDA) built cleanly with
no patches needed.

### envs/3d dependency changes for texgen

`diffusers` downgraded from `0.39.0` (installed for shapegen) to `0.32.2` — 0.39.0's
`diffusers.models.autoencoders` eagerly imports `AutoencoderRAE`, which needs
`transformers.Dinov2WithRegistersConfig` (very recent transformers, would have forced
transformers 5.x — too large a jump, high regression risk for TripoSR/SF3D/OpenLRM's
CLIP/DINO encoders). `0.32.2` is the SAME version already proven compatible with
`transformers==4.46.3` in `envs/image` (see docs/worker-envs.md) — reused that known-
good pair instead of chasing versions upward. `transformers` bumped `4.42.3` → `4.46.3`
(needed for `EncoderDecoderCache`, required by `peft>=0.17` which `diffusers` itself
version-gates) + `tokenizers` → `0.20.3` (transformers 4.46.3's own floor). Re-verified
shapegen (`Hunyuan3DDiTFlowMatchingPipeline`) still imports fine after the diffusers
downgrade. **Every one of these installs was followed by a `torch.cuda.is_available()`
check** — `accelerate`/`peft` resolution pulled unpinned CPU-only `torch` TWICE during
this (recurring bug, see `adapters/mesh.py` docstring); both times fixed with
`uv pip uninstall torch torchvision` and reconfirmed `2.11.0+cu128, True`.

### Texture/paint status: blocked by an upstream checkpoint/code mismatch (NOT CUDA)

`hunyuan3d-paint-v2-0-turbo/unet/modules.py`'s `UNet2p5DConditionModel.from_pretrained`
calls `unet.load_state_dict(ckpt, strict=True)` on a freshly-constructed UNet, but the
released `.safetensors`/`.bin` checkpoint contains 36 extra IP-Adapter keys
(`image_proj_model.{norm,proj}.{weight,bias}` + 32×
`*.attn2.to_{k,v}_ip.weight`) that the constructed model never creates — `modules.py`
never attaches an image-projection module or IP-Adapter attention processors before
the strict load. **Confirmed NOT a local/version issue**: fetched the current `main`
branch copy of this exact file from
https://huggingface.co/tencent/Hunyuan3D-2/raw/main/hunyuan3d-paint-v2-0-turbo/unet/modules.py
— byte-identical logic, same bug, still unpatched upstream as of 2026-07-20. Also
found (unrelated but same-shaped issue in the same repo):
https://github.com/Tencent-Hunyuan/Hunyuan3D-2/issues/280 (ShapeVAE missing-key
mismatch). Correctly reconstructing the missing IP-Adapter wiring would mean guessing
at Tencent's undocumented internal architecture from tensor shapes alone — a
`strict=False` workaround would "succeed" but silently drop the trained IP-Adapter
conditioning, producing plausible-but-wrong textures with no visible error. Not
attempted. `family == "hunyuan3d"` in `adapters/mesh.py` therefore stays shape-only;
`hy3dgen/texgen` is vendored (extensions build and the delight sub-pipeline loads
fine) but not called from the adapter.

One already-applied, unrelated local patch, needed regardless of the above:
`hunyuan3d-paint-v2-0-turbo/unet/modules.py` **on disk in the Model Manager download**
(`models/3d/Hunyuan3D-2/hunyuan3d-paint-v2-0-turbo/unet/modules.py`, NOT the vendored
copy here) hardcoded loading `diffusion_pytorch_model.bin` (pickle); we only downloaded
the `.safetensors` variant per the project's safetensors-preferred policy. Patched to
try `.safetensors` first via `safetensors.torch.load_file`, falling back to `.bin`.

## Local notes

- `hunyuan3d-vae-v2-0` (downloaded separately via Model Manager) is redundant for
  this adapter: the `hunyuan3d-dit-v2-0-fast/model.fp16.safetensors` checkpoint
  already bundles `model.*`, `vae.*`, and `conditioner.*` weights together (confirmed
  by inspecting the safetensors key prefixes). Only the `hunyuan3d-dit-v2-0-fast`
  folder's `config.yaml` + `model.fp16.safetensors` are read.
- `smart_load_model()`'s local-cache path convention (`~/.cache/hy3dgen/<repo>/<subfolder>`)
  is bypassed entirely — `adapters/mesh.py` calls
  `Hunyuan3DDiTFlowMatchingPipeline.from_single_file(ckpt_path, config_path, ...)`
  directly with the Model Manager's actual on-disk paths.
