# TRELLIS (vendored, with local patches)

Source: https://github.com/microsoft/TRELLIS
Commit: 442aa1e1afb9014e80681d3bf604e8d728a86ee7 (2025-11-05)
License: MIT (see `LICENSE`).

Submodule: `trellis/representations/mesh/flexicubes/` from
https://github.com/MaxtirError/FlexiCubes.git (commit `815e075`), license
Apache-2.0 (see `trellis/representations/mesh/flexicubes/LICENSE.txt`). Fetched
via `git submodule update --init` (the plain repo clone/copy doesn't populate
submodule content) and copied in — not a live git submodule in this repo.

Third-party dependency: `third_party/diff-gaussian-rasterization/` from
https://github.com/autonomousvision/mip-splatting.git's
`submodules/diff-gaussian-rasterization` (commit as of 2026-07-20 clone).
**License: Inria/MPII Gaussian-Splatting License — NON-COMMERCIAL, research/
evaluation use only** ("THE USER CANNOT USE, EXPLOIT OR DISTRIBUTE THE
SOFTWARE FOR COMMERCIAL PURPOSES", see `third_party/diff-gaussian-
rasterization/LICENSE.md`). Same treatment as OpenLRM's CC-BY-NC-4.0 — shown
as a UI warning in `MeshPanel.tsx`, not gated further.

## Why this dependency (not in TRELLIS's own eager imports)

`trellis/renderers/gaussian_render.py`'s top-level imports are pure PyTorch
(no CUDA extension) — this made it LOOK like TRELLIS doesn't need mip-
splatting at all. It does: `GaussianRenderer.render()` does a **lazy**
`from diff_gaussian_rasterization import GaussianRasterizer` inside the
function body, only hit when actually rendering a Gaussian (which
`postprocessing_utils.to_glb()` does 100x per call, to bake mesh texture
from the Gaussian appearance representation). Missed on first read of the
import list; found via the actual traceback during a live test.

## Local patches

### 1. `sdpa` ported to sparse (voxel) attention — `trellis/modules/sparse/attention/`

This machine has neither a working `flash_attn` (no official Windows wheel)
nor a working `xformers` (its compiled kernels reject the RTX 5090:
`requires device with capability <= (9, 0) but your GPU has capability
(12, 0) (too new)`) — confirmed by direct testing, not assumed. TRELLIS's
**dense** attention module (`trellis/modules/attention/full_attn.py`)
already ships a working `'sdpa'` backend option (using
`torch.nn.functional.scaled_dot_product_attention`); the **sparse** module
only accepted `'xformers'`/`'flash_attn'` and hard-`raise`d otherwise. Ported
the same approach (pad packed variable-length sequences to a batched dense
tensor + boolean mask, call `sdpa`, unpad) to:
- `sparse/attention/full_attn.py` — used by the flow-matching (diffusion)
  models, `attn_mode="full"` (the class default, confirmed unset —
  therefore default — in the actual downloaded `TRELLIS-image-large`
  checkpoint configs).
- `sparse/attention/windowed_attn.py` — used by `slat_decoder_mesh`
  (**not** `"full"` — found by an actual `UnboundLocalError` crash, NOT
  assumed; both the uniform-window and variable-length-window code paths
  needed an `sdpa` branch).
- `sparse/attention/serialized_attn.py` — only got an import-time guard
  (`elif ATTN == 'sdpa': pass`) to stop it crashing on load; never actually
  exercises the `xformers`/`flash_attn`-only function body in this pipeline,
  so no real `sdpa` implementation was needed there (unlike
  `windowed_attn.py`, where the assumption "never called" turned out wrong).
- `sparse/__init__.py` — its `SPARSE_ATTN_BACKEND`/`ATTN_BACKEND` validity
  check only accepted `['xformers', 'flash_attn']`; added `'sdpa'`.

This is a mechanical backend swap (same attention math, different kernel —
`sdpa` is mathematically equivalent to `flash_attn`/`xformers`, just a
different implementation of the identical scaled-dot-product-attention
operation), not a guess about model weights or architecture — unlike
Hunyuan3D-2's IP-Adapter checkpoint mismatch (see `vendor/hunyuan3d/
NOTICE.md`), which was correctly left unpatched because reconstructing it
would have required guessing at undocumented trained behavior.

**`xformers` was uninstalled from `envs/3d` entirely** after finding it
doesn't just fail to help — it actively breaks things: `facebookresearch/
dinov2`'s own torch.hub code (used as TRELLIS's `image_cond_model`)
unconditionally tries `memory_efficient_attention` whenever `xformers` is
importable, hitting the same sm_120-unsupported error, with **no way to
redirect it via TRELLIS's own env vars** (it's Meta's code, not TRELLIS's).
dinov2's `attention.py` has its own clean `try/except ImportError` fallback
to standard attention — removing `xformers` from the env lets that fallback
engage correctly.

### 2. `kaolin` dependency removed — `flexicubes/flexicubes.py`

`kaolin` has no prebuilt wheel for this machine's torch/CUDA combo (its own
index only lists up to `torch-2.4.0_cu121`; ours is `2.11.0+cu128`) and
building it from source on Windows is high-risk/unproven. Turned out
unnecessary: `trellis/` package code doesn't import kaolin anywhere — only
the vendored FlexiCubes submodule did, for exactly one function,
`kaolin.utils.testing.check_tensor(tensor, shape, throw=False)`, used
purely inside `assert` statements as an input-shape validator (no bearing
on computation). Replaced with an ~8-line local equivalent at the top of
`flexicubes.py`.

### 3. `torch/extension.h` + nvcc + MSVC 14.44 ambiguous-`std` bug (again)

Same bug and same fix as `custom_rasterizer_kernel` (see `vendor/hunyuan3d/
NOTICE.md` for the full root-cause writeup) — any `.cu` file compiled via
nvcc that pulls in `torch/extension.h` fails with `error C2872: 'std':
belirsiz simge` on this MSVC 14.44 + CUDA 12.8 combo. Hit again in
`third_party/diff-gaussian-rasterization/rasterize_points.cu` (and its
header `rasterize_points.h`, included by both the `.cu` file and by
`ext.cpp`). Fix: narrowed both to `<ATen/ATen.h>` + `<torch/types.h>`;
`ext.cpp` (the one `.cpp` file with the `PYBIND11_MODULE`, compiled by
`cl.exe` directly, never touched by nvcc) already had its own explicit
`#include <torch/extension.h>` and needed no change. Also built from a
short path (`C:\dgr_build\`, not the deeply-nested scratchpad clone path)
to avoid the same `LINK : fatal error LNK1104` MAX_PATH issue found with
`custom_rasterizer_kernel` — build-time-only workaround, doesn't affect
where the vendored source lives.

The core CUDA rasterizer files (`cuda_rasterizer/*.cu`/`*.h`) don't
reference `torch` at all — confirmed by grep before patching — so only the
PyTorch-facing wrapper (`rasterize_points.cu`/`.h`) needed the fix.

### 4. `trellis/pipelines/__init__.py` — text-to-3D import dropped

Upstream eagerly imports both `TrellisImageTo3DPipeline` and
`TrellisTextTo3DPipeline`; the latter needs `open3d` (~500MB, otherwise
unused here). This adapter is image-to-3D only (same convention as all
five other mesh families), so the text pipeline import was removed rather
than installing `open3d` for a feature never called.

### 5. `hunyuan3d-paint-v2-0-turbo`-style checkpoint mismatch — NOT hit here

Unlike Hunyuan3D-2's texgen, every TRELLIS checkpoint load in this pipeline
(`ss_flow_img_dit`, `slat_flow_img_dit`, `ss_dec_conv3d`, `slat_decoder_
mesh`, `slat_decoder_gs`) succeeded on the first real attempt, `strict=True`
and all — no reverse-engineering of undocumented architecture was needed.

## envs/3d dependencies added

`spconv-cu120` (sparse convolutions — the one genuinely CUDA-Toolkit-
independent-at-runtime piece, ships its own bundled CUDA 12.0 runtime,
confirmed working via a real `SubMConv3d` forward pass), `easydict`,
`pyvista`, `pymeshfix`, `igraph`, `utils3d` (git dependency, pinned commit),
`diff-gaussian-rasterization` (built from source, see above). `torch`/
`torchvision` were shadowed and fixed (uninstall + reverify
`torch.cuda.is_available()`) after both the `xformers` install attempt and
its later uninstall — the recurring risk documented in `adapters/mesh.py`'s
module docstring applies to `uv pip uninstall`-triggered re-resolution too,
not just installs.

## Canlı doğrulama (gerçek `/v1/jobs` API'si üzerinden)

`num_inference_steps=25`, `formats=["mesh", "gaussian"]`,
`texture_size=1024`, `simplify=0.95` → 10,107 vertices / 15,848 faces, real
baked `TextureVisuals` material, confirmed via `trimesh.load(...).visual`.
Peak VRAM (warm model, second call): 11,583MB. Inference-only time (warm
model, excludes ~130s one-time `facebookresearch/dinov2` hub download):
~28s, dominated by the 100-view Gaussian render + texture bake in `to_glb`
(~18s) rather than the diffusion sampling itself (~7s for both
sparse-structure and structured-latent stages combined).
