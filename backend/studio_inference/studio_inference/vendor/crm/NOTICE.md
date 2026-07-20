# CRM (Convolutional Reconstruction Model, vendored with local patches)

Source: https://github.com/thu-ml/CRM
License: MIT (see `LICENSE`).

## What's included

- `imagedream/` — a fully self-contained "ldm"-style (latent-diffusion-model)
  package for the two-stage multi-view diffusion (pixel-space + CCM/XYZ
  coordinate-space). Notably does **not** depend on the `diffusers` library
  for its model definitions (own UNet/attention/scheduler implementations,
  config-driven via `omegaconf` — same lineage as the original CompVis/
  Stability `stable-diffusion` repo), which sidestepped the `diffusers`
  version-pin conflicts that came up with Hunyuan3D-2's texgen and
  InstantMesh's zero123plus.
- `model/` — the CRM reconstruction model (triplane + SDF/RGB MLPs +
  FlexiCubes-based mesh extraction, its own vendored `util/flexicubes.py`
  copy, independent of TRELLIS's).
- `libs/`, `util/`, `configs/`, `pipelines.py` (`TwoStagePipeline`),
  `inference.py` (`generate3d`, the nvdiffrast-based UV-texture export).
- `crm_mesh.py` — upstream's own top-level `mesh.py`, **renamed** to avoid
  shadowing this project's `adapters/mesh.py` (both would otherwise be
  importable as a bare `mesh` module once `vendor/crm` is on `sys.path`).
  Only one reference needed updating: `inference.py`'s `from mesh import
  Mesh` → `from crm_mesh import Mesh`.

No `nvdiffrast`/CUDA-Toolkit blocker here — `inference.py` unconditionally
creates a `dr.RasterizeCudaContext()` for the final UV-texture-baking step,
but by this point in the kademeli liste nvdiffrast was already proven
working (see `vendor/hunyuan3d/NOTICE.md`, `vendor/instantmesh/NOTICE.md`).

## Local patches

### 1. `xformers` hard-`assert` — `imagedream/ldm/modules/attention.py`

Same root cause as TRELLIS (see `vendor/trellis/NOTICE.md`): this machine
has no working `xformers` (its compiled kernels reject the RTX 5090 —
`requires device with capability <= (9, 0) but your GPU has capability
(12, 0) (too new)`) or `flash_attn` (no official Windows wheel). Unlike
TRELLIS's sparse attention, `imagedream`'s `BasicTransformerBlock.__init__`
had a **hard, unconditional** `assert XFORMERS_IS_AVAILBLE` — there was no
config path to opt into a different attention backend at all. Fixed by:
- Removing the assert.
- Adding `_memory_efficient_attention(q, k, v, attention_op)` next to
  `MemoryEfficientCrossAttention`, which uses `xformers.ops.
  memory_efficient_attention` if `XFORMERS_IS_AVAILBLE` else falls back to
  `torch.nn.functional.scaled_dot_product_attention`.

This case was **simpler than TRELLIS's** sparse-attention patch: `q`/`k`/`v`
here are already reshaped to `[B*heads, N, dim_head]` (a uniform, non-
variable-length batch) before the attention call — `sdpa` treats the
leading dimension as a batch dim natively, so no packing/padding/masking
was needed, just a direct drop-in replacement of the function call.

`imagedream/ldm/modules/diffusionmodules/model.py` (the VAE) has its own,
separate xformers usage but was **already correctly guarded** (`if
XFORMERS_IS_AVAILBLE and attn_type == "vanilla": attn_type =
"vanilla-xformers"` — falls through to the plain `AttnBlock` otherwise) —
confirmed by reading, not patched.

`model/archs/unet.py` (`UNetPP`, wrapping a `diffusers.UNet2DModel`) called
`self.unet.enable_xformers_memory_efficient_attention()` unconditionally at
construction time with no availability check — this call was simply
removed/commented; without it, diffusers' `UNet2DModel` uses its own
default attention processor (`AttnProcessor2_0`, PyTorch's native `sdpa`),
which is what we want anyway.

### 2. Gated HuggingFace repo — `model/crm/model.py`

`CRM.__init__` did `DDIMScheduler.from_pretrained("stabilityai/
stable-diffusion-2-1-base", subfolder="scheduler")` — fetching only a tiny
scheduler *config* (hyperparameters, not weights). This repo returns
**HTTP 401** even for an anonymous, unauthenticated request (verified with
a plain `curl`, not just via the Python client — this is a real access
change on HuggingFace's side, not a local auth/token issue: `HF_TOKEN`/
`HUGGING_FACE_HUB_TOKEN` were both unset in this environment). Fixed by
constructing `DDIMScheduler` directly with SD2.1-base's well-documented,
widely-replicated standard hyperparameters (`beta_start=0.00085,
beta_end=0.012, beta_schedule="scaled_linear", prediction_type="epsilon",
clip_sample=False, set_alpha_to_one=False, steps_offset=1`) instead of
fetching them — no network call needed for a static, publicly-known config.

### 3. `kiui` package bug (not vendor code — a local `envs/3d` site-packages fix)

`crm_mesh.py` (upstream's `mesh.py`) does `from kiui.op import
safe_normalize, dot`. The installed `kiui==0.3.5` package's own
`kiui/typing.py` (meant to be a small shim re-exporting `Union`/`Tensor`/
`ndarray`/etc. for use across the package via `from kiui.typing import *`)
was **essentially empty** (one comment line) — a genuine upstream packaging
bug, not something in our vendored code. `kiui/op.py` uses `Union`,
`Tensor`, `ndarray`, and `Literal` in type hints, all `NameError`s at
import time. Fixed by writing the obviously-intended content into
`envs/3d/Lib/site-packages/kiui/typing.py` directly (this is a local
environment file, not shared/redistributed vendor source — same category
as the `hunyuan3d-paint-v2-0-turbo/unet/modules.py` safetensors patch in
`vendor/hunyuan3d/NOTICE.md`, which also lives outside this repo's vendor
tree).

## envs/3d dependencies added

`kiui`, `absl-py` (imported by `libs/base_utils.py` for logging, unrelated
to CRM's actual model code), `wandb` (imported unconditionally by the same
file for experiment tracking — also training-only, but easier to install
than to patch out of an otherwise-unused import path). None of these
installs shadowed the protected system `torch` (checked after each).
