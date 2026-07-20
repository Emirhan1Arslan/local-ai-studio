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

Only `hy3dgen/shapegen/` (image → untextured mesh, diffusion + VAE decode) and the
top-level `hy3dgen/__init__.py` stub. No local code patches — used exactly as
upstream designed it, loaded via `Hunyuan3DDiTFlowMatchingPipeline.from_single_file()`.

## What's excluded and why

`hy3dgen/texgen/` (the paint/texture pipeline) is deliberately NOT vendored.
`hy3dgen/texgen/custom_rasterizer/setup.py` builds `custom_rasterizer_kernel` as a
`torch.utils.cpp_extension.CUDAExtension` (compiling `rasterizer_gpu.cu` via nvcc) and
`custom_rasterizer/render.py` does an unconditional `import custom_rasterizer_kernel` —
there is no CPU fallback path (unlike `texture_baker`/`uv_unwrapper` in the
`stablefast3d` vendor dir, which check `CUDA_HOME is not None` and degrade to CPU
kernels). This is the exact same class of blocker as `nvdiffrast` for
InstantMesh/TRELLIS/CRM (see `adapters/mesh.py` module docstring): it requires the
NVIDIA CUDA Toolkit (`nvcc`) to be installed for JIT/build-time compilation, which is
not installed on this machine and is treated as a system-level change requiring
explicit user confirmation before installing.

`hy3dgen/texgen/differentiable_renderer/setup.py` (`mesh_processor`, a pybind11 CPU
extension) has no CUDA dependency itself, but it only supports the texture-baking
step of the paint pipeline, which cannot run without `custom_rasterizer_kernel`
regardless — so vendoring it alone would not unlock anything.

Net effect: this machine can run Hunyuan3D-2's shape generation (family `"hunyuan3d"`
in `adapters/mesh.py`) but not its texture/paint stage. Generated meshes are
untextured (vertex-color-free) GLB, same as TripoSR's raw output before any
paint/texture pass.

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
