# InstantMesh (vendored subset)

Source: https://github.com/TencentARC/InstantMesh
Commit: 08822c52fdc399b93ea00e4fa9e596344ed52ccc (2025-01-03)
License: Apache License 2.0 (see `LICENSE` in this directory). Permissive, no
commercial-use restriction, no UI warning needed (same treatment as TripoSR).

## What's included

- `src/` — the LRM+FlexiCubes reconstruction model (`src/models/lrm_mesh.py`'s
  `InstantMesh` class) and its geometry/renderer/encoder submodules, plus
  `src/utils/{camera_util,infer_util,mesh_util,train_util}.py`.
- `zero123plus/` — the multi-view diffusion pipeline (`Zero123PlusPipeline`, a
  `diffusers.StableDiffusionPipeline` subclass) and its custom UNet wrapper
  (`RefOnlyNoisedUNet`). Vendored locally instead of using upstream's
  `custom_pipeline="zero123plus"` (a bare string, which makes diffusers fetch the
  file from HF's `diffusers/examples/community` at runtime) — same "prefer local
  vendor code over a network fetch at load time" rule applied to OpenLRM/Hunyuan3D-2.
- `configs/instant-mesh-base.yaml` — model hyperparameters (encoder/transformer dims,
  FlexiCubes grid resolution) + default `infer_config` (texture_resolution,
  render_resolution).

No local code patches — used exactly as upstream designed it.

## What this needed that earlier families didn't

- `nvdiffrast` (see `vendor/hunyuan3d/NOTICE.md`'s "CUDA Toolkit" section, and
  `adapters/mesh.py`'s module docstring) — `src/models/lrm_mesh.py` imports it at
  module level, but `extract_mesh(use_texture_map=False)` (the default, vertex-
  colored path) never actually constructs a `RasterizeCudaContext` — only
  `use_texture_map=True` (UV-mapped texture output, via `xatlas_uvmap`) does. So
  `family="instantmesh"` works even if nvdiffrast's JIT compile were somehow
  unavailable, as long as `export_texmap` stays False.
- `zero123plus-v1.2` (`sudo-ai/zero123plus-v1.2`, ~4.3GB, standard diffusers SD1.5-
  shaped UNet+VAE+CLIP folder) — a full separate base model beyond InstantMesh's own
  `TencentARC/InstantMesh` weights (`diffusion_pytorch_model.bin` — the white-
  background UNet fine-tune loaded on top of zero123plus's base UNet — and
  `instant_mesh_base.ckpt`, the LRM reconstruction weights). Downloaded via the real
  Downloads Manager (`POST /v1/downloads/snapshot`, `subdir="3d"`) — it auto-selected
  only `.safetensors` variants, no manual pickle-quarantine needed this time.
- FlexiCubes itself (`src/models/geometry/rep_3d/flexicubes.py`) is a **fully
  vendored, pure-PyTorch implementation** bundled by InstantMesh's own authors — no
  extra `flexicubes` pip package or CUDA extension needed for it specifically.

## Local notes

- `run.py`'s `--diffusion_steps` default is 75 (zero123plus sampling) — high for an
  interactive UI; `adapters/mesh.py` exposes it as `num_inference_steps` with a lower
  UI default, same tradeoff already made for Hunyuan3D-2's `num_inference_steps`.
- `--export_texmap` defaults to False upstream (vertex-colored mesh, faster, no
  nvdiffrast rasterization needed) — kept as the adapter default too; exposed as
  `export_texmap` for users who want a real UV-mapped texture (uses nvdiffrast).
