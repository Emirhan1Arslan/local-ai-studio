Vendored from https://github.com/HiDream-ai/HiDream-O1-Image (commit `2c2d29f`,
2026-06-22, MIT License — see `LICENSE`). Model weights (`HiDream-ai/HiDream-O1-Image`
and `-Dev`) are also MIT.

Only the `models/` inference package is vendored (renamed to `hidream_o1/` so it can be
imported as a package from `adapters/imagex.py`): `qwen3_vl_transformers.py` (the
Qwen3-VL-based pixel DiT — `Qwen3VLForConditionalGeneration` with the `x_pred` image
head), `pipeline.py` (`generate_image()` sampling loop), `flash_scheduler.py`,
`fm_solvers_unipc.py`, `utils.py`. Not vendored: `inference.py`/`app.py`/`prompt_agent.py`
(demo CLI, Flask web demo and the optional Gemma-4 "prompt agent" — the adapter passes
prompts through unchanged).

**Local patch:** `pipeline.py` — `generate_image()` gained a `use_flash_attn`
keyword (default: `True` only if a `flash_attn` kernel was importable). Upstream
hardcodes `"use_flash_attn": True` and tells users to hand-edit line 341 when flash-attn
is missing; `flash_attn` has no Windows wheel and its kernels do not target RTX 5090
(sm_120) — same situation as TRELLIS/CRM (see vendor/trellis/NOTICE.md). With the flag
off the model uses its built-in SDPA path (`use_flash_attn=False` branch in
`qwen3_vl_transformers.py`), same math, different kernel.

Requires `transformers==4.57.x` (upstream pin; `Qwen3VLConfig`, `check_model_inputs`,
`create_causal_mask` imports) and `diffusers` for the schedulers — both live in the
`envs/imagex` worker env (docs/worker-envs.md), NOT in `envs/image` whose pins are
frozen for SDXL/LTX.
