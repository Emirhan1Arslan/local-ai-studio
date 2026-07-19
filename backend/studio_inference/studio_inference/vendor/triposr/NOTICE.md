Vendored from https://github.com/VAST-AI-Research/TripoSR (commit `3381600`, MIT License — see `LICENSE`).

Only the `tsr/` inference package is vendored (not `run.py`/`gradio_app.py`,
which are demo scripts this project doesn't use).

**Local patch:** `tsr/models/isosurface.py` — replaced the `torchmcubes`
CUDA marching-cubes call with `skimage.measure.marching_cubes` (CPU). This
machine has no NVIDIA CUDA Toolkit (`nvcc`) installed, so `torchmcubes`'s
CUDA extension cannot be compiled here (see docs/worker-envs.md, Faz 9).
The neural network inference (the VRAM-heavy part) still runs on GPU;
only this final geometry-extraction step runs on CPU.
