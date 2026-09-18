Vendored from https://github.com/hzwer/Practical-RIFE (MIT License — see `LICENSE`).

Files: `rife/IFNet_HDv3.py` (the v4.25 IFNet architecture, from the official
"v4.25 — 2024.09.19" model zip, Google Drive id `1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg`,
`train_log/IFNet_HDv3.py`) and `rife/warplayer.py` (`model/warplayer.py` from the repo,
main branch). Not vendored: `RIFE_HDv3.py`/`refine.py` (training wrapper: AdamW, DDP,
loss modules) — `adapters/videotool.py` builds `IFNet()` directly and runs the same
`forward(imgs, timestep, scale_list)` the wrapper's `inference()` used.

**Local patches:**
- `IFNet_HDv3.py`: `from model.warplayer import warp` → relative import; the module-level
  `device = torch.device(...)` constant removed (unused in inference; device is chosen by
  the caller).
- `warplayer.py`: the module-level cuda/mps/cpu `device` selection block removed
  (`warp()` already uses `tenFlow.device`).

**Weights:** upstream ships `flownet.pkl` (a pickled state_dict) only via Google Drive /
Baidu. For the safetensors-first policy the pkl was converted once
(`torch.load(weights_only=True)` → strip `module.` → `safetensors.torch.save_file`, fp32,
198 tensors, 24 MB) to `models/video/RIFE-4.25/flownet.safetensors` + `config.json`
(`{"arch": "IFNet_HDv3", "version": "4.25"}`) so the Model Manager lists it under `video`.
fp32 is kept on purpose: the coarse-to-fine flow is sensitive to fp16 (also noted by the
mlx-community port). Why not `rife-ncnn-vulkan-python`: PyPI has source-only dists needing
the Vulkan SDK + C++ toolchain (plan.md Faz 8 kalanı); this pure-torch path needs nothing
beyond `envs/image`.
