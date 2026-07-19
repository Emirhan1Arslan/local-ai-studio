from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from skimage import measure


def _marching_cubes_skimage(
    level: torch.FloatTensor, isovalue: float
) -> Tuple[torch.FloatTensor, torch.LongTensor]:
    """torchmcubes.marching_cubes'un CPU (scikit-image) ile yerine geçeni —
    Local AI Studio bu makinede NVIDIA CUDA Toolkit (nvcc) kurulu olmadığı için
    torchmcubes'un CUDA eklentisini derleyemiyor (bkz. docs/worker-envs.md,
    Faz 9). Sinir ağı çıkarımı (VRAM'in çoğunu kullanan kısım) yine GPU'da
    kalır; yalnızca bu son geometri-çıkarım adımı CPU'da çalışır."""
    volume = level.detach().cpu().numpy().astype(np.float64)
    verts, faces, _normals, _values = measure.marching_cubes(volume, level=isovalue)
    v_pos = torch.from_numpy(verts.astype(np.float32))
    t_pos_idx = torch.from_numpy(faces.astype(np.int64))
    return v_pos, t_pos_idx


class IsosurfaceHelper(nn.Module):
    points_range: Tuple[float, float] = (0, 1)

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        raise NotImplementedError


class MarchingCubeHelper(IsosurfaceHelper):
    def __init__(self, resolution: int) -> None:
        super().__init__()
        self.resolution = resolution
        self.mc_func: Callable = _marching_cubes_skimage
        self._grid_vertices: Optional[torch.FloatTensor] = None

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        if self._grid_vertices is None:
            # keep the vertices on CPU so that we can support very large resolution
            x, y, z = (
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
            )
            x, y, z = torch.meshgrid(x, y, z, indexing="ij")
            verts = torch.cat(
                [x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1
            ).reshape(-1, 3)
            self._grid_vertices = verts
        return self._grid_vertices

    def forward(
        self,
        level: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        level = -level.view(self.resolution, self.resolution, self.resolution)
        v_pos, t_pos_idx = self.mc_func(level.detach(), 0.0)
        v_pos = v_pos[..., [2, 1, 0]]
        v_pos = v_pos / (self.resolution - 1.0)
        return v_pos.to(level.device), t_pos_idx.to(level.device)
