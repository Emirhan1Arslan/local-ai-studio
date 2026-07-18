"""Donanım algılama servisi: CPU/RAM/disk + GPU backend tespiti (CUDA/ROCm/MPS/CPU).

Torch'a bağımlı DEĞİLDİR (çekirdek env slim kalır — ADR-0013): NVIDIA için nvidia-smi,
AMD için rocm-smi sorgulanır; ikisi de yoksa CPU fallback raporlanır.
"""

from __future__ import annotations

import platform
import shutil
import subprocess

import psutil
from pydantic import BaseModel

from studio_core.config import get_settings


class GpuInfo(BaseModel):
    index: int
    name: str
    vram_total_mb: int
    vram_used_mb: int
    backend: str  # cuda | rocm


class SystemInfo(BaseModel):
    os: str
    os_version: str
    arch: str
    python: str
    cpu_count: int
    ram_total_mb: int
    ram_available_mb: int
    disk_free_gb: float
    gpus: list[GpuInfo]
    gpu_backend: str  # cuda | rocm | mps | cpu
    data_dir: str


def _detect_nvidia() -> list[GpuInfo]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            gpus.append(
                GpuInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    vram_total_mb=int(float(parts[2])),
                    vram_used_mb=int(float(parts[3])),
                    backend="cuda",
                )
            )
    return gpus


def _detect_rocm() -> list[GpuInfo]:
    if not shutil.which("rocm-smi"):
        return []
    try:
        out = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--csv"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    gpus = []
    for i, line in enumerate(out.strip().splitlines()[1:]):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[1].isdigit():
            gpus.append(
                GpuInfo(
                    index=i,
                    name=parts[0],
                    vram_total_mb=int(parts[1]) // (1024 * 1024),
                    vram_used_mb=int(parts[2]) // (1024 * 1024),
                    backend="rocm",
                )
            )
    return gpus


def get_system_info() -> SystemInfo:
    settings = get_settings()
    settings.ensure_dirs()
    gpus = _detect_nvidia() or _detect_rocm()
    if gpus:
        backend = gpus[0].backend
    elif platform.system() == "Darwin" and platform.machine() == "arm64":
        backend = "mps"
    else:
        backend = "cpu"

    vm = psutil.virtual_memory()
    disk = psutil.disk_usage(str(settings.data_dir))
    return SystemInfo(
        os=platform.system(),
        os_version=platform.version(),
        arch=platform.machine(),
        python=platform.python_version(),
        cpu_count=psutil.cpu_count(logical=True) or 1,
        ram_total_mb=vm.total // (1024 * 1024),
        ram_available_mb=vm.available // (1024 * 1024),
        disk_free_gb=round(disk.free / (1024**3), 1),
        gpus=gpus,
        gpu_backend=backend,
        data_dir=str(settings.data_dir),
    )
