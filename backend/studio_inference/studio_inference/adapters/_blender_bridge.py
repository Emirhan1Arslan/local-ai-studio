"""Blender köprüsü — FBX export (meshtool.py). Blender bu projeyle birlikte
kurulmaz (büyük bir masaüstü uygulaması); kullanıcının kendi kurulumu
aranır (Ollama köprüsüne benzer, opsiyonel dış bağımlılık deseni).

DİKKAT: Bu modül worker env'inde çalışır. 3.11+ sözdizimi KULLANMA.
"""

from __future__ import annotations

import glob
import os
import subprocess
from pathlib import Path

_SCRIPT = str(Path(__file__).resolve().parent / "_blender_scripts" / "mesh_to_fbx.py")


def find_blender_exe() -> str:
    """Sıra: STUDIO_BLENDER_PATH ortam değişkeni → PATH'teki 'blender' →
    Program Files altında yaygın kurulum yolu (sürüm-bağımsız glob)."""
    override = os.environ.get("STUDIO_BLENDER_PATH", "")
    if override and Path(override).is_file():
        return override

    from shutil import which

    on_path = which("blender")
    if on_path:
        return on_path

    for base in (
        r"C:\Program Files\Blender Foundation\Blender *\blender.exe",
        r"C:\Program Files (x86)\Blender Foundation\Blender *\blender.exe",
    ):
        matches = sorted(glob.glob(base), reverse=True)  # en yeni sürüm önce
        if matches:
            return matches[0]

    return ""


def export_fbx(input_path: str, output_path: str, timeout: float = 120.0) -> None:
    """input_path (glb/gltf/obj/ply/stl) -> output_path (.fbx). Blender kurulu
    değilse anlaşılır bir hata verir (RIFE gibi sessizce atlanmaz — kullanıcı
    FBX istediyse Blender kurmalı, docs/worker-envs.md'de kurulum notu var)."""
    blender = find_blender_exe()
    if not blender:
        raise RuntimeError(
            "FBX dışa aktarımı Blender gerektiriyor ama bulunamadı. "
            "Blender'ı kurup STUDIO_BLENDER_PATH ile yolunu belirtebilir ya da "
            "PATH'e ekleyebilirsiniz. Kurulum reçetesi: docs/worker-envs.md"
        )

    proc = subprocess.run(
        [blender, "--background", "--python", _SCRIPT, "--", "--input", input_path, "--output", output_path],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0 or "FBX_BRIDGE_OK" not in proc.stdout:
        raise RuntimeError(f"Blender FBX dışa aktarımı başarısız: {proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")
