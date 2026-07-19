"""FFmpeg PATH yardımcısı — audio.py (XTTS-v2/torchcodec), video.py (kare→MP4
encode) ve videotool.py (kes/birleştir/hız/format) tarafından paylaşılır.

DİKKAT: Bu modül worker env'inde çalışır. 3.11+ sözdizimi KULLANMA.
"""

from __future__ import annotations

import os
from pathlib import Path


def ensure_ffmpeg_on_path() -> None:
    """`~/LocalAIStudio/tools/ffmpeg/bin`'i (elle indirilen build, bkz.
    docs/worker-envs.md) gerçek PATH'e ekler — llm.py'deki torch-DLL çözümüyle
    aynı desen (`os.add_dll_directory` değil, gerçek `PATH`; winmode farkı)."""
    data_dir = Path(os.environ.get("STUDIO_DATA_DIR") or (Path.home() / "LocalAIStudio"))
    ffmpeg_bin = data_dir / "tools" / "ffmpeg" / "bin"
    if not ffmpeg_bin.is_dir():
        return
    path = os.environ.get("PATH", "")
    if str(ffmpeg_bin) not in path:
        os.environ["PATH"] = str(ffmpeg_bin) + os.pathsep + path


def ffmpeg_exe() -> str:
    """Çalıştırılabilir ffmpeg yolu: elle indirilen build varsa onu, yoksa PATH'teki
    'ffmpeg' adını döner (sistemde ayrıca kuruluysa çalışır)."""
    ensure_ffmpeg_on_path()
    data_dir = Path(os.environ.get("STUDIO_DATA_DIR") or (Path.home() / "LocalAIStudio"))
    candidate = data_dir / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
    return str(candidate) if candidate.is_file() else "ffmpeg"


def ffprobe_exe() -> str:
    ensure_ffmpeg_on_path()
    data_dir = Path(os.environ.get("STUDIO_DATA_DIR") or (Path.home() / "LocalAIStudio"))
    candidate = data_dir / "tools" / "ffmpeg" / "bin" / "ffprobe.exe"
    return str(candidate) if candidate.is_file() else "ffprobe"


def has_audio_stream(path: str) -> bool:
    """LTX-Video gibi motorların çıktısı ses izi taşımaz — hız/birleştirme
    filtrelerinin var olmayan bir 'a' akışına referans verip patlamaması için
    önce yoklanır."""
    import subprocess

    proc = subprocess.run(
        [
            ffprobe_exe(),
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            path,
        ],
        capture_output=True,
        text=True,
    )
    return bool(proc.stdout.strip())
