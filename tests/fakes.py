"""Test yardımcıları: geçerli başlıklı sahte safetensors/GGUF dosyaları üretir."""

from __future__ import annotations

import json
import struct
from pathlib import Path


def write_fake_safetensors(
    path: Path, metadata: dict | None = None, tensor_names: list[str] | None = None
) -> None:
    header: dict = {"__metadata__": metadata or {"ss_network_module": "networks.lora"}}
    for name in tensor_names or ["weight"]:
        header[name] = {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]}
    blob = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(b"\x00\x00\x00\x00")  # 4 baytlık sahte tensör verisi


def _gguf_string(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def write_fake_gguf(path: Path, name: str = "Test Model", arch: str = "llama") -> None:
    kv = [
        ("general.architecture", arch),
        ("general.name", name),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))  # version
        f.write(struct.pack("<Q", 0))  # tensor_count
        f.write(struct.pack("<Q", len(kv)))  # kv_count
        for key, val in kv:
            f.write(_gguf_string(key))
            f.write(struct.pack("<I", 8))  # tip: string
            f.write(_gguf_string(val))
