"""Model dosyası metadata çıkarımı (Faz 2): safetensors header, GGUF metadata, sınıflandırma.

Güvenlik (ADR-0007): pickle tabanlı formatlar (.ckpt/.pt/.pth/.bin) ASLA açılmaz; yalnızca
uzantıdan "pickle" olarak işaretlenir — yükleme karantina + kullanıcı onayı gerektirir.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any, BinaryIO

# sha256'yı tarama sırasında yalnızca bu boyuta kadar hesapla; büyük dosyalar
# indirme doğrulamasında veya istek üzerine hash'lenir.
HASH_SIZE_LIMIT = 256 * 1024 * 1024

MODEL_EXTS = {".safetensors", ".gguf", ".onnx", ".ckpt", ".pt", ".pth", ".bin"}

_FORMAT_BY_EXT = {
    ".safetensors": "safetensors",
    ".gguf": "gguf",
    ".onnx": "onnx",
    ".ckpt": "pickle",
    ".pt": "pickle",
    ".pth": "pickle",
    ".bin": "pickle",
}

# models/<subdir> → ModelRecord.kind (plan.md §3 dizin şeması)
_KIND_BY_SUBDIR = {
    "image": "checkpoint",
    "video": "video",
    "3d": "3d",
    "tts": "tts",
    "stt": "stt",
    "ocr": "ocr",
    "embeddings": "embedding",
    "llm": "llm",
    "lora": "lora",
    "controlnet": "controlnet",
    "checkpoints": "checkpoint",
    "vae": "vae",
    "background_removal": "background_removal",
    "upscale": "upscale",
    "captioning": "captioning",
    "segmentation": "segmentation",
    "face_restoration": "face_restoration",
}


def read_safetensors_header(path: Path, max_header: int = 64 * 1024 * 1024) -> dict[str, Any]:
    """safetensors: ilk 8 bayt LE uint64 = JSON header uzunluğu, ardından header."""
    with path.open("rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        if header_len > max_header:
            raise ValueError(f"safetensors header çok büyük: {header_len}")
        header = json.loads(f.read(header_len))
    meta = header.pop("__metadata__", {}) or {}
    return {"metadata": meta, "tensor_count": len(header)}


# GGUF değer tipleri → (struct fmt, bayt)
_GGUF_SCALARS = {
    0: ("<B", 1),  # uint8
    1: ("<b", 1),  # int8
    2: ("<H", 2),  # uint16
    3: ("<h", 2),  # int16
    4: ("<I", 4),  # uint32
    5: ("<i", 4),  # int32
    6: ("<f", 4),  # float32
    7: ("<B", 1),  # bool
    10: ("<Q", 8),  # uint64
    11: ("<q", 8),  # int64
    12: ("<d", 8),  # float64
}
_GGUF_STRING = 8
_GGUF_ARRAY = 9


def _gguf_read_string(f: BinaryIO) -> str:
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", errors="replace")


def _gguf_read_value(f: BinaryIO, vtype: int, keep: bool) -> Any:
    """Değeri okur; keep=False ise array içeriğini atlayıp None döner."""
    if vtype in _GGUF_SCALARS:
        fmt, size = _GGUF_SCALARS[vtype]
        (val,) = struct.unpack(fmt, f.read(size))
        return bool(val) if vtype == 7 else val
    if vtype == _GGUF_STRING:
        return _gguf_read_string(f)
    if vtype == _GGUF_ARRAY:
        (elem_type,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        if elem_type in _GGUF_SCALARS and not keep:
            f.seek(_GGUF_SCALARS[elem_type][1] * count, 1)
            return None
        items = [_gguf_read_value(f, elem_type, keep=False) for _ in range(count)]
        return items if keep else None
    raise ValueError(f"bilinmeyen GGUF tipi: {vtype}")


def read_gguf_metadata(path: Path, max_kv: int = 4096) -> dict[str, Any]:
    """GGUF v2/v3 metadata kv çiftlerini okur; tensör verisine hiç dokunmaz.

    Yalnızca skaler/string değerler saklanır; büyük array'ler (ör. tokenizer sözlüğü)
    atlanır — Model Browser için general.* alanları yeterlidir.
    """
    kv: dict[str, Any] = {}
    with path.open("rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("GGUF magic yok")
        (version,) = struct.unpack("<I", f.read(4))
        (tensor_count,) = struct.unpack("<Q", f.read(8))
        (kv_count,) = struct.unpack("<Q", f.read(8))
        if kv_count > max_kv:
            raise ValueError(f"GGUF kv sayısı şüpheli: {kv_count}")
        for _ in range(kv_count):
            key = _gguf_read_string(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            val = _gguf_read_value(f, vtype, keep=False)
            if val is not None:
                kv[key] = val
    return {"version": version, "tensor_count": tensor_count, "metadata": kv}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def classify(path: Path, models_root: Path) -> str:
    """models/<subdir>/... yerleşiminden tür çıkarır; kök/bilinmeyen dizinde uzantıya düşer."""
    try:
        subdir = path.relative_to(models_root).parts[0].lower()
    except (ValueError, IndexError):
        subdir = ""
    if subdir in _KIND_BY_SUBDIR and path.relative_to(models_root).parts[1:]:
        return _KIND_BY_SUBDIR[subdir]
    return "llm" if path.suffix.lower() == ".gguf" else "unknown"


def group_key(path: Path, models_root: Path) -> str:
    """path'in ait olduğu mantıksal model grubunun anahtarını bulur — birden çok
    ağırlık dosyası tek bir model olarak birleştirilecekse (çok bileşenli bir
    diffusers pipeline'ı: model_index.json + transformer/vae/text_encoder alt
    klasörleri, bazıları parçalı; ya da tek klasörde birden çok ağırlık dosyası:
    XTTS-v2'nin model.pth + speakers_xtts.pth'i) grup anahtarı o klasördür.
    Aksi halde (ör. Kokoro'nun voices/*.pt ses paketleri — her biri bağımsız
    kullanılabilir bir varlık) dosyanın kendisi kendi grubudur."""
    d = path.parent
    cur = d
    while True:
        try:
            cur.relative_to(models_root)
        except ValueError:
            break
        if (cur / "model_index.json").exists():
            return str(cur)
        if cur == models_root:
            break
        cur = cur.parent
    if d != models_root and (d / "config.json").exists():
        return str(d)
    return str(path)


def extract_group(key: str, paths: list[Path], models_root: Path) -> dict[str, Any]:
    """Bir mantıksal model için (tek dosya ya da group_key ile gruplanmış çok
    bileşenli/parçalı bir ağırlık kümesi) tek bir ModelRecord alan seti üretir.
    Bozuk dosyada meta.error doldurulur."""
    paths = sorted(paths)  # deterministik: her taramada aynı temsilci dosya seçilir
    representative = max(paths, key=lambda p: p.stat().st_size)
    fmt = _FORMAT_BY_EXT.get(representative.suffix.lower(), "")
    total_size = sum(p.stat().st_size for p in paths)

    meta: dict[str, Any] = {}
    family = ""
    if len(paths) == 1:
        try:
            if fmt == "safetensors":
                meta = read_safetensors_header(representative)
            elif fmt == "gguf":
                meta = read_gguf_metadata(representative)
                family = str(meta["metadata"].get("general.architecture", ""))
        except (ValueError, OSError, json.JSONDecodeError, struct.error) as exc:
            meta = {"error": f"metadata okunamadı: {exc}"}
    else:
        meta = {"components": [p.name for p in paths]}

    root_path = Path(key)
    name = root_path.name if root_path.is_dir() else representative.stem

    return {
        "name": name,
        "family": family,
        "kind": classify(representative, models_root),
        "format": fmt,
        "path": str(representative),
        "size_bytes": total_size,
        "sha256": (
            sha256_of(representative) if len(paths) == 1 and total_size <= HASH_SIZE_LIMIT else ""
        ),
        "meta": meta,
    }


def extract(path: Path, models_root: Path) -> dict[str, Any]:
    """Tek dosya için ModelRecord alan seti üretir — extract_group'un tek-dosyalık
    kısayolu (scanner.py grup taraması için extract_group'u doğrudan kullanır)."""
    return extract_group(group_key(path, models_root), [path], models_root)
