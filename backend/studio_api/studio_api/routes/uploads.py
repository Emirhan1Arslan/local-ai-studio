"""Yükleme ucu (Faz 4): img2img/inpaint/controlnet için kaynak görsel + maske.

Tarayıcı UI'ı yerel dosya yoluna erişemez; görseli base64 ile buraya yükler,
dönen yol iş parametrelerinde (init_image / mask_image / controlnet image)
kullanılır. Dosyalar ~/LocalAIStudio/uploads altına yazılır.
"""

from __future__ import annotations

import asyncio
import base64
import uuid
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel
from studio_core.config import get_settings

router = APIRouter(prefix="/v1/uploads", tags=["uploads"])

_ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".webp"}
_MAX_BYTES = 32 * 1024 * 1024


class UploadRequest(BaseModel):
    filename: str
    data_base64: str


class UploadOut(BaseModel):
    path: str
    size_bytes: int


@router.post("")
async def create_upload(req: UploadRequest) -> UploadOut:
    from studio_api.app import ApiError

    name = Path(req.filename).name  # path traversal koruması: yalnız dosya adı
    ext = Path(name).suffix.lower()
    if ext not in _ALLOWED_EXT:
        raise ApiError(400, "bad_request", f"desteklenmeyen uzantı: {ext!r} (png/jpg/jpeg/webp)")
    try:
        data = base64.b64decode(req.data_base64, validate=True)
    except Exception as exc:
        raise ApiError(400, "bad_request", "geçersiz base64 verisi") from exc
    if len(data) > _MAX_BYTES:
        raise ApiError(413, "too_large", "dosya 32 MB sınırını aşıyor")

    settings = get_settings()
    settings.ensure_dirs()
    dest = settings.uploads_dir / f"{uuid.uuid4().hex[:8]}_{name}"
    await asyncio.to_thread(dest.write_bytes, data)
    return UploadOut(path=str(dest), size_bytes=len(data))
