"""İndirme rotaları (Faz 2): kuyruğa al, listele, duraklat/devam/iptal.

Güvenlik: hedef yol her zaman models/<subdir>/<dosya> olarak kurulur ve models dizini
dışına çıkamaz (path traversal reddedilir) — bu API dış istemcilere de açık (ADR-0003).
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from fastapi import APIRouter
from pydantic import BaseModel, Field
from sqlalchemy import select
from studio_core.config import MODEL_SUBDIRS, get_settings
from studio_core.db import session_scope
from studio_core.models import Download
from studio_models.downloader import hf_resolve_url, manager

router = APIRouter(prefix="/v1/downloads", tags=["downloads"])


class DownloadRequest(BaseModel):
    # Ya doğrudan url ya da HF repo_id+filename verilir.
    url: str = ""
    repo_id: str = ""
    filename: str = ""
    revision: str = "main"
    subdir: str = Field(description=f"models/ altındaki hedef: {', '.join(MODEL_SUBDIRS)}")
    sha256: str = ""


class DownloadOut(BaseModel):
    id: str
    url: str
    source: str
    dest_path: str
    status: str
    total_bytes: int
    downloaded_bytes: int
    sha256_expected: str
    sha256_actual: str
    error: str
    created_at: datetime
    finished_at: datetime | None


class ActionResult(BaseModel):
    ok: bool


def _bad_request(message: str):
    from studio_api.app import ApiError

    raise ApiError(400, "bad_request", message)


def _safe_dest(subdir: str, filename: str) -> Path:
    settings = get_settings()
    if subdir not in MODEL_SUBDIRS:
        _bad_request(f"geçersiz subdir: {subdir!r}")
    name = Path(filename).name  # yol bileşenlerini at
    if not name or name in (".", ".."):
        _bad_request(f"geçersiz dosya adı: {filename!r}")
    dest = (settings.models_dir / subdir / name).resolve()
    if settings.models_dir.resolve() not in dest.parents:
        _bad_request("hedef, models dizini dışında")
    return dest


@router.post("")
async def create_download(req: DownloadRequest) -> DownloadOut:
    settings = get_settings()
    headers: dict[str, str] = {}

    if req.repo_id:
        if not req.filename:
            _bad_request("HF indirmesi için filename gerekli")
        url = hf_resolve_url(req.repo_id, req.filename, req.revision)
        source = "hf"
        if settings.hf_token:
            headers["Authorization"] = f"Bearer {settings.hf_token}"
        filename = req.filename
    elif req.url:
        parsed = urlparse(req.url)
        if parsed.scheme not in ("http", "https"):
            _bad_request("yalnızca http/https URL desteklenir")
        url = req.url
        source = "url"
        filename = req.filename or Path(parsed.path).name
    else:
        _bad_request("url veya repo_id verilmeli")

    dest = _safe_dest(req.subdir, filename)
    download_id = await manager.enqueue(
        url, dest, source=source, sha256=req.sha256, headers=headers
    )
    return await _get_out(download_id)


@router.get("")
async def list_downloads() -> list[DownloadOut]:
    def _query() -> list[DownloadOut]:
        with session_scope() as session:
            stmt = select(Download).order_by(Download.created_at.desc())
            rows = session.scalars(stmt)
            return [DownloadOut.model_validate(r, from_attributes=True) for r in rows]

    return await asyncio.to_thread(_query)


async def _get_out(download_id: str) -> DownloadOut:
    def _query() -> DownloadOut | None:
        with session_scope() as session:
            rec = session.get(Download, download_id)
            return (
                DownloadOut.model_validate(rec, from_attributes=True) if rec is not None else None
            )

    out = await asyncio.to_thread(_query)
    if out is None:
        from studio_api.app import ApiError

        raise ApiError(404, "not_found", f"indirme bulunamadı: {download_id}")
    return out


@router.post("/{download_id}/pause")
async def pause_download(download_id: str) -> ActionResult:
    return ActionResult(ok=await manager.pause(download_id))


@router.post("/{download_id}/resume")
async def resume_download(download_id: str) -> ActionResult:
    settings = get_settings()
    headers: dict[str, str] = {}
    rec_out = await _get_out(download_id)
    if rec_out.source == "hf" and settings.hf_token:
        headers["Authorization"] = f"Bearer {settings.hf_token}"
    return ActionResult(ok=await manager.resume(download_id, headers))


@router.post("/{download_id}/cancel")
async def cancel_download(download_id: str) -> ActionResult:
    return ActionResult(ok=await manager.cancel(download_id))
