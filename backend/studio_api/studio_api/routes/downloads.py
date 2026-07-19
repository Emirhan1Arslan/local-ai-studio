"""İndirme rotaları (Faz 2): kuyruğa al, listele, duraklat/devam/iptal.

Güvenlik: hedef yol her zaman models/<subdir>/<dosya> olarak kurulur ve models dizini
dışına çıkamaz (path traversal reddedilir) — bu API dış istemcilere de açık (ADR-0003).
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter
from pydantic import BaseModel, Field
from sqlalchemy import select
from studio_core.config import MODEL_SUBDIRS, get_settings
from studio_core.db import session_scope
from studio_core.models import Download
from studio_models.downloader import DownloadError, hf_list_repo_files, hf_resolve_url, manager

router = APIRouter(prefix="/v1/downloads", tags=["downloads"])


class DownloadRequest(BaseModel):
    # Ya doğrudan url ya da HF repo_id+filename verilir.
    url: str = ""
    repo_id: str = ""
    filename: str = ""
    revision: str = "main"
    subdir: str = Field(description=f"models/ altındaki hedef: {', '.join(MODEL_SUBDIRS)}")
    sha256: str = ""


class SnapshotRequest(BaseModel):
    repo_id: str
    subdir: str = Field(description=f"models/ altındaki hedef: {', '.join(MODEL_SUBDIRS)}")
    revision: str = "main"


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


def _safe_dest(subdir: str, filename: str, extra_subdir: str = "") -> Path:
    settings = get_settings()
    if subdir not in MODEL_SUBDIRS:
        _bad_request(f"geçersiz subdir: {subdir!r}")
    name = Path(filename).name  # yol bileşenlerini at
    if not name or name in (".", ".."):
        _bad_request(f"geçersiz dosya adı: {filename!r}")
    base = settings.models_dir / subdir
    if extra_subdir:
        safe_extra = Path(extra_subdir).name  # snapshot indirmede repo adı — yol bileşeni değil
        if not safe_extra or safe_extra in (".", ".."):
            _bad_request(f"geçersiz repo adı: {extra_subdir!r}")
        base = base / safe_extra
    dest = (base / name).resolve()
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


@router.post("/snapshot")
async def create_snapshot_download(req: SnapshotRequest) -> list[DownloadOut]:
    """Çok dosyalı bir HF repo'sunu tek seferde kuyruğa alır (Faz 5: BiRefNet, Florence-2,
    SAM, Grounding DINO gibi tek-dosya olmayan modeller için). Her dosya, mevcut resumable
    tek-dosya indirme motoruyla ayrı ayrı indirilir; hepsi `models/<subdir>/<repo-adı>/`
    altında bir arada toplanır — from_pretrained(dizin) bunu doğrudan yükleyebilir.
    """
    settings = get_settings()
    if req.subdir not in MODEL_SUBDIRS:
        _bad_request(f"geçersiz subdir: {req.subdir!r}")

    try:
        all_files = await hf_list_repo_files(req.repo_id, transport=manager._transport)
    except (DownloadError, httpx.HTTPError) as exc:
        _bad_request(str(exc))

    filenames = [f for f in all_files if not f.endswith((".gitattributes", ".md"))]
    if not filenames:
        _bad_request("repo'da indirilecek dosya bulunamadı")
    if any("/" in f for f in filenames):
        _bad_request(
            "bu repo alt klasörlü dosyalar içeriyor (ör. varyant/onnx) — "
            "şimdilik yalnız düz yapılı repo'lar destekleniyor"
        )

    repo_slug = req.repo_id.rsplit("/", 1)[-1]
    headers: dict[str, str] = {}
    if settings.hf_token:
        headers["Authorization"] = f"Bearer {settings.hf_token}"

    ids: list[str] = []
    for filename in filenames:
        dest = _safe_dest(req.subdir, filename, extra_subdir=repo_slug)
        url = hf_resolve_url(req.repo_id, filename, req.revision)
        ids.append(await manager.enqueue(url, dest, source="hf", headers=headers))
    return [await _get_out(i) for i in ids]


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
