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
from studio_models.downloader import (
    DownloadError,
    estimate_safetensors_bytes,
    hf_list_repo_files,
    hf_resolve_url,
    hf_search_models,
    manager,
)

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


class HfModelOut(BaseModel):
    id: str
    pipeline_tag: str = ""
    library_name: str = ""
    downloads: int = 0
    likes: int = 0
    license: str = ""
    gated: bool = False
    size_bytes: int = 0  # yaklaşık — yalnız safetensors metadata'sı olan modellerde dolu


class HfFileOut(BaseModel):
    filename: str
    size: int


def _bad_request(message: str):
    from studio_api.app import ApiError

    raise ApiError(400, "bad_request", message)


def _safe_dest(
    subdir: str, filename: str, extra_subdir: str = "", preserve_subpath: bool = False
) -> Path:
    settings = get_settings()
    if subdir not in MODEL_SUBDIRS:
        _bad_request(f"geçersiz subdir: {subdir!r}")
    if preserve_subpath:
        # snapshot indirmede filename, hf_list_repo_files'tan gelen güvenilir bir repo-içi
        # göreli yoldur (ör. "voices/af_heart.pt") — alt klasör yapısı korunur.
        rel = Path(filename.replace("\\", "/"))
        if rel.is_absolute() or ".." in rel.parts or not rel.parts:
            _bad_request(f"geçersiz dosya adı: {filename!r}")
        name_part = rel
    else:
        name = Path(filename).name  # yol bileşenlerini at
        if not name or name in (".", ".."):
            _bad_request(f"geçersiz dosya adı: {filename!r}")
        name_part = Path(name)
    base = settings.models_dir / subdir
    if extra_subdir:
        safe_extra = Path(extra_subdir).name  # snapshot indirmede repo adı — yol bileşeni değil
        if not safe_extra or safe_extra in (".", ".."):
            _bad_request(f"geçersiz repo adı: {extra_subdir!r}")
        base = base / safe_extra
    dest = (base / name_part).resolve()
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


@router.get("/search")
async def search_hf_models(
    q: str = "", pipeline_tag: str = "", limit: int = 30
) -> list[HfModelOut]:
    """Keşfet paneli: Hugging Face Hub'da model arar (plan.md Faz 2'nin
    'Model Browser UI: filtre/arama' hedefini kaynak seçim aşamasına taşır)."""
    if limit < 1 or limit > 50:
        _bad_request("limit 1-50 arasında olmalı")
    try:
        raw = await hf_search_models(q, pipeline_tag, limit, transport=manager._transport)
    except (DownloadError, httpx.HTTPError) as exc:
        _bad_request(str(exc))

    out: list[HfModelOut] = []
    for m in raw:
        license_tag = next(
            (t.removeprefix("license:") for t in m.get("tags", []) if t.startswith("license:")), ""
        )
        safetensors = m.get("safetensors")
        out.append(
            HfModelOut(
                id=m["id"],
                pipeline_tag=m.get("pipeline_tag") or "",
                library_name=m.get("library_name") or "",
                downloads=m.get("downloads", 0),
                likes=m.get("likes", 0),
                license=license_tag,
                gated=bool(m.get("gated", False)),
                size_bytes=estimate_safetensors_bytes(safetensors) if safetensors else 0,
            )
        )
    return out


@router.get("/files")
async def list_hf_repo_files(repo_id: str) -> list[HfFileOut]:
    """Bir HF repo'sunun dosya listesi (ad + boyut) — Keşfet panelinde tek dosya mı
    yoksa tüm repo'yu mu (snapshot) indireceğine karar vermek için kullanılır."""
    try:
        files = await hf_list_repo_files(repo_id, transport=manager._transport)
    except (DownloadError, httpx.HTTPError) as exc:
        _bad_request(str(exc))
    return [HfFileOut(**f) for f in files]


@router.post("/snapshot")
async def create_snapshot_download(req: SnapshotRequest) -> list[DownloadOut]:
    """Çok dosyalı bir HF repo'sunu tek seferde kuyruğa alır (Faz 5: BiRefNet, Florence-2,
    SAM, Grounding DINO gibi tek-dosya olmayan modeller için; Faz 7'de Kokoro'nun
    `voices/` alt klasörü için de kullanılıyor). Her dosya, mevcut resumable tek-dosya
    indirme motoruyla ayrı ayrı indirilir; hepsi `models/<subdir>/<repo-adı>/` altında
    (repo'daki alt klasör yapısı korunarak) bir arada toplanır —
    from_pretrained(dizin) ya da doğrudan yerel dosya yolu bunu okuyabilir.
    """
    settings = get_settings()
    if req.subdir not in MODEL_SUBDIRS:
        _bad_request(f"geçersiz subdir: {req.subdir!r}")

    try:
        all_files = await hf_list_repo_files(req.repo_id, transport=manager._transport)
    except (DownloadError, httpx.HTTPError) as exc:
        _bad_request(str(exc))

    filenames = [
        f["filename"] for f in all_files if not f["filename"].endswith((".gitattributes", ".md"))
    ]
    if not filenames:
        _bad_request("repo'da indirilecek dosya bulunamadı")

    repo_slug = req.repo_id.rsplit("/", 1)[-1]
    headers: dict[str, str] = {}
    if settings.hf_token:
        headers["Authorization"] = f"Bearer {settings.hf_token}"

    ids: list[str] = []
    for filename in filenames:
        dest = _safe_dest(req.subdir, filename, extra_subdir=repo_slug, preserve_subpath=True)
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
