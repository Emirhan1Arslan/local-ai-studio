"""İş kuyruğu rotaları (Faz 3): oluştur, listele, iptal/duraklat/devam."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel, Field
from sqlalchemy import select
from studio_core.config import get_settings
from studio_core.db import session_scope
from studio_core.models import Job, JobStatus
from studio_inference.adapters import available_kinds
from studio_inference.scheduler import scheduler

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


class JobRequest(BaseModel):
    kind: str
    params: dict = Field(default_factory=dict)
    priority: int = 0
    project_id: str | None = None


class BatchJobRequest(JobRequest):
    count: int = Field(default=1, ge=1, le=16)


class JobOut(BaseModel):
    id: str
    kind: str
    status: str
    priority: int
    owner: str
    project_id: str | None
    params: dict
    result: dict
    error: str
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class ActionResult(BaseModel):
    ok: bool


@router.post("")
async def create_job(req: JobRequest) -> JobOut:
    if req.kind not in available_kinds():
        from studio_api.app import ApiError

        raise ApiError(400, "bad_request", f"bilinmeyen iş türü: {req.kind!r}")

    job_id = await asyncio.to_thread(_create_job_row, req.kind, dict(req.params), req)
    scheduler.notify()
    return await _get_out(job_id)


@router.post("/batch")
async def create_jobs_batch(req: BatchJobRequest) -> list[JobOut]:
    """Batch üretim + seed varyasyonları (Faz 4): sabit seed s → s, s+1, … s+n-1."""
    if req.kind not in available_kinds():
        from studio_api.app import ApiError

        raise ApiError(400, "bad_request", f"bilinmeyen iş türü: {req.kind!r}")

    base_seed = req.params.get("seed", -1)
    job_ids: list[str] = []
    for i in range(req.count):
        params = dict(req.params)
        if isinstance(base_seed, int) and base_seed >= 0:
            params["seed"] = base_seed + i
        params.pop("output_path", None)  # her iş kendi benzersiz çıktısını alır
        job_id = await asyncio.to_thread(_create_job_row, req.kind, params, req)
        job_ids.append(job_id)
    scheduler.notify()
    return [await _get_out(j) for j in job_ids]


# imgtool op'ları görsel (PNG) ya da yapılandırılmış metin (JSON) üretir
_IMGTOOL_IMAGE_OPS = {"bg_remove", "upscale", "segment", "crop", "adjust", "face_restore"}


def _create_job_row(kind: str, params: dict, req: JobRequest) -> str:
    # Görüntü işleri çıktıyı outputs/ altına yazar (proje sistemi bunu asset'e bağlar)
    if kind == "image" and not params.get("output_path"):
        settings = get_settings()
        settings.ensure_dirs()
        params["output_path"] = str(settings.outputs_dir / f"{uuid.uuid4().hex}.png")
    elif kind == "imgtool" and not params.get("output_path"):
        settings = get_settings()
        settings.ensure_dirs()
        ext = "png" if params.get("op") in _IMGTOOL_IMAGE_OPS else "json"
        params["output_path"] = str(settings.outputs_dir / f"{uuid.uuid4().hex}.{ext}")
    elif kind == "audio" and not params.get("output_path"):
        settings = get_settings()
        settings.ensure_dirs()
        ext = "wav" if params.get("op") == "tts" else "json"
        params["output_path"] = str(settings.outputs_dir / f"{uuid.uuid4().hex}.{ext}")
    elif kind == "video" and not params.get("output_path"):
        settings = get_settings()
        settings.ensure_dirs()
        params["output_path"] = str(settings.outputs_dir / f"{uuid.uuid4().hex}.mp4")
    elif kind == "videotool" and not params.get("output_path"):
        settings = get_settings()
        settings.ensure_dirs()
        ext = str(params.get("format", "mp4") or "mp4")
        params["output_path"] = str(settings.outputs_dir / f"{uuid.uuid4().hex}.{ext}")
    with session_scope() as session:
        job = Job(
            kind=kind,
            params=params,
            priority=req.priority,
            project_id=req.project_id,
            status=JobStatus.QUEUED,
        )
        session.add(job)
        session.flush()
        return job.id


@router.get("")
async def list_jobs(status: str | None = None, limit: int = 100) -> list[JobOut]:
    def _query() -> list[JobOut]:
        with session_scope() as session:
            stmt = select(Job).order_by(Job.created_at.desc()).limit(limit)
            if status:
                stmt = stmt.where(Job.status == status)
            return [JobOut.model_validate(j, from_attributes=True) for j in session.scalars(stmt)]

    return await asyncio.to_thread(_query)


async def _get_out(job_id: str) -> JobOut:
    def _query() -> JobOut | None:
        with session_scope() as session:
            job = session.get(Job, job_id)
            return JobOut.model_validate(job, from_attributes=True) if job is not None else None

    out = await asyncio.to_thread(_query)
    if out is None:
        from studio_api.app import ApiError

        raise ApiError(404, "not_found", f"iş bulunamadı: {job_id}")
    return out


@router.get("/{job_id}")
async def get_job(job_id: str) -> JobOut:
    return await _get_out(job_id)


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str) -> ActionResult:
    return ActionResult(ok=await scheduler.cancel(job_id))


@router.post("/{job_id}/pause")
async def pause_job(job_id: str) -> ActionResult:
    return ActionResult(ok=await scheduler.pause(job_id))


@router.post("/{job_id}/resume")
async def resume_job(job_id: str) -> ActionResult:
    return ActionResult(ok=await scheduler.resume(job_id))
