"""İş kuyruğu rotaları (Faz 3): oluştur, listele, iptal/duraklat/devam."""

from __future__ import annotations

import asyncio
from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel, Field
from sqlalchemy import select
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

    def _create() -> str:
        with session_scope() as session:
            job = Job(
                kind=req.kind,
                params=req.params,
                priority=req.priority,
                project_id=req.project_id,
                status=JobStatus.QUEUED,
            )
            session.add(job)
            session.flush()
            return job.id

    job_id = await asyncio.to_thread(_create)
    scheduler.notify()
    out = await _get_out(job_id)
    return out


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
