"""Asset rotaları (Faz 4): listele + tek tıkla yeniden üret."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import select
from studio_core.config import get_settings
from studio_core.db import session_scope
from studio_core.models import Asset, Job, JobStatus
from studio_inference.scheduler import scheduler

router = APIRouter(prefix="/v1/assets", tags=["assets"])


class AssetOut(BaseModel):
    id: str
    project_id: str
    parent_id: str | None
    kind: str
    role: str
    path: str
    meta: dict
    created_at: datetime


class RegenerateOut(BaseModel):
    job_id: str


@router.get("")
async def list_assets(
    project_id: str | None = None, kind: str | None = None, limit: int = 50
) -> list[AssetOut]:
    def _query() -> list[AssetOut]:
        with session_scope() as session:
            stmt = select(Asset).order_by(Asset.created_at.desc()).limit(limit)
            if project_id:
                stmt = stmt.where(Asset.project_id == project_id)
            if kind:
                stmt = stmt.where(Asset.kind == kind)
            return [AssetOut.model_validate(a, from_attributes=True) for a in session.scalars(stmt)]

    return await asyncio.to_thread(_query)


@router.post("/{asset_id}/regenerate")
async def regenerate(asset_id: str) -> RegenerateOut:
    """Asset'in üretim parametreleriyle yeni bir iş kuyruklar (tek tıkla yeniden üret)."""

    def _create() -> str | None:
        with session_scope() as session:
            asset = session.get(Asset, asset_id)
            if asset is None:
                return None
            params = dict(asset.meta.get("params", {}))
            kind = asset.meta.get("job_kind", asset.kind)
            if kind == "image":
                settings = get_settings()
                settings.ensure_dirs()
                params["output_path"] = str(settings.outputs_dir / f"{uuid.uuid4().hex}.png")
            elif kind == "imgtool":
                settings = get_settings()
                settings.ensure_dirs()
                image_ops = ("bg_remove", "upscale", "segment", "crop", "adjust")
                ext = "png" if params.get("op") in image_ops else "json"
                params["output_path"] = str(settings.outputs_dir / f"{uuid.uuid4().hex}.{ext}")
            job = Job(
                kind=kind,
                params=params,
                priority=0,
                project_id=asset.project_id,
                status=JobStatus.QUEUED,
            )
            session.add(job)
            session.flush()
            return job.id

    job_id = await asyncio.to_thread(_create)
    if job_id is None:
        from studio_api.app import ApiError

        raise ApiError(404, "not_found", f"asset bulunamadı: {asset_id}")
    scheduler.notify()
    return RegenerateOut(job_id=job_id)
