"""Model registry rotaları (Faz 2): listele + yeniden tara."""

from __future__ import annotations

import asyncio
from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import select
from studio_core.db import session_scope
from studio_core.models import ModelRecord
from studio_models.scanner import scan

router = APIRouter(prefix="/v1/models", tags=["models"])


class ModelOut(BaseModel):
    id: str
    name: str
    family: str
    kind: str
    format: str
    path: str
    size_bytes: int
    sha256: str
    license: str
    meta: dict
    created_at: datetime


class ScanSummary(BaseModel):
    added: int
    removed: int
    updated: int
    total: int


@router.get("")
async def list_models(kind: str | None = None) -> list[ModelOut]:
    def _query() -> list[ModelOut]:
        with session_scope() as session:
            stmt = select(ModelRecord).order_by(ModelRecord.created_at.desc())
            if kind:
                stmt = stmt.where(ModelRecord.kind == kind)
            return [ModelOut.model_validate(r, from_attributes=True) for r in session.scalars(stmt)]

    return await asyncio.to_thread(_query)


@router.post("/rescan")
async def rescan() -> ScanSummary:
    result = await asyncio.to_thread(scan)
    return ScanSummary(
        added=len(result.added),
        removed=len(result.removed),
        updated=len(result.updated),
        total=result.total,
    )
