"""Proje rotaları (Faz 4, proje sistemi v1)."""

from __future__ import annotations

import asyncio
from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import select
from studio_core.db import session_scope
from studio_core.models import Project

router = APIRouter(prefix="/v1/projects", tags=["projects"])


class ProjectRequest(BaseModel):
    name: str
    description: str = ""


class ProjectOut(BaseModel):
    id: str
    name: str
    description: str
    created_at: datetime


@router.post("")
async def create_project(req: ProjectRequest) -> ProjectOut:
    def _create() -> ProjectOut:
        with session_scope() as session:
            project = Project(name=req.name, description=req.description)
            session.add(project)
            session.flush()
            return ProjectOut.model_validate(project, from_attributes=True)

    return await asyncio.to_thread(_create)


@router.get("")
async def list_projects() -> list[ProjectOut]:
    def _query() -> list[ProjectOut]:
        with session_scope() as session:
            stmt = select(Project).order_by(Project.created_at.desc())
            return [
                ProjectOut.model_validate(p, from_attributes=True) for p in session.scalars(stmt)
            ]

    return await asyncio.to_thread(_query)
