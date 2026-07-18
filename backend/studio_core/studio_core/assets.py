"""Proje sistemi v1 (Faz 4): varsayılan proje + iş çıktısından Asset kaydı.

Asset Graph ilkesi (önemli.md): üretilen her varlık bir üretim ağının düğümüdür.
Bu ilk sürümde her başarılı işin dosya çıktısı, üretim parametreleriyle birlikte
asset olarak kaydedilir → "tek tıkla yeniden üret" bunun üzerinden çalışır.
"""

from __future__ import annotations

from sqlalchemy import select

from studio_core.db import session_scope
from studio_core.logging import get_logger
from studio_core.models import Asset, Job, Project

log = get_logger("studio_core.assets")

DEFAULT_PROJECT_NAME = "Varsayılan"


def get_or_create_default_project() -> str:
    with session_scope() as session:
        project = session.scalars(
            select(Project).where(Project.name == DEFAULT_PROJECT_NAME)
        ).first()
        if project is None:
            project = Project(name=DEFAULT_PROJECT_NAME, description="Otomatik oluşturuldu")
            session.add(project)
            session.flush()
        return project.id


def register_asset_for_job(job_id: str) -> str | None:
    """Dosya çıktısı olan biten işi asset'e bağlar. Scheduler on_done kancasından çağrılır."""
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None:
            return None
        path = (job.result or {}).get("path", "")
        if not path:
            return None
        project_id = job.project_id
    if project_id is None:
        project_id = get_or_create_default_project()

    with session_scope() as session:
        job = session.get(Job, job_id)
        # Asset Graph: kaynak görselden türetilen çıktı (img2img/inpaint/outpaint)
        # parent_id ile kaynağına bağlanır — türev zinciri buradan kurulur.
        parent_id = None
        wanted_parent = (job.params or {}).get("parent_asset_id")
        if wanted_parent and session.get(Asset, wanted_parent) is not None:
            parent_id = wanted_parent
        asset = Asset(
            project_id=project_id,
            parent_id=parent_id,
            kind=job.kind,
            path=path,
            meta={
                "job_id": job_id,
                "job_kind": job.kind,
                "params": dict(job.params),
                "result": dict(job.result or {}),
            },
        )
        session.add(asset)
        session.flush()
        asset_id = asset.id
    log.info("asset_registered", asset_id=asset_id, job_id=job_id, kind=job.kind)
    return asset_id
