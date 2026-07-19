"""Ayarlar rotası (Faz 1 cila): canlı-etkili ayar override'ları + tanı paketi.

Yol alanları salt okunurdur: değiştirmek veri taşımayı gerektirir, bu Faz 15
ilk-çalıştırma sihirbazının kapsamındadır. Burada yalnız restart gerektirmeyen,
zaten her kullanımda taze okunan alanlar (telemetri aralığı, worker idle timeout,
worker eşzamanlılığı, HF token, indirme hız sınırı) düzenlenebilir.
"""

from __future__ import annotations

import asyncio
import json
import zipfile
from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel, Field
from sqlalchemy import select
from studio_core.config import get_settings
from studio_core.db import session_scope
from studio_core.hardware import get_system_info
from studio_core.models import Job, JobStatus
from studio_core.settings_store import save_override
from studio_inference.scheduler import scheduler

router = APIRouter(prefix="/v1/settings", tags=["settings"])


class PathsOut(BaseModel):
    data_dir: str
    models_dir: str
    outputs_dir: str
    cache_dir: str
    logs_dir: str
    envs_dir: str
    uploads_dir: str
    db_path: str


class SettingsOut(BaseModel):
    hf_token_set: bool
    telemetry_interval: float
    worker_idle_timeout: float
    worker_concurrency: int
    download_speed_limit_mbps: float
    paths: PathsOut


class SettingsUpdate(BaseModel):
    hf_token: str | None = None  # "" => temizle
    telemetry_interval: float | None = Field(default=None, ge=0.5, le=30)
    worker_idle_timeout: float | None = Field(default=None, ge=10, le=3600)
    worker_concurrency: int | None = Field(default=None, ge=1, le=4)
    # 0 = sınırsız
    download_speed_limit_mbps: float | None = Field(default=None, ge=0, le=1000)


class DiagnosticsOut(BaseModel):
    path: str


def _snapshot() -> SettingsOut:
    s = get_settings()
    return SettingsOut(
        hf_token_set=bool(s.hf_token),
        telemetry_interval=s.telemetry_interval,
        worker_idle_timeout=s.worker_idle_timeout,
        worker_concurrency=s.worker_concurrency,
        download_speed_limit_mbps=s.download_speed_limit_mbps,
        paths=PathsOut(
            data_dir=str(s.data_dir),
            models_dir=str(s.models_dir),
            outputs_dir=str(s.outputs_dir),
            cache_dir=str(s.cache_dir),
            logs_dir=str(s.logs_dir),
            envs_dir=str(s.envs_dir),
            uploads_dir=str(s.uploads_dir),
            db_path=str(s.db_path),
        ),
    )


@router.get("")
async def get_settings_route() -> SettingsOut:
    return await asyncio.to_thread(_snapshot)


@router.put("")
async def update_settings(body: SettingsUpdate) -> SettingsOut:
    def _apply() -> SettingsOut:
        s = get_settings()
        data = body.model_dump(exclude_unset=True)
        if "hf_token" in data:
            s.hf_token = data["hf_token"] or ""
            save_override("hf_token", s.hf_token)
        if "telemetry_interval" in data:
            s.telemetry_interval = data["telemetry_interval"]
            save_override("telemetry_interval", s.telemetry_interval)
        if "worker_idle_timeout" in data:
            s.worker_idle_timeout = data["worker_idle_timeout"]
            save_override("worker_idle_timeout", s.worker_idle_timeout)
        if "worker_concurrency" in data:
            s.worker_concurrency = data["worker_concurrency"]
            scheduler._concurrency = data["worker_concurrency"]
            save_override("worker_concurrency", s.worker_concurrency)
        if "download_speed_limit_mbps" in data:
            s.download_speed_limit_mbps = data["download_speed_limit_mbps"]
            save_override("download_speed_limit_mbps", s.download_speed_limit_mbps)
        return _snapshot()

    return await asyncio.to_thread(_apply)


@router.post("/diagnostics")
async def create_diagnostics_bundle() -> DiagnosticsOut:
    """Yerel tanı paketi (plan.md Faz 1): loglar + sistem bilgisi + son başarısız
    işler tek zip'e paketlenir. Hiçbir yere gönderilmez; kullanıcı elle paylaşır."""

    def _build() -> str:
        s = get_settings()
        diag_dir = s.data_dir / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = diag_dir / f"tani-paketi-{ts}.zip"

        with session_scope() as session:
            stmt = (
                select(Job)
                .where(Job.status == JobStatus.FAILED)
                .order_by(Job.created_at.desc())
                .limit(20)
            )
            failed = [
                {
                    "id": j.id,
                    "kind": j.kind,
                    "error": j.error,
                    "created_at": j.created_at.isoformat(),
                }
                for j in session.scalars(stmt)
            ]

        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("system_info.json", get_system_info().model_dump_json(indent=2))
            zf.writestr("settings.json", _snapshot().model_dump_json(indent=2))
            zf.writestr(
                "son_basarisiz_isler.json", json.dumps(failed, ensure_ascii=False, indent=2)
            )
            for f in sorted(s.logs_dir.glob("studio.log*")):
                zf.write(f, arcname=f"logs/{f.name}")
        return str(out_path)

    path = await asyncio.to_thread(_build)
    return DiagnosticsOut(path=path)
