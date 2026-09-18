"""Gerçek Job sistemine blocking dispatch (Faz 10): `WorkflowExecutor`'ın
`job_dispatcher` kancasına takılır. Mevcut Scheduler/Job altyapısını yeniden
kullanır — job oluşturma, kuyruklama, worker yönetimi, VRAM/iptal mantığı hep
oradan gelir; burada yalnızca job'ı oluşturup bitmesini bekliyoruz.

`WorkflowExecutor.run()` senkron/blocking olduğundan bu dispatcher da
blocking'dir (kısa aralıklarla poll eder) — workflow yürütmesi bu yüzden bir
arka plan thread/task içinde çalıştırılmalı (API katmanı `asyncio.to_thread`
ile sağlar, bkz. `studio_api/routes/jobs.py`'deki aynı desen).
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import Any

from studio_core.config import get_settings
from studio_core.db import session_scope
from studio_core.models import Job, JobStatus
from studio_inference.scheduler import scheduler

from .nodes import JobDispatch

# imgtool op'ları görsel (PNG) ya da yapılandırılmış metin (JSON) üretir —
# studio_api/routes/jobs.py::_create_job_row ile aynı sözleşme. Yeni job `kind`
# dispatch eden node türleri eklendikçe bu tablo genişletilecek.
_IMGTOOL_IMAGE_OPS = {"bg_remove", "upscale", "segment", "crop", "adjust", "face_restore"}
_EXT_BY_KIND = {"image": "png", "imagex": "png"}


class JobDispatchError(Exception):
    pass


def _output_ext(kind: str, params: dict[str, Any]) -> str:
    if kind == "imgtool":
        return "png" if params.get("op") in _IMGTOOL_IMAGE_OPS else "json"
    if kind == "audio":
        return "wav" if params.get("op") == "tts" else "json"
    if kind in ("video", "videotool"):
        return str(params.get("format", "mp4") or "mp4")
    if kind in ("mesh", "meshtool"):
        return str(params.get("format", "glb") or "glb")
    return _EXT_BY_KIND.get(kind, "bin")


def make_job_dispatcher(
    *,
    project_id: str | None = None,
    priority: int = 0,
    poll_interval: float = 0.1,
    timeout: float = 3600.0,
) -> Callable[[JobDispatch, dict[str, Any]], dict[str, Any]]:
    def dispatch(strategy: JobDispatch, job_params: dict[str, Any]) -> dict[str, Any]:
        job_params = dict(job_params)
        if not job_params.get("output_path"):
            settings = get_settings()
            settings.ensure_dirs()
            ext = _output_ext(strategy.kind, job_params)
            job_params["output_path"] = str(settings.outputs_dir / f"{uuid.uuid4().hex}.{ext}")

        with session_scope() as session:
            job = Job(
                kind=strategy.kind,
                params=job_params,
                priority=priority,
                project_id=project_id,
                status=JobStatus.QUEUED,
            )
            session.add(job)
            session.flush()
            job_id = job.id
        scheduler.notify()

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with session_scope() as session:
                job = session.get(Job, job_id)
                if job is None:
                    raise JobDispatchError(f"iş kayboldu: {job_id}")
                if job.status == JobStatus.DONE:
                    return dict(job.result)
                if job.status in (JobStatus.FAILED, JobStatus.CANCELED):
                    raise JobDispatchError(job.error or f"iş {job.status.value} oldu")
            time.sleep(poll_interval)
        raise JobDispatchError(f"iş zaman aşımına uğradı ({strategy.kind}): {job_id}")

    return dispatch
