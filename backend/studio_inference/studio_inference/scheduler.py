"""Scheduler (Faz 3): öncelikli kuyruk + worker süreç yönetimi. Projenin kalbi.

- Sıra: priority DESC, created_at ASC; iş DB'de RUNNING olarak sahiplenilir (claim).
- Her iş ayrı worker sürecinde koşar; iptal = süreç kill → VRAM garanti geri döner.
- Worker çökerse (kill -9 dahil) iş FAILED olur, uygulama ayakta kalır.
- Açılışta önceki süreçten RUNNING kalmış işler FAILED'a çekilir (recover).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from studio_core.config import get_settings
from studio_core.db import session_scope
from studio_core.logging import get_logger
from studio_core.models import Job, JobStatus

log = get_logger("studio_inference.scheduler")

EmitFn = Callable[..., None]

# Worker'ın studio_inference'ı kaynaktan import edebilmesi için (kurulum gerektirmez)
_SRC_DIR = str(Path(__file__).resolve().parent.parent)


def _now() -> datetime:
    return datetime.now(UTC)


def worker_python(kind: str) -> str:
    """İş türüne göre worker yorumlayıcısı (ADR-0005 izole env stratejisi).

    Sıra: STUDIO_WORKER_PYTHONS override → ~/LocalAIStudio/envs/<kind> venv'i →
    ana venv (sys.executable). Mevcut sistem ortamlarına asla dokunulmaz.
    """
    settings = get_settings()
    override = settings.worker_pythons.get(kind, "")
    if override:
        return override
    sub = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    candidate = settings.envs_dir / kind / sub
    if candidate.exists():
        return str(candidate)
    return sys.executable


class _WorkerHandle:
    def __init__(self, proc: asyncio.subprocess.Process) -> None:
        self.proc = proc
        self.cancel_requested = False

    @property
    def pid(self) -> int:
        return self.proc.pid


class Scheduler:
    def __init__(self, emit: EmitFn | None = None, concurrency: int = 1) -> None:
        self._emit: EmitFn = emit or (lambda *a, **k: None)
        self._concurrency = concurrency
        self._running: dict[str, _WorkerHandle] = {}
        self._tasks: set[asyncio.Task] = set()
        self._wake: asyncio.Event | None = None
        self._loop_task: asyncio.Task | None = None
        self._stopping = False

    def set_emit(self, emit: EmitFn) -> None:
        self._emit = emit

    # -- yaşam döngüsü -------------------------------------------------------

    async def start(self) -> None:
        self._stopping = False
        self._wake = asyncio.Event()
        self._running = {}
        self._tasks = set()
        await asyncio.to_thread(self._recover)
        self._loop_task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stopping = True
        if self._wake is not None:
            self._wake.set()
        if self._loop_task is not None:
            await self._loop_task
        for handle in list(self._running.values()):
            handle.proc.kill()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    def _recover(self) -> None:
        with session_scope() as session:
            for job in session.query(Job).filter(Job.status == JobStatus.RUNNING):
                job.status = JobStatus.FAILED
                job.error = "uygulama yeniden başlatıldı; iş yarıda kaldı"
                job.finished_at = _now()

    # -- kuyruk döngüsü --------------------------------------------------------

    def notify(self) -> None:
        """Yeni iş kuyruğa girdiğinde route katmanı çağırır."""
        if self._wake is not None:
            self._wake.set()

    async def _loop(self) -> None:
        assert self._wake is not None
        while not self._stopping:
            started = False
            if len(self._running) < self._concurrency:
                job_id = await asyncio.to_thread(self._claim_next)
                if job_id is not None:
                    task = asyncio.create_task(self._run_job(job_id))
                    self._tasks.add(task)
                    task.add_done_callback(self._tasks.discard)
                    started = True
            if not started:
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=0.5)
                except TimeoutError:
                    pass
                self._wake.clear()

    def _claim_next(self) -> str | None:
        """Sıradaki işi RUNNING olarak sahiplenir (tek yazar — ADR-0006)."""
        with session_scope() as session:
            stmt = (
                select(Job)
                .where(Job.status == JobStatus.QUEUED)
                .order_by(Job.priority.desc(), Job.created_at.asc())
            )
            job = session.scalars(stmt).first()
            if job is None:
                return None
            job.status = JobStatus.RUNNING
            job.started_at = _now()
            return job.id

    # -- iş yürütme ------------------------------------------------------------

    async def _run_job(self, job_id: str) -> None:
        with session_scope() as session:
            job = session.get(Job, job_id)
            kind, params = job.kind, dict(job.params)

        self._emit("job.started", id=job_id, kind=kind)
        log.info("job_started", id=job_id, kind=kind)

        child_env = dict(os.environ)
        child_env["PYTHONPATH"] = _SRC_DIR + (
            os.pathsep + child_env["PYTHONPATH"] if child_env.get("PYTHONPATH") else ""
        )
        child_env["PYTHONIOENCODING"] = "utf-8"
        proc = await asyncio.create_subprocess_exec(
            worker_python(kind),
            "-u",
            "-m",
            "studio_inference.worker",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=child_env,
        )
        handle = _WorkerHandle(proc)
        self._running[job_id] = handle
        stderr_task = asyncio.create_task(proc.stderr.read())

        result: dict | None = None
        error = ""
        try:
            payload = json.dumps({"cmd": "run", "job_id": job_id, "kind": kind, "params": params})
            proc.stdin.write((payload + "\n").encode())
            await proc.stdin.drain()

            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                etype = event.get("event")
                if etype == "progress":
                    self._emit(
                        "job.progress",
                        id=job_id,
                        value=event.get("value", 0),
                        message=event.get("message", ""),
                    )
                elif etype == "done":
                    result = event.get("result", {})
                elif etype == "error":
                    error = str(event.get("error", ""))
        finally:
            returncode = await proc.wait()
            stderr_tail = (await stderr_task)[-2000:].decode(errors="replace").strip()
            self._running.pop(job_id, None)

        if handle.cancel_requested:
            status, error = JobStatus.CANCELED, ""
        elif result is not None and returncode == 0:
            status = JobStatus.DONE
        else:
            status = JobStatus.FAILED
            if not error:
                error = f"worker beklenmedik şekilde sonlandı (çıkış kodu {returncode})"
            if stderr_tail:
                error = f"{error}\n{stderr_tail}"

        with session_scope() as session:
            job = session.get(Job, job_id)
            if job is not None:
                job.status = status
                job.result = result or {}
                job.error = error
                job.finished_at = _now()

        topic = {
            JobStatus.DONE: "job.done",
            JobStatus.FAILED: "job.failed",
            JobStatus.CANCELED: "job.canceled",
        }[status]
        self._emit(topic, id=job_id, **({"error": error} if error else {}))
        log.info("job_finished", id=job_id, status=status.value)
        if self._wake is not None:
            self._wake.set()

    # -- kullanıcı eylemleri -----------------------------------------------------

    async def cancel(self, job_id: str) -> bool:
        handle = self._running.get(job_id)
        if handle is not None:
            handle.cancel_requested = True
            handle.proc.kill()  # VRAM'in geri dönmesinin tek garantili yolu
            return True
        return await asyncio.to_thread(self._finish_inactive, job_id, JobStatus.CANCELED)

    def _finish_inactive(self, job_id: str, status: JobStatus) -> bool:
        with session_scope() as session:
            job = session.get(Job, job_id)
            if job is None or job.status not in (JobStatus.QUEUED, JobStatus.PAUSED):
                return False
            job.status = status
            job.finished_at = _now()
        self._emit("job.canceled", id=job_id)
        return True

    async def pause(self, job_id: str) -> bool:
        def _do() -> bool:
            with session_scope() as session:
                job = session.get(Job, job_id)
                if job is None or job.status != JobStatus.QUEUED:
                    return False
                job.status = JobStatus.PAUSED
            return True

        ok = await asyncio.to_thread(_do)
        if ok:
            self._emit("job.paused", id=job_id)
        return ok

    async def resume(self, job_id: str) -> bool:
        def _do() -> bool:
            with session_scope() as session:
                job = session.get(Job, job_id)
                if job is None or job.status != JobStatus.PAUSED:
                    return False
                job.status = JobStatus.QUEUED
            return True

        ok = await asyncio.to_thread(_do)
        if ok:
            self._emit("job.queued", id=job_id)
            self.notify()
        return ok

    def running_pid(self, job_id: str) -> int | None:
        handle = self._running.get(job_id)
        return handle.pid if handle else None


scheduler = Scheduler()
