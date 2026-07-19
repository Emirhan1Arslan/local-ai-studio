"""Scheduler (Faz 3/4): öncelikli kuyruk + kalıcı worker havuzu. Projenin kalbi.

- Sıra: priority DESC, created_at ASC; iş DB'de RUNNING olarak sahiplenilir (claim).
- İş türü başına KALICI worker süreci: model komutlar arasında yüklü kalır (LRU yerine
  tür başına tek worker + idle timeout — Faz 4). İptal/çökme = süreç kill → VRAM
  garanti geri döner; sonraki iş için worker yeniden doğar.
- Worker çökerse (kill -9 dahil) iş FAILED olur, uygulama ayakta kalır.
- Açılışta önceki süreçten RUNNING kalmış işler FAILED'a çekilir (recover).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections import deque
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


# Bazı iş türleri ayrı bir venv gerektirmeyecek kadar hafif ek bağımlılık ister
# (Faz 5 görüntü araçları: torch/transformers zaten "image" venv'inde) — kendi
# venv'i doğurmak yerine mevcut olanı paylaşır (docs/worker-envs.md).
_ENV_ALIAS: dict[str, str] = {"imgtool": "image"}


def worker_python(kind: str) -> str:
    """İş türüne göre worker yorumlayıcısı (ADR-0005 izole env stratejisi).

    Sıra: STUDIO_WORKER_PYTHONS override → ~/LocalAIStudio/envs/<env_kind> venv'i
    (_ENV_ALIAS ile başka bir türün venv'ini paylaşabilir) → ana venv (sys.executable).
    Mevcut sistem ortamlarına asla dokunulmaz.
    """
    settings = get_settings()
    override = settings.worker_pythons.get(kind, "")
    if override:
        return override
    env_kind = _ENV_ALIAS.get(kind, kind)
    sub = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    candidate = settings.envs_dir / env_kind / sub
    if candidate.exists():
        return str(candidate)
    return sys.executable


class _WorkerHandle:
    def __init__(self, kind: str, proc: asyncio.subprocess.Process) -> None:
        self.kind = kind
        self.proc = proc
        self.current_job: str | None = None
        self.cancel_requested = False
        self.last_used = 0.0
        self.stderr_tail: deque[str] = deque(maxlen=40)

    @property
    def pid(self) -> int:
        return self.proc.pid

    @property
    def alive(self) -> bool:
        return self.proc.returncode is None


class Scheduler:
    def __init__(self, emit: EmitFn | None = None, concurrency: int = 1) -> None:
        self._emit: EmitFn = emit or (lambda *a, **k: None)
        self._concurrency = concurrency
        self._workers: dict[str, _WorkerHandle] = {}  # kind → kalıcı worker
        self._running: dict[str, _WorkerHandle] = {}  # job_id → worker
        self._tasks: set[asyncio.Task] = set()
        self._wake: asyncio.Event | None = None
        self._loop_task: asyncio.Task | None = None
        self._stopping = False
        self._on_done: Callable[[str], None] | None = None

    def set_emit(self, emit: EmitFn) -> None:
        self._emit = emit

    def set_on_done(self, fn: Callable[[str], None]) -> None:
        """Başarıyla biten iş için uygulama katmanı kancası (ör. Asset kaydı)."""
        self._on_done = fn

    # -- yaşam döngüsü -------------------------------------------------------

    async def start(self) -> None:
        self._stopping = False
        self._wake = asyncio.Event()
        self._workers = {}
        self._running = {}
        self._tasks = set()
        await asyncio.to_thread(self._recover)
        self._loop_task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """Sınırlı süreli kapanış: geç doğan worker'lar için tekrarlı öldürme taraması.

        stop ile uçuştaki spawn yarışabilir (claim edilmiş iş, tarama geçtikten sonra
        worker'ını kaydeder). Kaçak canlı worker'ın stderr pompası EOF alamayacağı için
        tek turlu gather sonsuza dek bekleyebilirdi — bu yüzden: öldür → kısa bekle →
        kalan varsa yeniden öldür → hâlâ bitmeyen görevleri iptal et.
        """
        self._stopping = True
        if self._wake is not None:
            self._wake.set()
        if self._loop_task is not None:
            await self._loop_task

        for _ in range(3):
            self._kill_all_workers()
            if not self._tasks:
                break
            _done, pending = await asyncio.wait(self._tasks, timeout=5)
            self._tasks = set(pending)
            if not pending:
                break
        if self._tasks:  # son çare: kalanları iptal et
            for task in self._tasks:
                task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks = set()
        self._kill_all_workers()
        self._workers.clear()

    def _kill_all_workers(self) -> None:
        for handle in list(self._workers.values()) + list(self._running.values()):
            if handle.alive:
                handle.proc.kill()

    def _recover(self) -> None:
        with session_scope() as session:
            for job in session.query(Job).filter(Job.status == JobStatus.RUNNING):
                job.status = JobStatus.FAILED
                job.error = "uygulama yeniden başlatıldı; iş yarıda kaldı"
                job.finished_at = _now()

    # -- kuyruk döngüsü --------------------------------------------------------

    def notify(self) -> None:
        if self._wake is not None:
            self._wake.set()

    async def _loop(self) -> None:
        assert self._wake is not None
        while not self._stopping:
            self._reap_idle_workers()
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

    def _reap_idle_workers(self) -> None:
        """Boşta kalan/ölen worker'ları temizle (idle timeout = model cache ömrü)."""
        timeout = get_settings().worker_idle_timeout
        now = asyncio.get_running_loop().time()
        for kind, handle in list(self._workers.items()):
            if handle.current_job is not None:
                continue
            if not handle.alive:
                self._workers.pop(kind, None)
            elif now - handle.last_used > timeout:
                log.info("worker_idle_killed", kind=kind, pid=handle.pid)
                handle.proc.kill()
                self._workers.pop(kind, None)

    def _claim_next(self) -> str | None:
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

    # -- worker havuzu -----------------------------------------------------------

    async def _spawn_worker(self, kind: str) -> _WorkerHandle:
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
            # Varsayılan 64 KiB satır tamponu, bir kütüphane stdout'a \r'li ilerleme
            # yazarsa (newline'sız uzun satır) "Separator is not found" ile patlar
            # (canlıda EasyOCR'ın ilk indirmesinde yakalandı) — savunma amaçlı büyütüldü.
            limit=4 * 1024 * 1024,
        )
        handle = _WorkerHandle(kind, proc)
        handle.last_used = asyncio.get_running_loop().time()
        pump = asyncio.create_task(self._pump_stderr(handle))
        self._tasks.add(pump)
        pump.add_done_callback(self._tasks.discard)
        log.info("worker_spawned", kind=kind, pid=proc.pid, python=worker_python(kind))
        return handle

    async def _pump_stderr(self, handle: _WorkerHandle) -> None:
        while True:
            line = await handle.proc.stderr.readline()
            if not line:
                break
            handle.stderr_tail.append(line.decode(errors="replace").rstrip())

    async def _ensure_worker(self, kind: str) -> _WorkerHandle:
        handle = self._workers.get(kind)
        # Yeniden kullanım yalnızca: canlı + boşta + iptal bayrağı taşımıyor.
        # (kill sonrası returncode asyncio'da gecikmeli set edilir; bayrak bunu kapatır)
        if (
            handle is not None
            and handle.alive
            and handle.current_job is None
            and not handle.cancel_requested
        ):
            return handle
        handle = await self._spawn_worker(kind)
        self._workers[kind] = handle  # eski kayıt üzerine yazılır; kendi işi bitince temizlenir
        return handle

    # -- iş yürütme ------------------------------------------------------------

    async def _send_command(self, kind: str, payload: str) -> _WorkerHandle:
        """Komutu gönder; worker aradan ölmüşse bir kez yeniden doğur."""
        for attempt in (1, 2):
            handle = await self._ensure_worker(kind)
            try:
                handle.proc.stdin.write((payload + "\n").encode())
                await handle.proc.stdin.drain()
                return handle
            except (BrokenPipeError, ConnectionResetError, OSError):
                self._workers.pop(kind, None)
                if attempt == 2:
                    raise
        raise RuntimeError("erişilemez")  # for type-checker

    async def _run_job(self, job_id: str) -> None:
        with session_scope() as session:
            job = session.get(Job, job_id)
            kind, params = job.kind, dict(job.params)

        self._emit("job.started", id=job_id, kind=kind)
        log.info("job_started", id=job_id, kind=kind)

        result: dict | None = None
        error = ""
        eof = False
        handle: _WorkerHandle | None = None
        try:
            payload = json.dumps({"cmd": "run", "job_id": job_id, "kind": kind, "params": params})
            handle = await self._send_command(kind, payload)
            handle.current_job = job_id
            self._running[job_id] = handle

            while True:
                line = await handle.proc.stdout.readline()
                if not line:
                    eof = True  # worker öldü (crash / kill / iptal)
                    break
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                etype = event.get("event")
                if etype == "progress":
                    extra = {"preview": event["preview"]} if event.get("preview") else {}
                    self._emit(
                        "job.progress",
                        id=job_id,
                        value=event.get("value", 0),
                        message=event.get("message", ""),
                        **extra,
                    )
                elif etype == "done":
                    result = event.get("result", {})
                    break
                elif etype == "error":
                    error = str(event.get("error", ""))
                    break
        except Exception as exc:  # spawn/IPC hatası
            error = f"worker başlatılamadı: {exc}"
        finally:
            self._running.pop(job_id, None)
            if handle is not None:
                if eof:
                    # returncode'un gerçekten set edilmesini bekle: 'alive' kontrolleri
                    # kill sonrası gecikmeli reaping yüzünden yanlış pozitif vermesin
                    try:
                        await asyncio.wait_for(handle.proc.wait(), timeout=10)
                    except TimeoutError:
                        pass
                handle.current_job = None
                handle.last_used = asyncio.get_running_loop().time()
                # Havuzdan düşmüş/üzerine yazılmışsa sızıntı olmasın: öldür
                if handle.alive and self._workers.get(kind) is not handle:
                    handle.proc.kill()

        canceled = handle is not None and handle.cancel_requested
        if handle is not None and not handle.alive:
            # Yalnızca KENDİ handle'ımızı havuzdan düşür; bu arada doğmuş yeni worker kalsın
            if self._workers.get(kind) is handle:
                self._workers.pop(kind, None)
            handle.cancel_requested = False

        if canceled:
            status, error = JobStatus.CANCELED, ""
        elif result is not None:
            status = JobStatus.DONE
        else:
            status = JobStatus.FAILED
            if not error:
                rc = handle.proc.returncode if handle is not None else "?"
                error = f"worker beklenmedik şekilde sonlandı (çıkış kodu {rc})"
                tail = "\n".join(list(handle.stderr_tail)[-5:]) if handle is not None else ""
                if tail:
                    error = f"{error}\n{tail}"

        with session_scope() as session:
            job = session.get(Job, job_id)
            if job is not None:
                job.status = status
                job.result = result or {}
                job.error = error
                job.finished_at = _now()

        if status == JobStatus.DONE and self._on_done is not None:
            try:
                await asyncio.to_thread(self._on_done, job_id)
            except Exception:
                log.exception("on_done_hook_failed", id=job_id)

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

    def worker_pid(self, kind: str) -> int | None:
        handle = self._workers.get(kind)
        return handle.pid if handle is not None and handle.alive else None


scheduler = Scheduler()
