"""İndirme motoru (Faz 2): resume (HTTP Range), sha256 doğrulama, pause/resume/cancel.

Tasarım:
- Her indirme `<hedef>.part` dosyasına akar; bitince atomik `os.replace` ile yerine taşınır.
  Hedef models/ altındaysa watchdog dosyayı görür ve registry kendiliğinden güncellenir.
- Resume: .part varsa `Range: bytes=N-` gönderilir; sunucu 200 dönerse (Range desteklemiyor)
  baştan başlanır. Hash, .part'ın mevcut kısmı yeniden okunarak sürdürülür.
- Pause = task iptali + .part korunur; Cancel = task iptali + .part silinir.
- Süreç yeniden başlarsa 'downloading' kalmış kayıtlar 'paused'a çekilir (recover()).
"""

from __future__ import annotations

import asyncio
import hashlib
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import httpx
from studio_core.db import session_scope
from studio_core.logging import get_logger
from studio_core.models import Download, DownloadStatus

log = get_logger("studio_models.downloader")

CHUNK_SIZE = 1024 * 1024
PROGRESS_INTERVAL = 0.5  # sn
DISK_MARGIN_BYTES = 500 * 1024 * 1024  # hedef diskte bırakılacak pay

# Olay yayını: app katmanı bus.emit'i bağlar; testlerde no-op olabilir.
EmitFn = Callable[..., None]


class DownloadError(Exception):
    pass


def _update(download_id: str, **fields) -> None:
    with session_scope() as session:
        rec = session.get(Download, download_id)
        if rec is not None:
            for k, v in fields.items():
                setattr(rec, k, v)


def _get(download_id: str) -> Download | None:
    with session_scope() as session:
        return session.get(Download, download_id)


class DownloadManager:
    def __init__(
        self,
        emit: EmitFn | None = None,
        concurrency: int = 2,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._emit: EmitFn = emit or (lambda *a, **k: None)
        self._concurrency = concurrency
        self._sem = asyncio.Semaphore(concurrency)
        self._tasks: dict[str, asyncio.Task] = {}
        self._pause_requested: set[str] = set()
        self._transport = transport  # testlerde ASGITransport enjekte edilir

    def set_emit(self, emit: EmitFn) -> None:
        self._emit = emit

    def reset(self) -> None:
        """Yeni event loop'a bağlan (açılışta çağrılır): semafor ve task tablosu tazelenir."""
        self._sem = asyncio.Semaphore(self._concurrency)
        self._tasks = {}
        self._pause_requested = set()

    # -- kuyruk yönetimi ---------------------------------------------------

    def recover(self) -> None:
        """Açılışta: süreçle birlikte ölen 'downloading' kayıtlarını paused'a çek."""
        with session_scope() as session:
            for rec in session.query(Download).filter(
                Download.status == DownloadStatus.DOWNLOADING
            ):
                rec.status = DownloadStatus.PAUSED

    async def enqueue(
        self,
        url: str,
        dest: Path,
        *,
        source: str = "url",
        sha256: str = "",
        headers: dict[str, str] | None = None,
    ) -> str:
        with session_scope() as session:
            rec = Download(
                url=url,
                source=source,
                dest_path=str(dest),
                sha256_expected=sha256.lower(),
                status=DownloadStatus.QUEUED,
            )
            session.add(rec)
            session.flush()
            download_id = rec.id
        self._emit("download.queued", id=download_id, url=url, dest=str(dest))
        self._start_task(download_id, headers or {})
        return download_id

    def _start_task(self, download_id: str, headers: dict[str, str]) -> None:
        self._tasks[download_id] = asyncio.create_task(self._run(download_id, headers))

    async def pause(self, download_id: str) -> bool:
        task = self._tasks.get(download_id)
        if task is None or task.done():
            return False
        self._pause_requested.add(download_id)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return True

    async def resume(self, download_id: str, headers: dict[str, str] | None = None) -> bool:
        rec = _get(download_id)
        if rec is None or rec.status not in (DownloadStatus.PAUSED, DownloadStatus.FAILED):
            return False
        self._start_task(download_id, headers or {})
        return True

    async def cancel(self, download_id: str) -> bool:
        rec = _get(download_id)
        if rec is None:
            return False
        task = self._tasks.get(download_id)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        Path(rec.dest_path + ".part").unlink(missing_ok=True)
        _update(download_id, status=DownloadStatus.CANCELED, finished_at=datetime.now(UTC))
        self._emit("download.canceled", id=download_id)
        return True

    # -- indirme döngüsü ----------------------------------------------------

    async def _run(self, download_id: str, headers: dict[str, str]) -> None:
        try:
            async with self._sem:
                await self._download(download_id, headers)
        except asyncio.CancelledError:
            if download_id in self._pause_requested:
                self._pause_requested.discard(download_id)
                _update(download_id, status=DownloadStatus.PAUSED)
                self._emit("download.paused", id=download_id)
            raise
        except Exception as exc:  # ağ/IO hataları kayda yazılır, süreci düşürmez
            log.warning("download_failed", id=download_id, error=str(exc))
            _update(
                download_id,
                status=DownloadStatus.FAILED,
                error=str(exc),
                finished_at=datetime.now(UTC),
            )
            self._emit("download.failed", id=download_id, error=str(exc))

    async def _download(self, download_id: str, headers: dict[str, str]) -> None:
        rec = _get(download_id)
        if rec is None:
            return
        dest = Path(rec.dest_path)
        part = Path(rec.dest_path + ".part")
        dest.parent.mkdir(parents=True, exist_ok=True)

        resume_from = part.stat().st_size if part.exists() else 0
        req_headers = dict(headers)
        if resume_from:
            req_headers["Range"] = f"bytes={resume_from}-"

        _update(download_id, status=DownloadStatus.DOWNLOADING, error="")

        hasher = hashlib.sha256()
        if resume_from:
            # mevcut kısmı hash'e dahil et
            with part.open("rb") as f:
                while chunk := f.read(CHUNK_SIZE):
                    hasher.update(chunk)

        client_kwargs: dict = {"follow_redirects": True, "timeout": httpx.Timeout(30, read=120)}
        if self._transport is not None:
            client_kwargs["transport"] = self._transport

        async with httpx.AsyncClient(**client_kwargs) as client:
            async with client.stream("GET", rec.url, headers=req_headers) as resp:
                if resume_from and resp.status_code == 200:
                    # sunucu Range desteklemiyor → baştan
                    resume_from = 0
                    hasher = hashlib.sha256()
                elif resume_from and resp.status_code != 206:
                    resp.raise_for_status()
                resp.raise_for_status()

                total = resume_from + int(resp.headers.get("content-length", 0) or 0)
                free = shutil.disk_usage(dest.parent).free
                if total and free < (total - resume_from) + DISK_MARGIN_BYTES:
                    raise DownloadError(
                        f"disk alanı yetersiz: {free // 2**20} MB boş, "
                        f"{(total - resume_from) // 2**20} MB gerekli"
                    )

                downloaded = resume_from
                _update(download_id, total_bytes=total, downloaded_bytes=downloaded)
                last_emit = 0.0
                mode = "ab" if resume_from else "wb"
                with part.open(mode) as f:
                    async for chunk in resp.aiter_bytes(CHUNK_SIZE):
                        f.write(chunk)
                        hasher.update(chunk)
                        downloaded += len(chunk)
                        now = asyncio.get_running_loop().time()
                        if now - last_emit >= PROGRESS_INTERVAL:
                            last_emit = now
                            _update(download_id, downloaded_bytes=downloaded, total_bytes=total)
                            self._emit(
                                "download.progress",
                                id=download_id,
                                downloaded=downloaded,
                                total=total,
                            )

        actual = hasher.hexdigest()
        if rec.sha256_expected and actual != rec.sha256_expected:
            part.unlink(missing_ok=True)
            raise DownloadError(
                f"sha256 uyuşmuyor: beklenen {rec.sha256_expected[:12]}…, gelen {actual[:12]}…"
            )

        part.replace(dest)
        _update(
            download_id,
            status=DownloadStatus.DONE,
            downloaded_bytes=downloaded,
            total_bytes=total or downloaded,
            sha256_actual=actual,
            finished_at=datetime.now(UTC),
        )
        self._emit("download.done", id=download_id, dest=str(dest), sha256=actual)
        log.info("download_done", id=download_id, dest=str(dest))


manager = DownloadManager()


def hf_resolve_url(repo_id: str, filename: str, revision: str = "main") -> str:
    """Hugging Face Hub dosya URL'i. Token gerekiyorsa Authorization header'ı eklenir."""
    return f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"


async def hf_list_repo_files(
    repo_id: str, transport: httpx.AsyncBaseTransport | None = None
) -> list[str]:
    """HF repo'sundaki dosya adlarını döner (çok dosyalı snapshot indirmeleri için).

    transport parametresi, tek-dosya indirme motoruyla aynı test enjeksiyon noktasını
    paylaşır — testlerde manager._transport (ASGITransport) buraya da verilir.
    """
    client_kwargs: dict = {"timeout": httpx.Timeout(20)}
    if transport is not None:
        client_kwargs["transport"] = transport
    async with httpx.AsyncClient(**client_kwargs) as client:
        resp = await client.get(f"https://huggingface.co/api/models/{repo_id}")
    if resp.status_code != 200:
        raise DownloadError(f"HF repo bulunamadı: {repo_id!r} ({resp.status_code})")
    return [s["rfilename"] for s in resp.json().get("siblings", [])]
