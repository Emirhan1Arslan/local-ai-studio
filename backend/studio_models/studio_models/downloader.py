"""İndirme motoru (Faz 2): resume (HTTP Range), sha256 doğrulama, pause/resume/cancel.

Tasarım:
- Her indirme `<hedef>.part` dosyasına akar; bitince atomik `os.replace` ile yerine taşınır.
  Hedef models/ altındaysa watchdog dosyayı görür ve registry kendiliğinden güncellenir.
- Resume: .part varsa `Range: bytes=N-` gönderilir; sunucu 200 dönerse (Range desteklemiyor)
  baştan başlanır. Hash, .part'ın mevcut kısmı yeniden okunarak sürdürülür.
- Pause = task iptali + .part korunur; Cancel = task iptali + .part silinir.
- Süreç yeniden başlarsa 'downloading' kalmış kayıtlar 'paused'a çekilir (recover()).
- Hız sınırı (Faz 8'de eklendi, kullanıcı isteğiyle): tüm eşzamanlı indirmeler
  arasında PAYLAŞILAN tek bir _RateLimiter — Ayarlar/İndirmeler panelinden
  (settings.download_speed_limit_mbps) canlı değiştirilebilir, restart gerekmez.
"""

from __future__ import annotations

import asyncio
import hashlib
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import httpx
from studio_core.config import get_settings
from studio_core.db import session_scope
from studio_core.logging import get_logger
from studio_core.models import Download, DownloadStatus

log = get_logger("studio_models.downloader")

CHUNK_SIZE = 1024 * 1024
PROGRESS_INTERVAL = 0.5  # sn
DISK_MARGIN_BYTES = 500 * 1024 * 1024  # hedef diskte bırakılacak pay


class _RateLimiter:
    """Sanal zaman damgalı ortak kova: her indirme chunk yazdıktan sonra bunu
    çağırır, toplam hız (tüm eşzamanlı indirmeler dahil) ayarlanan sınırı
    aşarsa uyur. Sınır her çağrıda settings'ten TAZE okunur — kullanıcı
    Ayarlar/İndirmeler panelinden değiştirdiğinde bir sonraki chunk'ta hemen
    etkili olur."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._next_time = 0.0

    async def throttle(self, n_bytes: int) -> None:
        limit_mbps = get_settings().download_speed_limit_mbps
        if limit_mbps <= 0 or n_bytes <= 0:
            return
        bytes_per_sec = limit_mbps * 1024 * 1024
        async with self._lock:
            now = asyncio.get_running_loop().time()
            start = max(now, self._next_time)
            self._next_time = start + n_bytes / bytes_per_sec
            delay = start - now
        if delay > 0:
            await asyncio.sleep(delay)

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
        self._rate_limiter = _RateLimiter()

    def set_emit(self, emit: EmitFn) -> None:
        self._emit = emit

    def reset(self) -> None:
        """Yeni event loop'a bağlan (açılışta çağrılır): semafor ve task tablosu tazelenir."""
        self._sem = asyncio.Semaphore(self._concurrency)
        self._tasks = {}
        self._pause_requested = set()
        self._rate_limiter = _RateLimiter()  # Lock eski event loop'a bağlıydı

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
        existing = self._tasks.get(download_id)
        if existing is not None and not existing.done():
            # Zaten çalışan bir task var — ikinci kez başlatırsak aynı .part dosyasına
            # iki coroutine birden yazar ve dosya bozulur (arka arkaya iki resume çağrısı,
            # ör. toplu 'devam ettir' + ayrı bir yeniden deneme aynı ID'yi vurursa).
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

        if dest.exists() and not part.exists():
            # Dosya diskte zaten var: önceki denemede part->dest rename başarılı olmuş
            # ama kayıt DONE'a çevrilmeden kesilmiş olmalı (crash/cancel tam o arada).
            # .part yoksa resume için ağa gidilemez zaten. Ama körü körüne güvenme:
            # bilinen total_bytes varsa boyutu doğrula — uyuşmuyorsa dosya bozuk demektir
            # (ör. eşzamanlı iki resume aynı .part'a yazmış olabilir), sil ve baştan indir.
            size = dest.stat().st_size
            if not rec.total_bytes or size == rec.total_bytes:
                _update(
                    download_id,
                    status=DownloadStatus.DONE,
                    downloaded_bytes=size,
                    total_bytes=size,
                    finished_at=datetime.now(UTC),
                )
                self._emit("download.done", id=download_id, dest=str(dest), sha256="")
                log.info("download_already_on_disk", id=download_id, dest=str(dest))
                return
            log.warning(
                "download_dest_size_mismatch",
                id=download_id,
                dest=str(dest),
                actual=size,
                expected=rec.total_bytes,
            )
            dest.unlink()

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

        total = resume_from
        downloaded = resume_from
        already_complete = False

        async with httpx.AsyncClient(**client_kwargs) as client:
            async with client.stream("GET", rec.url, headers=req_headers) as resp:
                if resume_from and resp.status_code == 200:
                    # sunucu Range desteklemiyor → baştan
                    resume_from = 0
                    downloaded = 0
                    hasher = hashlib.sha256()
                elif resume_from and resp.status_code == 416:
                    # .part zaten resume_from kadar bayt içeriyor ve sunucu "eklenecek
                    # bayt yok" diyor: son chunk yazılmış ama rename'e sıra gelmeden
                    # (pause/crash/çakışan resume) kesilmiş olmalı. Content-Range ile
                    # gerçek boyutu doğrula, eşleşiyorsa dosya zaten tamamlanmış say —
                    # eşleşmiyorsa .part bozuk demektir, sıfırdan indir.
                    content_range = resp.headers.get("content-range", "")
                    true_size: int | None = None
                    if content_range.startswith("bytes */"):
                        try:
                            true_size = int(content_range.split("/")[-1])
                        except ValueError:
                            true_size = None
                    if true_size == resume_from:
                        already_complete = True
                        total = resume_from
                        log.info("download_already_complete_on_resume", id=download_id)
                    else:
                        part.unlink(missing_ok=True)
                        raise DownloadError(
                            f"416 Range Not Satisfiable ve .part boyutu ({resume_from}) "
                            f"sunucudaki gerçek boyutla ({true_size}) uyuşmuyor — .part silindi, "
                            "yeniden indirilmesi gerekiyor"
                        )
                elif resume_from and resp.status_code != 206:
                    resp.raise_for_status()

                if not already_complete:
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
                            await self._rate_limiter.throttle(len(chunk))
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
) -> list[dict]:
    """HF repo'sundaki dosyaları (ad + bayt cinsinden boyut) döner (çok dosyalı
    snapshot indirmeleri ve Keşfet panelinin dosya listesi için).

    transport parametresi, tek-dosya indirme motoruyla aynı test enjeksiyon noktasını
    paylaşır — testlerde manager._transport (ASGITransport) buraya da verilir.
    """
    client_kwargs: dict = {"timeout": httpx.Timeout(20)}
    if transport is not None:
        client_kwargs["transport"] = transport
    async with httpx.AsyncClient(**client_kwargs) as client:
        resp = await client.get(
            f"https://huggingface.co/api/models/{repo_id}", params={"blobs": "true"}
        )
    if resp.status_code != 200:
        raise DownloadError(f"HF repo bulunamadı: {repo_id!r} ({resp.status_code})")
    return [
        {"filename": s["rfilename"], "size": s.get("size") or 0}
        for s in resp.json().get("siblings", [])
    ]


# safetensors dtype adı → bayt genişliği; büyüklük tahmini için (bilinmeyen tip: 2 varsayılır)
_DTYPE_BYTES = {
    "F64": 8,
    "I64": 8,
    "F32": 4,
    "I32": 4,
    "F16": 2,
    "BF16": 2,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}


def estimate_safetensors_bytes(safetensors: dict) -> int:
    """HF'nin search yanıtındaki safetensors.parameters (dtype → parametre sayısı)
    dökümünden yaklaşık disk boyutu hesaplar (gerçek dosya boyutuyla ~aynı)."""
    parameters = safetensors.get("parameters") or {}
    return sum(count * _DTYPE_BYTES.get(dtype, 2) for dtype, count in parameters.items())


async def hf_search_models(
    query: str,
    pipeline_tag: str = "",
    limit: int = 30,
    transport: httpx.AsyncBaseTransport | None = None,
) -> list[dict]:
    """HF Hub'da model arar (Keşfet paneli). Ham HF şemasını döner; rota katmanı
    kendi çıktı modeline dönüştürür.

    expand[] alanları tek tek verilmezse HF yalnızca id/_id/trendingScore döner —
    ihtiyaç duyulan her alan (safetensors dahil, boyut tahmini için) açıkça istenir.
    """
    client_kwargs: dict = {"timeout": httpx.Timeout(20)}
    if transport is not None:
        client_kwargs["transport"] = transport
    params: list[tuple[str, str | int]] = [
        ("sort", "downloads"),
        ("direction", -1),
        ("limit", limit),
    ]
    if query:
        params.append(("search", query))
    if pipeline_tag:
        params.append(("pipeline_tag", pipeline_tag))
    for field in (
        "safetensors",
        "pipeline_tag",
        "downloads",
        "likes",
        "tags",
        "library_name",
        "gated",
    ):
        params.append(("expand[]", field))
    async with httpx.AsyncClient(**client_kwargs) as client:
        resp = await client.get("https://huggingface.co/api/models", params=params)
    if resp.status_code != 200:
        raise DownloadError(f"HF arama başarısız: {resp.status_code}")
    return resp.json()
