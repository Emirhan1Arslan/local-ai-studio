"""watchdog ile models/ dizinini canlı izleme: yeni model restart'sız algılanır (Faz 2 DoD).

Olay fırtınasına karşı debounce: art arda gelen değişiklikler tek taramada toplanır.
Callback watchdog thread'inde çalışır; asyncio tarafına geçiş çağıranın sorumluluğudur.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from studio_core.config import get_settings
from studio_core.logging import get_logger
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from studio_models.metadata import MODEL_EXTS

log = get_logger("studio_models.watcher")


class _Handler(FileSystemEventHandler):
    def __init__(self, trigger: Callable[[], None]) -> None:
        self._trigger = trigger

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        paths = [getattr(event, "dest_path", ""), event.src_path]
        if any(str(p).lower().endswith(tuple(MODEL_EXTS)) for p in paths if p):
            self._trigger()


class ModelWatcher:
    def __init__(self, on_change: Callable[[], None], debounce_sec: float = 1.5) -> None:
        self._on_change = on_change
        self._debounce = debounce_sec
        self._observer: Observer | None = None
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def _trigger(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._debounce, self._fire)
            self._timer.daemon = True
            self._timer.start()

    def _fire(self) -> None:
        try:
            self._on_change()
        except Exception:
            log.exception("model_watcher_callback_failed")

    def start(self) -> None:
        root = get_settings().models_dir
        self._observer = Observer(timeout=1)
        self._observer.schedule(_Handler(self._trigger), str(root), recursive=True)
        self._observer.daemon = True
        self._observer.start()
        log.info("model_watcher_started", path=str(root))

    def stop(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
