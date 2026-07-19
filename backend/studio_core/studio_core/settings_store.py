"""Kalıcı ayar override'ları (Faz 1 cila): DB'de saklanır, süreç başında yüklenir.

Ortam değişkenleri (STUDIO_*) ilk açılışta okunur; bu katman kullanıcının UI'dan
değiştirdiği değerleri DB'ye yazar ve get_settings() tekil nesnesine uygular.
Zaten her kullanım noktasında get_settings() TAZE okunduğu için (scheduler,
telemetri, indirme motoru) — bu alanlar restart gerektirmeden canlı etkili olur.
"""

from __future__ import annotations

from typing import Any

from studio_core.config import get_settings
from studio_core.db import session_scope
from studio_core.models import Setting

EDITABLE_KEYS = {
    "hf_token",
    "telemetry_interval",
    "worker_idle_timeout",
    "worker_concurrency",
    "download_speed_limit_mbps",
}

_CASTERS = {
    "hf_token": str,
    "telemetry_interval": float,
    "worker_idle_timeout": float,
    "worker_concurrency": int,
    "download_speed_limit_mbps": float,
}


def load_overrides() -> None:
    """Süreç başlarken DB'deki override'ları get_settings() nesnesine uygular."""
    settings = get_settings()
    with session_scope() as session:
        rows = session.query(Setting).filter(Setting.key.in_(EDITABLE_KEYS)).all()
        for row in rows:
            cast = _CASTERS[row.key]
            setattr(settings, row.key, cast(row.value.get("v")))


def save_override(key: str, value: Any) -> None:
    if key not in EDITABLE_KEYS:
        raise ValueError(f"düzenlenemez ayar: {key!r}")
    with session_scope() as session:
        row = session.get(Setting, key)
        if row is None:
            session.add(Setting(key=key, value={"v": value}))
        else:
            row.value = {"v": value}
