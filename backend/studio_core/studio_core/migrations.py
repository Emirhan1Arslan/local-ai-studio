"""Programatik migration: uygulama açılışında şemayı head'e taşır.

Benimseme kuralı: DB, Alembic öncesi bir sürümle (create_all) oluşturulmuşsa —
yani tablolar var ama alembic_version yok — şema ilk migration'la birebir aynı
olduğundan DB 'head' olarak damgalanır ve sonraki migration'lar normal uygulanır.
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text

from studio_core.db import get_engine

_INI = Path(__file__).resolve().parent.parent / "alembic.ini"


def _current_revision(tables: set[str]) -> str | None:
    if "alembic_version" not in tables:
        return None
    with get_engine().connect() as conn:
        row = conn.execute(text("SELECT version_num FROM alembic_version")).fetchone()
    return row[0] if row else None


def upgrade_to_head() -> None:
    cfg = Config(str(_INI))
    tables = set(inspect(get_engine()).get_table_names())
    if _current_revision(tables) is None and "jobs" in tables:
        # Alembic öncesi (create_all) DB: şema ilk migration ile aynı → damgala
        command.stamp(cfg, "head")
    else:
        command.upgrade(cfg, "head")
