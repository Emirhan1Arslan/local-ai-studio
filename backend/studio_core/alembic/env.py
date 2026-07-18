"""Alembic ortamı: URL'i studio_core.Settings'ten alır, metadata'yı ORM'den okur."""

from __future__ import annotations

from alembic import context
from sqlalchemy import create_engine
from studio_core.config import get_settings
from studio_core.models import Base

target_metadata = Base.metadata


def _db_url() -> str:
    settings = get_settings()
    settings.ensure_dirs()
    return f"sqlite:///{settings.db_path}"


def run_migrations_offline() -> None:
    context.configure(
        url=_db_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        render_as_batch=True,  # SQLite ALTER kısıtları için batch modu
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(_db_url())
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
        )
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
