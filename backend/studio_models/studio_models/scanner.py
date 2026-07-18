"""Model dizini tarayıcı: dosya sistemi ↔ registry (DB) senkronu.

Kural: dosya sistemi gerçeğin kaynağıdır. Yeni dosya → kayıt eklenir; kaybolan dosya →
kayıt silinir; boyutu değişen dosya → metadata yeniden çıkarılır.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import select
from studio_core.config import get_settings
from studio_core.db import session_scope
from studio_core.logging import get_logger
from studio_core.models import ModelRecord

from studio_models.metadata import MODEL_EXTS, extract

log = get_logger("studio_models.scanner")


@dataclass
class ScanResult:
    added: list[dict] = field(default_factory=list)
    removed: list[dict] = field(default_factory=list)
    updated: list[dict] = field(default_factory=list)
    total: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.added or self.removed or self.updated)


def _record_summary(rec: ModelRecord) -> dict:
    return {"id": rec.id, "name": rec.name, "kind": rec.kind, "path": rec.path}


def scan() -> ScanResult:
    """Modeller dizinini tarar ve registry'yi senkronlar."""
    settings = get_settings()
    settings.ensure_dirs()
    root = settings.models_dir

    found: dict[str, Path] = {
        str(p): p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in MODEL_EXTS
    }

    result = ScanResult()
    with session_scope() as session:
        existing = {r.path: r for r in session.scalars(select(ModelRecord)).all()}

        for path_str, rec in existing.items():
            if path_str not in found:
                result.removed.append(_record_summary(rec))
                session.delete(rec)

        for path_str, path in found.items():
            rec = existing.get(path_str)
            if rec is None:
                data = extract(path, root)
                rec = ModelRecord(**data)
                session.add(rec)
                session.flush()
                result.added.append(_record_summary(rec))
            elif rec.size_bytes != path.stat().st_size:
                data = extract(path, root)
                for k, v in data.items():
                    setattr(rec, k, v)
                result.updated.append(_record_summary(rec))

        result.total = len(found)

    if result.changed:
        log.info(
            "model_scan",
            added=len(result.added),
            removed=len(result.removed),
            updated=len(result.updated),
            total=result.total,
        )
    return result
