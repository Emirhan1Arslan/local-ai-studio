"""Ortak fixture'lar: izole veri dizini + TestClient."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDIO_DATA_DIR", str(tmp_path / "LocalAIStudio"))
    # Singleton'ları test başına sıfırla
    import studio_core.config as cfg
    import studio_core.db as db

    cfg._settings = None
    db._engine = None
    db._session_factory = None

    from studio_api.app import create_app

    with TestClient(create_app()) as c:
        yield c
