"""Domain ORM modelleri (Faz 1 iskeleti): Setting, Project, Asset, ModelRecord, Job.

Asset Graph ilişkisi (önemli.md): her asset bir üretim ağının düğümüdür; parent_id ile
türev zinciri kurulur (karakter → portrait/icon/sprite/mesh/voice…). Faz 4'te genişletilir.
"""

from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Setting(Base):
    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[dict] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    assets: Mapped[list[Asset]] = relationship(back_populates="project")


class Asset(Base):
    """Asset Graph düğümü: parent_id → türetildiği ana varlık."""

    __tablename__ = "assets"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    parent_id: Mapped[str | None] = mapped_column(ForeignKey("assets.id"), nullable=True)
    kind: Mapped[str] = mapped_column(String(32))  # image/video/audio/mesh/text/…
    role: Mapped[str] = mapped_column(String(32), default="")  # portrait/icon/sprite/voice/…
    path: Mapped[str] = mapped_column(Text)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)  # prompt, seed, model, parametreler
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    project: Mapped[Project] = relationship(back_populates="assets")
    parent: Mapped[Asset | None] = relationship(remote_side=[id])


class ModelRecord(Base):
    """Model registry (Faz 2'de watchdog + metadata çıkarımıyla beslenir)."""

    __tablename__ = "models"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255))
    family: Mapped[str] = mapped_column(String(64), default="")  # sdxl/flux/qwen/…
    kind: Mapped[str] = mapped_column(String(32))  # checkpoint/lora/vae/controlnet/llm/…
    format: Mapped[str] = mapped_column(String(16), default="")  # safetensors/gguf/onnx/pickle
    path: Mapped[str] = mapped_column(Text)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    sha256: Mapped[str] = mapped_column(String(64), default="")
    license: Mapped[str] = mapped_column(String(128), default="")
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class DownloadStatus(enum.StrEnum):
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    PAUSED = "paused"
    DONE = "done"
    FAILED = "failed"
    CANCELED = "canceled"


class Download(Base):
    """İndirme kaydı: süreç yeniden başlasa da .part dosyasından sürdürülebilir."""

    __tablename__ = "downloads"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    url: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(16), default="url")  # url/hf/civitai
    dest_path: Mapped[str] = mapped_column(Text)
    status: Mapped[DownloadStatus] = mapped_column(
        Enum(DownloadStatus, values_callable=lambda e: [m.value for m in e]),
        default=DownloadStatus.QUEUED,
    )
    total_bytes: Mapped[int] = mapped_column(Integer, default=0)
    downloaded_bytes: Mapped[int] = mapped_column(Integer, default=0)
    sha256_expected: Mapped[str] = mapped_column(String(64), default="")
    sha256_actual: Mapped[str] = mapped_column(String(64), default="")
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class ChatSession(Base):
    """Faz 6: sohbet oturumu — model/sistem prompt'u sabitlenir, mesajlar ayrı tabloda."""

    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    title: Mapped[str] = mapped_column(String(255), default="Yeni sohbet")
    model_path: Mapped[str] = mapped_column(Text, default="")
    system_prompt: Mapped[str] = mapped_column(Text, default="")
    project_id: Mapped[str | None] = mapped_column(ForeignKey("projects.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)

    messages: Mapped[list[ChatMessage]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.created_at"
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(ForeignKey("chat_sessions.id"))
    role: Mapped[str] = mapped_column(String(16))  # system/user/assistant
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    session: Mapped[ChatSession] = relationship(back_populates="messages")


class JobStatus(enum.StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELED = "canceled"
    PAUSED = "paused"


class Job(Base):
    """İş kuyruğu kaydı. Durum makinesi: queued → running → done|failed|canceled|paused."""

    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    kind: Mapped[str] = mapped_column(String(32))  # image/llm/video/audio/3d/tool
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, values_callable=lambda e: [m.value for m in e]),
        default=JobStatus.QUEUED,
    )
    priority: Mapped[int] = mapped_column(Integer, default=0)
    project_id: Mapped[str | None] = mapped_column(ForeignKey("projects.id"), nullable=True)
    owner: Mapped[str] = mapped_column(String(64), default="ui")  # ui/api istemci kimliği
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    result: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
