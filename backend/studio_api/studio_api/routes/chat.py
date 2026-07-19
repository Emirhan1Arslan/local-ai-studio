"""Sohbet rotaları (Faz 6): oturum CRUD + mesaj gönderme (bir 'llm' işi tetikler).

Yanıt metni WS üzerinden job.progress olaylarının 'delta' alanıyla akar (scheduler'ın
genel ek-alan geçirme mekanizması, bkz. studio_inference.scheduler); iş bitince asistan
mesajı otomatik DB'ye yazılır (studio_core.chat.save_chat_message_for_job, on_done kancası).
"""

from __future__ import annotations

import asyncio
from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel, Field
from sqlalchemy import select
from studio_core.db import session_scope
from studio_core.models import ChatMessage, ChatSession, Job, JobStatus
from studio_inference.scheduler import scheduler

router = APIRouter(prefix="/v1/chat", tags=["chat"])


class SessionCreate(BaseModel):
    model_path: str
    title: str = "Yeni sohbet"
    system_prompt: str = ""


class SessionOut(BaseModel):
    id: str
    title: str
    model_path: str
    system_prompt: str
    created_at: datetime
    updated_at: datetime


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    created_at: datetime


class MessageCreate(BaseModel):
    content: str
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=1024, ge=1, le=8192)


class MessageCreateOut(BaseModel):
    job_id: str
    user_message_id: str


def _not_found(session_id: str):
    from studio_api.app import ApiError

    raise ApiError(404, "not_found", f"oturum bulunamadı: {session_id}")


@router.post("")
async def create_session(req: SessionCreate) -> SessionOut:
    def _create() -> SessionOut:
        with session_scope() as session:
            s = ChatSession(
                model_path=req.model_path, title=req.title, system_prompt=req.system_prompt
            )
            session.add(s)
            session.flush()
            return SessionOut.model_validate(s, from_attributes=True)

    return await asyncio.to_thread(_create)


@router.get("")
async def list_sessions() -> list[SessionOut]:
    def _query() -> list[SessionOut]:
        with session_scope() as session:
            stmt = select(ChatSession).order_by(ChatSession.updated_at.desc())
            return [
                SessionOut.model_validate(s, from_attributes=True) for s in session.scalars(stmt)
            ]

    return await asyncio.to_thread(_query)


@router.delete("/{session_id}")
async def delete_session(session_id: str) -> dict:
    def _delete() -> bool:
        with session_scope() as session:
            s = session.get(ChatSession, session_id)
            if s is None:
                return False
            session.delete(s)
            return True

    return {"ok": await asyncio.to_thread(_delete)}


@router.get("/{session_id}/messages")
async def list_messages(session_id: str) -> list[MessageOut]:
    def _query() -> list[MessageOut] | None:
        with session_scope() as session:
            if session.get(ChatSession, session_id) is None:
                return None
            stmt = (
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at)
            )
            return [
                MessageOut.model_validate(m, from_attributes=True) for m in session.scalars(stmt)
            ]

    out = await asyncio.to_thread(_query)
    if out is None:
        _not_found(session_id)
    return out


@router.post("/{session_id}/messages")
async def send_message(session_id: str, req: MessageCreate) -> MessageCreateOut:
    def _prep() -> tuple[str, list[dict], str] | None:
        with session_scope() as session:
            s = session.get(ChatSession, session_id)
            if s is None:
                return None
            user_msg = ChatMessage(session_id=session_id, role="user", content=req.content)
            session.add(user_msg)
            session.flush()

            stmt = (
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at)
            )
            history = [{"role": m.role, "content": m.content} for m in session.scalars(stmt)]
            if s.system_prompt:
                history.insert(0, {"role": "system", "content": s.system_prompt})

            if s.title == "Yeni sohbet":
                s.title = req.content.strip()[:60] or s.title
            return user_msg.id, history, s.model_path

    prepped = await asyncio.to_thread(_prep)
    if prepped is None:
        _not_found(session_id)
    user_msg_id, history, model_path = prepped

    params = {
        "session_id": session_id,
        "model_path": model_path,
        "messages": history,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
    }

    def _create_job() -> str:
        with session_scope() as session:
            job = Job(kind="llm", params=params, status=JobStatus.QUEUED)
            session.add(job)
            session.flush()
            return job.id

    job_id = await asyncio.to_thread(_create_job)
    scheduler.notify()
    return MessageCreateOut(job_id=job_id, user_message_id=user_msg_id)
