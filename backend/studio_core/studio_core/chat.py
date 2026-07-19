"""Sohbet mesaj kalıcılığı (Faz 6): biten 'llm' işini asistan mesajına dönüştürür.

Asset kaydı gibi (studio_core.assets) scheduler'ın on_done kancasından çağrılır —
bu sayede token akışı bittiğinde mesaj otomatik DB'ye yazılır.
"""

from __future__ import annotations

from studio_core.db import session_scope
from studio_core.logging import get_logger
from studio_core.models import ChatMessage, ChatSession, Job

log = get_logger("studio_core.chat")


def save_chat_message_for_job(job_id: str) -> str | None:
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None or job.kind != "llm":
            return None
        session_id = str((job.params or {}).get("session_id", ""))
        text = str((job.result or {}).get("text", ""))
        if not session_id or not text:
            return None
        if session.get(ChatSession, session_id) is None:
            return None
        msg = ChatMessage(session_id=session_id, role="assistant", content=text)
        session.add(msg)
        session.flush()
        msg_id = msg.id
    log.info("chat_message_saved", session_id=session_id, job_id=job_id)
    return msg_id
