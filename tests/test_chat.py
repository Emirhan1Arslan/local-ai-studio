"""Faz 6 testleri: sohbet oturumu/mesaj CRUD'ı + iş kablolaması.

Test ortamında llm worker env'i (llama_cpp) yoktur → mesaj gönderimi iş
kuyruklar, iş kurulum talimatı içeren anlaşılır bir hatayla FAILED olmalı.
Gerçek üretim (token streaming) GPU smoke testindedir.
"""

from __future__ import annotations

import sys
import time

from studio_inference.scheduler import worker_python


def _poll_job(client, job_id: str, want: set[str], timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    last: dict = {}
    while time.monotonic() < deadline:
        last = client.get(f"/v1/jobs/{job_id}").json()
        if last["status"] in want:
            return last
        time.sleep(0.1)
    raise AssertionError(f"{want} beklenirken durum: {last.get('status')!r} ({last.get('error')})")


def _create_session(client, **overrides) -> dict:
    body = {"model_path": "yok.gguf", "title": "test", "system_prompt": ""}
    body.update(overrides)
    r = client.post("/v1/chat", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_worker_python_llm_env(client):
    assert worker_python("llm") == sys.executable  # test veri dizininde env yok → düşer


def test_create_and_list_session(client):
    s = _create_session(client, title="Sohbet 1")
    assert s["title"] == "Sohbet 1"
    assert s["model_path"] == "yok.gguf"

    rows = client.get("/v1/chat").json()
    assert any(r["id"] == s["id"] for r in rows)


def test_delete_session(client):
    s = _create_session(client)
    assert client.delete(f"/v1/chat/{s['id']}").json()["ok"] is True
    assert client.delete(f"/v1/chat/{s['id']}").json()["ok"] is False  # ikinci sefer yok


def test_messages_for_unknown_session_404(client):
    assert client.get("/v1/chat/yok/messages").status_code == 404


def test_send_message_unknown_session_404(client):
    r = client.post("/v1/chat/yok/messages", json={"content": "merhaba"})
    assert r.status_code == 404


def test_send_message_creates_user_message_and_job(client):
    s = _create_session(client, system_prompt="Sen yardımsever bir asistansın.")
    r = client.post(f"/v1/chat/{s['id']}/messages", json={"content": "merhaba"})
    assert r.status_code == 200
    body = r.json()
    assert body["job_id"]
    assert body["user_message_id"]

    msgs = client.get(f"/v1/chat/{s['id']}/messages").json()
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "merhaba"

    job = client.get(f"/v1/jobs/{body['job_id']}").json()
    assert job["kind"] == "llm"
    assert job["params"]["session_id"] == s["id"]
    assert job["params"]["messages"][0] == {
        "role": "system",
        "content": "Sen yardımsever bir asistansın.",
    }
    assert job["params"]["messages"][1] == {"role": "user", "content": "merhaba"}


def test_send_message_sets_title_from_first_message(client):
    s = _create_session(client, title="Yeni sohbet")
    content = "Python'da liste nasıl ters çevrilir?"
    client.post(f"/v1/chat/{s['id']}/messages", json={"content": content})
    rows = client.get("/v1/chat").json()
    updated = next(r for r in rows if r["id"] == s["id"])
    assert updated["title"] == content


def test_send_message_job_fails_gracefully_without_env(client):
    s = _create_session(client)
    r = client.post(f"/v1/chat/{s['id']}/messages", json={"content": "merhaba"})
    job = _poll_job(client, r.json()["job_id"], {"failed"})
    assert "worker ortamı eksik" in job["error"] or "llama_cpp" in job["error"]


def test_save_chat_message_for_job_persists_assistant_reply(client):
    """studio_core.chat.save_chat_message_for_job — scheduler'ın on_done kancası
    gerçek bir 'llm' işi bittiğinde bunu çağırır; burada doğrudan test edilir."""
    from studio_core.chat import save_chat_message_for_job
    from studio_core.db import session_scope
    from studio_core.models import Job, JobStatus

    s = _create_session(client)

    def _make_done_job() -> str:
        with session_scope() as session:
            job = Job(
                kind="llm",
                status=JobStatus.DONE,
                params={"session_id": s["id"]},
                result={"text": "Merhaba! Nasıl yardımcı olabilirim?", "tokens": 8},
            )
            session.add(job)
            session.flush()
            return job.id

    job_id = _make_done_job()
    msg_id = save_chat_message_for_job(job_id)
    assert msg_id is not None

    msgs = client.get(f"/v1/chat/{s['id']}/messages").json()
    assert len(msgs) == 1
    assert msgs[0]["role"] == "assistant"
    assert msgs[0]["content"] == "Merhaba! Nasıl yardımcı olabilirim?"


def test_save_chat_message_ignores_non_llm_jobs(client):
    from studio_core.chat import save_chat_message_for_job
    from studio_core.db import session_scope
    from studio_core.models import Job, JobStatus

    def _make_image_job() -> str:
        with session_scope() as session:
            job = Job(kind="image", status=JobStatus.DONE, params={}, result={"path": "x.png"})
            session.add(job)
            session.flush()
            return job.id

    assert save_chat_message_for_job(_make_image_job()) is None
