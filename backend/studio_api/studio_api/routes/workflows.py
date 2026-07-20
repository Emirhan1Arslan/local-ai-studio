"""Workflow rotaları (Faz 10): node-graph kaydet/yükle/çalıştır.

Kaydetme VE çalıştırma anında grafik doğrulanır (DoD: "bozuk workflow
yüklenince anlaşılır doğrulama hatası verir") — bilinmeyen düğüm türü/port,
tip uyumsuzluğu, bağlı olmayan zorunlu giriş, döngü ⇒ 400 + anlaşılır mesaj.

Çalıştırma arka planda (`asyncio.to_thread`) yürütülür; her workflow_id için
process-içi bir node-çıktı önbelleği (`_run_caches`) tutulur ki ikinci
çalıştırmada değişmeyen dallar cache'ten gelsin (DoD) — süreç yeniden
başlayınca önbellek sıfırlanır, bu MVP için kabul edilebilir bir sınırdır.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from fastapi import APIRouter
from pydantic import BaseModel, Field
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import select
from studio_core.db import session_scope
from studio_core.models import Workflow, WorkflowRun, WorkflowRunStatus
from studio_workflows import NODE_TYPES, Graph, GraphValidationError, WorkflowExecutor
from studio_workflows.dispatch import make_job_dispatcher

from studio_api.bus import bus

router = APIRouter(prefix="/v1/workflows", tags=["workflows"])

# workflow_id -> content-hash cache (bkz. WorkflowExecutor). Süreç ömrü boyunca
# yaşar; birden çok run() çağrısı arasında paylaşılır (Faz 10 DoD: cache).
_run_caches: dict[str, dict[str, dict]] = {}

# Arka planda çalışan run task'larına referans tutulur — Python, referanssız
# asyncio.Task'ları rastgele bir noktada GC edip sessizce iptal edebilir.
_background_tasks: set[asyncio.Task] = set()


class WorkflowRequest(BaseModel):
    name: str = "Adsız workflow"
    project_id: str | None = None
    graph: dict = Field(default_factory=dict)


class WorkflowOut(BaseModel):
    id: str
    name: str
    project_id: str | None
    graph: dict
    created_at: datetime
    updated_at: datetime


class RunRequest(BaseModel):
    seed: int = 0


class WorkflowRunOut(BaseModel):
    id: str
    workflow_id: str
    status: str
    seed: int
    node_results: dict
    error: str
    created_at: datetime
    finished_at: datetime | None


def _validate_graph(graph_dict: dict) -> Graph:
    from studio_api.app import ApiError

    try:
        graph = Graph.model_validate(graph_dict)
        graph.validate_structure(NODE_TYPES)
    except GraphValidationError as exc:
        raise ApiError(400, "invalid_workflow", str(exc)) from exc
    except PydanticValidationError as exc:
        raise ApiError(400, "invalid_workflow", f"workflow şeması geçersiz: {exc}") from exc
    return graph


@router.post("")
async def create_workflow(req: WorkflowRequest) -> WorkflowOut:
    _validate_graph(req.graph)

    def _create() -> WorkflowOut:
        with session_scope() as session:
            wf = Workflow(name=req.name, project_id=req.project_id, graph=req.graph)
            session.add(wf)
            session.flush()
            return WorkflowOut.model_validate(wf, from_attributes=True)

    return await asyncio.to_thread(_create)


@router.get("")
async def list_workflows(project_id: str | None = None) -> list[WorkflowOut]:
    def _query() -> list[WorkflowOut]:
        with session_scope() as session:
            stmt = select(Workflow).order_by(Workflow.updated_at.desc())
            if project_id:
                stmt = stmt.where(Workflow.project_id == project_id)
            return [
                WorkflowOut.model_validate(w, from_attributes=True) for w in session.scalars(stmt)
            ]

    return await asyncio.to_thread(_query)


def _get_workflow_row(session, workflow_id: str) -> Workflow:
    from studio_api.app import ApiError

    wf = session.get(Workflow, workflow_id)
    if wf is None:
        raise ApiError(404, "not_found", f"workflow bulunamadı: {workflow_id}")
    return wf


@router.get("/{workflow_id}")
async def get_workflow(workflow_id: str) -> WorkflowOut:
    def _query() -> WorkflowOut:
        with session_scope() as session:
            wf = _get_workflow_row(session, workflow_id)
            return WorkflowOut.model_validate(wf, from_attributes=True)

    return await asyncio.to_thread(_query)


@router.put("/{workflow_id}")
async def update_workflow(workflow_id: str, req: WorkflowRequest) -> WorkflowOut:
    _validate_graph(req.graph)

    def _update() -> WorkflowOut:
        with session_scope() as session:
            wf = _get_workflow_row(session, workflow_id)
            wf.name = req.name
            wf.project_id = req.project_id
            wf.graph = req.graph
            session.flush()
            return WorkflowOut.model_validate(wf, from_attributes=True)

    return await asyncio.to_thread(_update)


@router.delete("/{workflow_id}")
async def delete_workflow(workflow_id: str) -> dict:
    def _delete() -> dict:
        with session_scope() as session:
            wf = _get_workflow_row(session, workflow_id)
            session.delete(wf)
        _run_caches.pop(workflow_id, None)
        return {"ok": True}

    return await asyncio.to_thread(_delete)


def _finish_run(
    run_id: str, *, status: WorkflowRunStatus, error: str = "", node_results: dict | None = None
) -> None:
    with session_scope() as session:
        run = session.get(WorkflowRun, run_id)
        if run is None:
            return
        run.status = status
        run.error = error
        if node_results is not None:
            run.node_results = node_results
        run.finished_at = datetime.now(UTC)


def _execute_workflow_run(
    run_id: str, workflow_id: str, graph: Graph, seed: int, project_id: str | None
) -> None:
    cache = _run_caches.setdefault(workflow_id, {})

    def on_event(topic: str, data: dict) -> None:
        bus.emit(topic, run_id=run_id, workflow_id=workflow_id, **data)

    bus.emit("workflow.started", run_id=run_id, workflow_id=workflow_id)
    executor = WorkflowExecutor(
        NODE_TYPES,
        cache=cache,
        job_dispatcher=make_job_dispatcher(project_id=project_id),
        on_event=on_event,
    )
    try:
        run_result = executor.run(graph, workflow_seed=seed)
    except Exception as exc:  # noqa: BLE001 - beklenmeyen yürütme hatasını run kaydına yazıyoruz
        _finish_run(run_id, status=WorkflowRunStatus.FAILED, error=str(exc))
        bus.emit("workflow.failed", run_id=run_id, workflow_id=workflow_id, error=str(exc))
        return

    node_results = {
        nid: {
            "status": r.status,
            "outputs": r.outputs,
            "error": r.error,
            "from_cache": r.from_cache,
        }
        for nid, r in run_result.results.items()
    }
    failed = [f"{nid}: {r.error}" for nid, r in run_result.results.items() if r.status == "failed"]
    status = WorkflowRunStatus.DONE if run_result.ok else WorkflowRunStatus.FAILED
    _finish_run(run_id, status=status, error="; ".join(failed), node_results=node_results)
    bus.emit(
        "workflow.done" if run_result.ok else "workflow.failed",
        run_id=run_id,
        workflow_id=workflow_id,
    )


def _spawn_background(coro) -> None:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


@router.post("/{workflow_id}/run")
async def run_workflow(workflow_id: str, req: RunRequest) -> WorkflowRunOut:
    def _load() -> Workflow:
        with session_scope() as session:
            wf = _get_workflow_row(session, workflow_id)
            session.expunge(wf)
            return wf

    wf = await asyncio.to_thread(_load)
    graph = _validate_graph(wf.graph)  # kaydedilmiş grafiği çalıştırmadan önce tekrar doğrula

    def _create_run() -> WorkflowRunOut:
        with session_scope() as session:
            run = WorkflowRun(
                workflow_id=workflow_id, seed=req.seed, status=WorkflowRunStatus.RUNNING
            )
            session.add(run)
            session.flush()
            return WorkflowRunOut.model_validate(run, from_attributes=True)

    run_out = await asyncio.to_thread(_create_run)
    _spawn_background(
        asyncio.to_thread(
            _execute_workflow_run, run_out.id, workflow_id, graph, req.seed, wf.project_id
        )
    )
    return run_out


@router.get("/{workflow_id}/runs")
async def list_workflow_runs(workflow_id: str) -> list[WorkflowRunOut]:
    def _query() -> list[WorkflowRunOut]:
        with session_scope() as session:
            stmt = (
                select(WorkflowRun)
                .where(WorkflowRun.workflow_id == workflow_id)
                .order_by(WorkflowRun.created_at.desc())
            )
            return [
                WorkflowRunOut.model_validate(r, from_attributes=True)
                for r in session.scalars(stmt)
            ]

    return await asyncio.to_thread(_query)


@router.get("/runs/{run_id}")
async def get_workflow_run(run_id: str) -> WorkflowRunOut:
    from studio_api.app import ApiError

    def _query() -> WorkflowRunOut:
        with session_scope() as session:
            run = session.get(WorkflowRun, run_id)
            if run is None:
                raise ApiError(404, "not_found", f"workflow run bulunamadı: {run_id}")
            return WorkflowRunOut.model_validate(run, from_attributes=True)

    return await asyncio.to_thread(_query)
