"""Faz 10: WorkflowExecutor'ın gerçek Job sistemine (Scheduler+DB) blocking
dispatch köprüsü. Dummy adapter kullanır — GPU gerekmez, ama gerçek worker
SÜRECİ üzerinden uçtan uca job yaşam döngüsünü (queued→running→done/failed)
test eder, `client` fixture'ı ile aynı scheduler'ı paylaşır."""

from __future__ import annotations

import pytest
from studio_workflows.dispatch import JobDispatchError, make_job_dispatcher
from studio_workflows.nodes import JobDispatch


def test_dispatch_waits_for_done_and_returns_result(client):
    dispatch = make_job_dispatcher(poll_interval=0.05, timeout=20.0)
    strategy = JobDispatch(kind="dummy", build_params=lambda i, p: {})

    result = dispatch(strategy, {"steps": 2, "step_ms": 10})

    assert result["steps"] == 2


def test_dispatch_raises_on_failed_job(client):
    dispatch = make_job_dispatcher(poll_interval=0.05, timeout=20.0)
    strategy = JobDispatch(kind="dummy", build_params=lambda i, p: {})

    with pytest.raises(JobDispatchError, match="kasıtlı test hatası"):
        dispatch(strategy, {"fail": True})


def test_executor_runs_job_dispatch_node_against_real_scheduler(client):
    from studio_workflows.executor import WorkflowExecutor
    from studio_workflows.graph import Graph
    from studio_workflows.graph import NodeInstance as N
    from studio_workflows.nodes import NodeTypeDef, PortSpec

    node_types = {
        "test.dummy_job": NodeTypeDef(
            type_id="test.dummy_job",
            category="test",
            label="DummyJob",
            outputs=(PortSpec("steps", "number"),),
            execute=JobDispatch(
                kind="dummy",
                build_params=lambda i, p: {"steps": 3, "step_ms": 10},
                map_output=lambda r: {"steps": r["steps"]},
            ),
        ),
    }
    executor = WorkflowExecutor(node_types, job_dispatcher=make_job_dispatcher(poll_interval=0.05))
    g = Graph(nodes=[N(id="j", type="test.dummy_job")], edges=[])

    result = executor.run(g)

    assert result.ok
    assert result.results["j"].outputs["steps"] == 3
