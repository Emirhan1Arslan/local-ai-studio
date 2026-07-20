"""Faz 10 WorkflowExecutor testleri: topolojik yürütme, node cache (ikinci
çalıştırmada değişmeyen dallar cache'ten gelir), hata yayılımı (upstream hata
→ downstream skipped), deterministik seed türetme. Sentetik/GPU'suz node
registry kullanır — gerçek job sistemine hiç dokunmaz, hızlı çalışır."""

from __future__ import annotations

import pytest
from studio_workflows.executor import WorkflowExecutor, derive_seed
from studio_workflows.graph import Edge, Graph, GraphValidationError, NodeInstance
from studio_workflows.nodes import NodeTypeDef, ParamSpec, PortSpec


def _node_types(call_log: list[str]) -> dict[str, NodeTypeDef]:
    def const_exec(inputs, params):
        call_log.append("const")
        return {"value": params.get("value", 0)}

    def add_exec(inputs, params):
        call_log.append("add")
        return {"sum": inputs["a"] + inputs["b"]}

    def fail_exec(inputs, params):
        call_log.append("fail")
        raise RuntimeError("kasıtlı test hatası")

    def seed_echo_exec(inputs, params):
        call_log.append("seed_echo")
        return {"seed": params["seed"]}

    types = [
        NodeTypeDef(
            type_id="test.const", category="test", label="Const",
            outputs=(PortSpec("value", "number"),),
            params=(ParamSpec("value", "number", default=0),),
            execute=const_exec,
        ),
        NodeTypeDef(
            type_id="test.add", category="test", label="Add",
            inputs=(PortSpec("a", "number"), PortSpec("b", "number")),
            outputs=(PortSpec("sum", "number"),),
            execute=add_exec,
        ),
        NodeTypeDef(
            type_id="test.fail", category="test", label="Fail",
            inputs=(PortSpec("a", "number"),),
            outputs=(PortSpec("sum", "number"),),
            execute=fail_exec,
        ),
        NodeTypeDef(
            type_id="test.seed_echo", category="test", label="SeedEcho",
            outputs=(PortSpec("seed", "number"),),
            params=(ParamSpec("seed", "number"),),
            execute=seed_echo_exec,
        ),
    ]
    return {t.type_id: t for t in types}


def _linear_graph() -> Graph:
    return Graph(
        nodes=[
            NodeInstance(id="c1", type="test.const", params={"value": 2}),
            NodeInstance(id="c2", type="test.const", params={"value": 3}),
            NodeInstance(id="sum", type="test.add"),
        ],
        edges=[
            Edge(id="e1", from_node="c1", from_port="value", to_node="sum", to_port="a"),
            Edge(id="e2", from_node="c2", from_port="value", to_node="sum", to_port="b"),
        ],
    )


def test_topological_execution_and_result():
    call_log: list[str] = []
    executor = WorkflowExecutor(_node_types(call_log))
    result = executor.run(_linear_graph())
    assert result.ok
    assert result.order.index("c1") < result.order.index("sum")
    assert result.order.index("c2") < result.order.index("sum")
    assert result.results["sum"].outputs["sum"] == 5
    assert call_log.count("add") == 1


def test_second_run_hits_cache_for_unchanged_graph():
    call_log: list[str] = []
    executor = WorkflowExecutor(_node_types(call_log))  # cache, executor ömrü boyunca korunur
    g = _linear_graph()
    executor.run(g)
    calls_before = len(call_log)

    result2 = executor.run(g)

    assert len(call_log) == calls_before  # hiçbir node yeniden çalıştırılmadı
    assert all(r.from_cache for r in result2.results.values())
    assert result2.results["sum"].outputs["sum"] == 5


def test_changed_upstream_param_invalidates_only_dependent_cache():
    call_log: list[str] = []
    executor = WorkflowExecutor(_node_types(call_log))
    executor.run(_linear_graph())
    calls_before = len(call_log)

    g2 = _linear_graph()
    g2.get_node("c1").params["value"] = 99  # yalnızca bir dal değişti
    result = executor.run(g2)

    assert result.results["c1"].from_cache is False  # değişen düğüm yeniden hesaplandı
    assert result.results["c2"].from_cache is True  # değişmeyen dal cache'ten geldi
    assert result.results["sum"].from_cache is False  # girdisi değişti, o da yeniden hesaplandı
    assert result.results["sum"].outputs["sum"] == 102
    assert len(call_log) > calls_before


def test_upstream_failure_skips_downstream():
    call_log: list[str] = []
    node_types = _node_types(call_log)
    g = Graph(
        nodes=[
            NodeInstance(id="c1", type="test.const", params={"value": 1}),
            NodeInstance(id="c2", type="test.const", params={"value": 5}),
            NodeInstance(id="bad", type="test.fail"),
            NodeInstance(id="tail", type="test.add"),
        ],
        edges=[
            Edge(id="e1", from_node="c1", from_port="value", to_node="bad", to_port="a"),
            Edge(id="e2", from_node="bad", from_port="sum", to_node="tail", to_port="a"),
            Edge(id="e3", from_node="c2", from_port="value", to_node="tail", to_port="b"),
        ],
    )
    result = WorkflowExecutor(node_types).run(g)

    assert not result.ok
    assert result.results["bad"].status == "failed"
    assert "kasıtlı test hatası" in result.results["bad"].error
    assert result.results["tail"].status == "skipped"
    assert "add" not in call_log  # tail hiç çalıştırılmadı


def test_run_rejects_invalid_graph_before_executing():
    call_log: list[str] = []
    executor = WorkflowExecutor(_node_types(call_log))
    g = Graph(nodes=[NodeInstance(id="sum", type="test.add")], edges=[])  # zorunlu girişler yok

    with pytest.raises(GraphValidationError):
        executor.run(g)
    assert call_log == []  # doğrulama başarısız olunca hiçbir node çalıştırılmadı


def test_deterministic_seed_derivation_is_stable_and_distinct():
    assert derive_seed(42, "node_a") == derive_seed(42, "node_a")
    assert derive_seed(42, "node_a") != derive_seed(42, "node_b")
    assert derive_seed(42, "node_a") != derive_seed(7, "node_a")


def test_node_gets_derived_seed_when_not_overridden():
    call_log: list[str] = []
    executor = WorkflowExecutor(_node_types(call_log))
    g = Graph(nodes=[NodeInstance(id="s", type="test.seed_echo")], edges=[])

    result = executor.run(g, workflow_seed=42)

    assert result.results["s"].outputs["seed"] == derive_seed(42, "s")


def test_explicit_seed_param_overrides_derivation():
    call_log: list[str] = []
    executor = WorkflowExecutor(_node_types(call_log))
    g = Graph(nodes=[NodeInstance(id="s", type="test.seed_echo", params={"seed": 12345})], edges=[])

    result = executor.run(g, workflow_seed=42)

    assert result.results["s"].outputs["seed"] == 12345
