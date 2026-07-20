"""Faz 10 graph çekirdeği testleri: port tip uyumu, döngü tespiti, anlaşılır
doğrulama hataları (DoD: "bozuk workflow yüklenince anlaşılır doğrulama hatası
verir"). Sentetik node türleri kullanır, gerçek node kütüphanesine dokunmaz."""

from __future__ import annotations

import pytest
from studio_workflows.graph import Edge, Graph, GraphValidationError, NodeInstance
from studio_workflows.nodes import NodeTypeDef, PortSpec

NUMBER_SOURCE = NodeTypeDef(
    type_id="test.number_source", category="test", label="Number",
    outputs=(PortSpec("value", "number"),),
)
NUMBER_SINK = NodeTypeDef(
    type_id="test.number_sink", category="test", label="Sink",
    inputs=(PortSpec("value", "number"),),
)
TEXT_SINK = NodeTypeDef(
    type_id="test.text_sink", category="test", label="TextSink",
    inputs=(PortSpec("value", "text"),),
)
ANY_SINK = NodeTypeDef(
    type_id="test.any_sink", category="test", label="AnySink",
    inputs=(PortSpec("value", "any"),),
)
OPTIONAL_SINK = NodeTypeDef(
    type_id="test.optional_sink", category="test", label="OptSink",
    inputs=(PortSpec("value", "number", required=False),),
)

_ALL_TEST_TYPES = [NUMBER_SOURCE, NUMBER_SINK, TEXT_SINK, ANY_SINK, OPTIONAL_SINK]
NODE_TYPES = {t.type_id: t for t in _ALL_TEST_TYPES}


def _graph(nodes: list[dict], edges: list[dict]) -> Graph:
    return Graph(nodes=[NodeInstance(**n) for n in nodes], edges=[Edge(**e) for e in edges])


def test_valid_graph_passes():
    g = _graph(
        [{"id": "a", "type": "test.number_source"}, {"id": "b", "type": "test.number_sink"}],
        [{"id": "e1", "from_node": "a", "from_port": "value", "to_node": "b", "to_port": "value"}],
    )
    g.validate_structure(NODE_TYPES)


def test_unknown_node_type_rejected():
    g = _graph([{"id": "a", "type": "test.yok"}], [])
    with pytest.raises(GraphValidationError, match="bilinmeyen düğüm türü"):
        g.validate_structure(NODE_TYPES)


def test_dangling_edge_endpoint_rejected():
    g = _graph(
        [{"id": "a", "type": "test.number_source"}],
        [
            {
                "id": "e1", "from_node": "a", "from_port": "value",
                "to_node": "yok", "to_port": "value",
            }
        ],
    )
    with pytest.raises(GraphValidationError, match="hedef düğüm yok"):
        g.validate_structure(NODE_TYPES)


def test_unknown_port_rejected():
    g = _graph(
        [{"id": "a", "type": "test.number_source"}, {"id": "b", "type": "test.number_sink"}],
        [{"id": "e1", "from_node": "a", "from_port": "yok", "to_node": "b", "to_port": "value"}],
    )
    with pytest.raises(GraphValidationError, match="çıkış portu yok"):
        g.validate_structure(NODE_TYPES)


def test_incompatible_port_types_rejected():
    g = _graph(
        [{"id": "a", "type": "test.number_source"}, {"id": "b", "type": "test.text_sink"}],
        [{"id": "e1", "from_node": "a", "from_port": "value", "to_node": "b", "to_port": "value"}],
    )
    with pytest.raises(GraphValidationError, match="tip uyumsuz"):
        g.validate_structure(NODE_TYPES)


def test_any_port_accepts_anything():
    g = _graph(
        [{"id": "a", "type": "test.number_source"}, {"id": "b", "type": "test.any_sink"}],
        [{"id": "e1", "from_node": "a", "from_port": "value", "to_node": "b", "to_port": "value"}],
    )
    g.validate_structure(NODE_TYPES)


def test_missing_required_input_rejected():
    g = _graph([{"id": "b", "type": "test.number_sink"}], [])
    with pytest.raises(GraphValidationError, match="zorunlu giriş portu"):
        g.validate_structure(NODE_TYPES)


def test_optional_input_may_be_unconnected():
    g = _graph([{"id": "b", "type": "test.optional_sink"}], [])
    g.validate_structure(NODE_TYPES)


def test_cycle_rejected():
    g = _graph(
        [{"id": "a", "type": "test.number_sink"}, {"id": "b", "type": "test.number_sink"}],
        [
            {
                "id": "e1", "from_node": "a", "from_port": "value",
                "to_node": "b", "to_port": "value",
            },
            {
                "id": "e2", "from_node": "b", "from_port": "value",
                "to_node": "a", "to_port": "value",
            },
        ],
    )
    with pytest.raises(GraphValidationError, match="döngü"):
        g.topological_order()


def test_duplicate_node_id_rejected():
    dupe_nodes = [
        {"id": "a", "type": "test.number_source"},
        {"id": "a", "type": "test.number_source"},
    ]
    with pytest.raises(GraphValidationError, match="yinelenen"):
        _graph(dupe_nodes, [])
