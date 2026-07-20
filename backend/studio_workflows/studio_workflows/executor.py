"""WorkflowExecutor (Faz 10 adım 3): DAG'ı topolojik sırada yürütür.

- Cache: her düğümün çıktısı `(tür, çözümlenmiş parametreler, girdi
  değerleri)` üçlüsünün hash'iyle anahtarlanır — aynı `WorkflowExecutor`
  (dolayısıyla aynı `cache` sözlüğü) ile ikinci `run()` çağrısında değişmeyen
  dallar cache'ten gelir, değişen bir düğümün altındaki her şey otomatik
  olarak yeniden hesaplanır (çünkü onların girdi hash'i de değişir).
- Hata yayılımı: bir düğüm hata verirse, ona (doğrudan/dolaylı) bağımlı TÜM
  alt düğümler çalıştırılmadan "skipped" olarak işaretlenir.
- Deterministik seed: workflow-seviyesi seed + node id -> düğüme özel seed,
  node kendi parametrelerinde açıkça bir seed vermedikçe.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .graph import Graph, NodeInstance
from .nodes import JobDispatch, NodeTypeDef


class NodeExecutionError(Exception):
    def __init__(self, node_id: str, message: str):
        super().__init__(f"düğüm {node_id!r}: {message}")
        self.node_id = node_id
        self.message = message


@dataclass
class NodeResult:
    node_id: str
    status: str  # "done" | "failed" | "skipped"
    outputs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    from_cache: bool = False
    cache_key: str | None = None


@dataclass
class WorkflowRunResult:
    order: list[str]
    results: dict[str, NodeResult]

    @property
    def ok(self) -> bool:
        return all(r.status != "failed" for r in self.results.values())


def derive_seed(workflow_seed: int, node_id: str) -> int:
    digest = hashlib.sha256(f"{workflow_seed}:{node_id}".encode()).hexdigest()
    return int(digest[:8], 16)


def _stable(value: Any) -> Any:
    """Cache anahtarı için değerleri kararlı/sıralı hale getirir. Gerçek
    node'lar çıktı olarak büyük opak nesneler (tensor vb.) değil, dosya yolu
    veya job id gibi hafif tanımlayıcılar döndürmeli — bu fonksiyon onların
    JSON'a serileşebilir olduğunu varsayar."""
    if isinstance(value, dict):
        return {k: _stable(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_stable(v) for v in value]
    return value


def _content_hash(
    node: NodeInstance, resolved_params: dict[str, Any], input_values: dict[str, Any]
) -> str:
    payload = {
        "type": node.type,
        "params": _stable(resolved_params),
        "inputs": _stable(input_values),
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


class WorkflowExecutor:
    def __init__(
        self,
        node_types: dict[str, NodeTypeDef],
        *,
        cache: dict[str, dict[str, Any]] | None = None,
        job_dispatcher: Callable[[JobDispatch, dict[str, Any]], dict[str, Any]] | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ):
        self.node_types = node_types
        self.cache = cache if cache is not None else {}
        self.job_dispatcher = job_dispatcher
        self.on_event = on_event or (lambda topic, data: None)

    def run(self, graph: Graph, *, workflow_seed: int = 0) -> WorkflowRunResult:
        graph.validate_structure(self.node_types)
        order = graph.topological_order()

        results: dict[str, NodeResult] = {}
        blocked: set[str] = set()

        for node_id in order:
            node = graph.get_node(node_id)
            node_type = self.node_types[node.type]

            if any(e.from_node in blocked for e in graph.incoming_edges(node_id)):
                results[node_id] = NodeResult(node_id=node_id, status="skipped")
                blocked.add(node_id)
                self.on_event("workflow.node_skipped", {"node_id": node_id})
                continue

            input_values: dict[str, Any] = {}
            for edge in graph.incoming_edges(node_id):
                input_values[edge.to_port] = results[edge.from_node].outputs[edge.from_port]

            resolved_params = dict(node.params)
            if "seed" not in resolved_params and any(p.name == "seed" for p in node_type.params):
                resolved_params["seed"] = derive_seed(workflow_seed, node_id)

            cache_key = _content_hash(node, resolved_params, input_values)
            self.on_event("workflow.node_started", {"node_id": node_id, "cache_key": cache_key})

            if cache_key in self.cache:
                results[node_id] = NodeResult(
                    node_id=node_id, status="done", outputs=self.cache[cache_key],
                    from_cache=True, cache_key=cache_key,
                )
                self.on_event("workflow.node_done", {"node_id": node_id, "from_cache": True})
                continue

            try:
                outputs = self._execute_node(node_type, input_values, resolved_params)
            except Exception as exc:  # noqa: BLE001 - node hatasını sonuca çeviriyoruz, yürütmeyi durdurmuyoruz
                results[node_id] = NodeResult(
                    node_id=node_id, status="failed", error=str(exc), cache_key=cache_key
                )
                blocked.add(node_id)
                self.on_event("workflow.node_failed", {"node_id": node_id, "error": str(exc)})
                continue

            self.cache[cache_key] = outputs
            results[node_id] = NodeResult(
                node_id=node_id, status="done", outputs=outputs, cache_key=cache_key
            )
            self.on_event("workflow.node_done", {"node_id": node_id, "from_cache": False})

        return WorkflowRunResult(order=order, results=results)

    def _execute_node(
        self, node_type: NodeTypeDef, inputs: dict[str, Any], params: dict[str, Any]
    ) -> dict[str, Any]:
        strategy = node_type.execute
        if isinstance(strategy, JobDispatch):
            if self.job_dispatcher is None:
                raise NodeExecutionError(
                    node_type.type_id,
                    "job_dispatcher tanımlı değil, JobDispatch node çalıştırılamaz",
                )
            job_params = strategy.build_params(inputs, params)
            job_result = self.job_dispatcher(strategy, job_params)
            return strategy.map_output(job_result) if strategy.map_output else job_result
        if callable(strategy):
            return strategy(inputs, params)
        raise NodeExecutionError(node_type.type_id, "yürütme stratejisi tanımlı değil")
