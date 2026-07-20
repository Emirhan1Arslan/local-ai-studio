"""Node türü kayıt defteri (Faz 10 adım 2): her node türünün portları, parametre
şeması ve yürütme stratejisini tanımlar.

İki yürütme stratejisi var:
- `JobDispatch`: mevcut Scheduler/Job altyapısına dispatch edilir (GPU-ağır
  işler: görsel üretim, upscale, bg-remove, mesh...). Var olan `/v1/jobs`
  sistemini yeniden icat etmek yerine ona devrediyoruz — VRAM/iptal/kuyruk
  mantığı bedava geliyor.
- Doğrudan bir Python callable (`InlineFn` imzasında): process içinde çalışan
  hafif/kontrol-akışı düğümleri (Prompt metin şablonu, Save, Condition...) için.

Gerçek düğüm türleri `node_library.py`'de `register_node_type()` ile bu
`NODE_TYPES` defterine kaydedilir. Bu modül yalnızca ŞEMAyı tanımlar — testler
kendi sahte `NodeTypeDef`'lerini ayrı, izole bir dict'te kurar (bkz.
`tests/test_executor.py`), gerçek defteri kirletmez.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from .graph import PortType


@dataclass(frozen=True)
class PortSpec:
    name: str
    type: PortType
    required: bool = True


@dataclass(frozen=True)
class ParamSpec:
    name: str
    type: str  # "string" | "number" | "boolean" | "enum" | "json"
    default: Any = None
    choices: tuple[str, ...] | None = None


@dataclass(frozen=True)
class JobDispatch:
    """Mevcut Job sistemine dispatch: `kind` iş türü, `build_params` node
    girdilerinden/parametrelerinden job `params` sözlüğüne eşleme yapar."""

    kind: str
    build_params: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


class InlineFn(Protocol):
    def __call__(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class NodeTypeDef:
    type_id: str
    category: str
    label: str
    inputs: tuple[PortSpec, ...] = ()
    outputs: tuple[PortSpec, ...] = ()
    params: tuple[ParamSpec, ...] = ()
    execute: JobDispatch | InlineFn | None = None


NODE_TYPES: dict[str, NodeTypeDef] = {}


def register_node_type(node_type: NodeTypeDef) -> NodeTypeDef:
    if node_type.type_id in NODE_TYPES:
        raise ValueError(f"node türü zaten kayıtlı: {node_type.type_id!r}")
    NODE_TYPES[node_type.type_id] = node_type
    return node_type
