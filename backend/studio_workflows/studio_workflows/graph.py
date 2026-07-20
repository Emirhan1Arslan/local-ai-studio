"""Graph çekirdeği (Faz 10 adım 1): tipli port sistemi + node/edge veri modeli + DAG doğrulama.

Bir `Graph`, `NodeInstance` (düğüm) ve `Edge` (bağlantı) listesinden oluşur. Her
düğümün TÜRÜ (`NodeInstance.type`) `nodes.py`'deki `NODE_TYPES` kayıt defterinde
tanımlı bir `NodeTypeDef`'e karşılık gelir — port isimleri/tipleri ve parametre
şeması oradan gelir, `Graph` yalnızca YAPIYI (hangi düğüm hangi düğüme, hangi
portlardan bağlı) ve bunun geçerliliğini (döngüsüz, port tipleri uyumlu, portlar
gerçekten var) doğrular.

Port tipleri plan.md'nin Faz 10 tanımından: image, latent, text, audio, video,
mesh, tensor, any — Kontrol akışı düğümleri (Condition/Loop) için gereken
number/boolean eklendi (mantıksal bir gereklilik, plan.md'nin ruhuna aykırı
değil — "Condition" bir koşulsuz `any` ile karşılaştırma yapamaz).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

PortType = Literal[
    "image", "latent", "text", "audio", "video", "mesh", "tensor", "any", "number", "boolean"
]


class GraphValidationError(Exception):
    """Bozuk workflow yüklenince anlaşılır doğrulama hatası (Faz 10 DoD maddesi).

    `node_id`/`edge_id` doluysa hata UI'da doğrudan ilgili düğüme/bağlantıya
    işaretlenebilir — genel bir "workflow geçersiz" mesajından daha kullanışlı.
    """

    def __init__(self, message: str, *, node_id: str | None = None, edge_id: str | None = None):
        super().__init__(message)
        self.message = message
        self.node_id = node_id
        self.edge_id = edge_id


class NodeInstance(BaseModel):
    id: str
    type: str
    params: dict = Field(default_factory=dict)
    # UI-only alanlar (canvas konumu vb.) — executor bunları hiç okumaz, yalnızca
    # düz geçirilir (round-trip: kaydet → yükle → aynı yerleşim).
    ui: dict = Field(default_factory=dict)


class Edge(BaseModel):
    id: str
    from_node: str
    from_port: str
    to_node: str
    to_port: str


def port_types_compatible(a: PortType, b: PortType) -> bool:
    return a == b or a == "any" or b == "any"


class Graph(BaseModel):
    nodes: list[NodeInstance] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)

    @field_validator("nodes")
    @classmethod
    def _unique_node_ids(cls, nodes: list[NodeInstance]) -> list[NodeInstance]:
        seen = set()
        for n in nodes:
            if n.id in seen:
                raise GraphValidationError(f"yinelenen düğüm id'si: {n.id!r}", node_id=n.id)
            seen.add(n.id)
        return nodes

    def get_node(self, node_id: str) -> NodeInstance | None:
        return next((n for n in self.nodes if n.id == node_id), None)

    def incoming_edges(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.to_node == node_id]

    def outgoing_edges(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.from_node == node_id]

    def validate_structure(self, node_types: dict[str, NodeTypeDef]) -> None:  # noqa: F821
        """Tam doğrulama: bilinmeyen düğüm türü/port adı, tip uyumsuzluğu, döngü.

        `node_types` çağıran taraftan geçiriliyor (nodes.py'ye döngüsel import
        olmasın diye) — `NODE_TYPES` kayıt defteri normalde geçirilir.
        """
        node_ids = {n.id for n in self.nodes}
        for node in self.nodes:
            if node.type not in node_types:
                raise GraphValidationError(
                    f"bilinmeyen düğüm türü: {node.type!r} (düğüm {node.id!r})", node_id=node.id
                )

        connected_inputs: dict[str, set[str]] = {n.id: set() for n in self.nodes}
        for edge in self.edges:
            if edge.to_node in connected_inputs:
                connected_inputs[edge.to_node].add(edge.to_port)

        for node in self.nodes:
            type_def = node_types.get(node.type)
            if type_def is None:
                continue
            for port in type_def.inputs:
                if port.required and port.name not in connected_inputs[node.id]:
                    raise GraphValidationError(
                        f"düğüm {node.id!r} ({node.type!r}): zorunlu giriş portu "
                        f"{port.name!r} bağlı değil",
                        node_id=node.id,
                    )

        for edge in self.edges:
            if edge.from_node not in node_ids:
                raise GraphValidationError(
                    f"bağlantı {edge.id!r}: kaynak düğüm yok: {edge.from_node!r}", edge_id=edge.id
                )
            if edge.to_node not in node_ids:
                raise GraphValidationError(
                    f"bağlantı {edge.id!r}: hedef düğüm yok: {edge.to_node!r}", edge_id=edge.id
                )
            from_node = self.get_node(edge.from_node)
            to_node = self.get_node(edge.to_node)
            from_type_def = node_types.get(from_node.type) if from_node else None
            to_type_def = node_types.get(to_node.type) if to_node else None
            if from_type_def is None or to_type_def is None:
                continue  # yukarıda zaten bilinmeyen-tür hatası verildi
            from_port = next((p for p in from_type_def.outputs if p.name == edge.from_port), None)
            to_port = next((p for p in to_type_def.inputs if p.name == edge.to_port), None)
            if from_port is None:
                raise GraphValidationError(
                    f"bağlantı {edge.id!r}: {from_node.type!r} düğümünde "
                    f"{edge.from_port!r} adlı çıkış portu yok",
                    edge_id=edge.id,
                )
            if to_port is None:
                raise GraphValidationError(
                    f"bağlantı {edge.id!r}: {to_node.type!r} düğümünde "
                    f"{edge.to_port!r} adlı giriş portu yok",
                    edge_id=edge.id,
                )
            if not port_types_compatible(from_port.type, to_port.type):
                raise GraphValidationError(
                    f"bağlantı {edge.id!r}: tip uyumsuz ({from_port.type!r} → {to_port.type!r})",
                    edge_id=edge.id,
                )

        self.topological_order()  # döngü varsa burada GraphValidationError fırlatır

    def topological_order(self) -> list[str]:
        """Kahn algoritması — döngü varsa GraphValidationError fırlatır (hangi
        düğümlerin döngüde kaldığı hata mesajında listelenir, teşhisi kolaylaştırır)."""
        in_degree: dict[str, int] = {n.id: 0 for n in self.nodes}
        adjacency: dict[str, list[str]] = {n.id: [] for n in self.nodes}
        for edge in self.edges:
            if edge.from_node not in in_degree or edge.to_node not in in_degree:
                continue
            adjacency[edge.from_node].append(edge.to_node)
            in_degree[edge.to_node] += 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order: list[str] = []
        remaining = dict(in_degree)
        while queue:
            # Kararlı sıralama: girdi listesindeki sırayı koru (deterministik yürütme).
            queue.sort(key=lambda nid: [n.id for n in self.nodes].index(nid))
            nid = queue.pop(0)
            order.append(nid)
            for neighbor in adjacency[nid]:
                remaining[neighbor] -= 1
                if remaining[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self.nodes):
            stuck = sorted(nid for nid, deg in remaining.items() if deg > 0)
            raise GraphValidationError(f"graph döngü içeriyor, düğümler: {stuck}")
        return order
