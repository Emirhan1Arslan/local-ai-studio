"""Mesh düzenleme adapter'ı (Faz 9 adım 2-3): trimesh tabanlı decimate/onarım/
format dönüştürme + inceleme. imgtool.py/videotool.py ile aynı op-dispatch
deseni.

DİKKAT: Bu modül worker env'inde (envs/3d ile paylaşılır — scheduler._ENV_ALIAS)
çalışır. Model gerektirmez — saf trimesh/CPU işlemleri.

params:
- op: convert | decimate | repair | info (zorunlu)

convert:
- mesh_path (zorunlu) · output_path (zorunlu — uzantı hedef formatı belirler:
  .glb/.gltf/.obj/.stl/.ply doğrudan trimesh ile; .fbx Blender köprüsüyle
  (kurulu olması gerekir — bkz. _blender_bridge.py, docs/worker-envs.md).
  decimate/repair op'larının output_path'i de .fbx olabilir, aynı köprüyü
  kullanır.

decimate:
- mesh_path · output_path (ikisi de zorunlu)
- target_ratio: 0-1 (vars. 0.5, hedef yüz sayısı = mevcut × oran) — target_faces
  verilirse target_ratio yok sayılır
- target_faces: kesin hedef yüz sayısı (opsiyonel)
- watertight kaybolursa best-effort onarılır ama GARANTİ değil — sonuçtaki
  `watertight` alanı gerçek durumu gösterir (bkz. docs/worker-envs.md)

repair:
- mesh_path · output_path (ikisi de zorunlu) — normal yönü düzeltme
  (fix_normals/fix_winding) + delik doldurma (fill_holes) uygular

info:
- mesh_path (zorunlu) — dosya yazmaz; vertices/faces/watertight/bounds/
  volume/area döner (STL bir slicer'da açılmadan önce sağlık kontrolü)

Tüm op'lar (info hariç): output_path (zorunlu)
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any


def _import_error(hint: str, exc: ImportError) -> RuntimeError:
    return RuntimeError(
        f"meshtool worker ortamı eksik ({hint}). "
        "Kurulum reçetesi: docs/worker-envs.md  |  " + str(exc)
    )


def _load_mesh(mesh_path: str):
    try:
        import trimesh
    except ImportError as exc:
        raise _import_error("trimesh bulunamadı", exc) from exc

    loaded = trimesh.load(mesh_path, force="mesh")
    if not isinstance(loaded, trimesh.Trimesh):
        raise RuntimeError(f"mesh yüklenemedi ya da desteklenmeyen içerik: {mesh_path!r}")
    return loaded


def _mesh_info(mesh) -> dict[str, Any]:
    return {
        "vertices": int(mesh.vertices.shape[0]),
        "faces": int(mesh.faces.shape[0]),
        "watertight": bool(mesh.is_watertight),
        "bounds": mesh.bounds.tolist(),
        "volume": float(mesh.volume) if mesh.is_watertight else None,
        "area": float(mesh.area),
    }


class MeshToolAdapter:
    def __init__(self) -> None:
        pass

    def load(self, model_path: str, **kwargs: Any) -> None:
        pass  # model gerektirmez

    def infer(self, params: dict) -> Iterator[dict]:
        op = params.get("op", "")
        if op == "convert":
            yield from self._convert(params)
        elif op == "decimate":
            yield from self._decimate(params)
        elif op == "repair":
            yield from self._repair(params)
        elif op == "info":
            yield from self._info(params)
        else:
            raise RuntimeError(f"bilinmeyen op: {op!r} (convert|decimate|repair|info)")

    def _mesh_path(self, params: dict) -> str:
        mesh_path = params.get("mesh_path", "")
        if not mesh_path:
            raise RuntimeError("mesh_path gerekli")
        return mesh_path

    def _export(self, mesh, output_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        if output_path.lower().endswith(".fbx"):
            # trimesh Autodesk'in kapalı FBX formatını yazamıyor — Blender
            # köprüsü kullanılır (bkz. _blender_bridge.py): önce geçici bir
            # GLB'ye export edilir, Blender onu FBX'e çevirir.
            import tempfile

            from studio_inference.adapters._blender_bridge import export_fbx

            with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
                tmp_glb = tmp.name
            try:
                mesh.export(tmp_glb)
                export_fbx(tmp_glb, output_path)
            finally:
                os.unlink(tmp_glb)
        else:
            mesh.export(output_path)

    def _convert(self, params: dict) -> Iterator[dict]:
        mesh_path = self._mesh_path(params)
        output_path = params.get("output_path", "")
        if not output_path:
            raise RuntimeError("output_path gerekli")
        mesh = _load_mesh(mesh_path)
        yield {"event": "progress", "message": "dışa aktarılıyor…"}
        self._export(mesh, output_path)
        yield {"event": "done", "result": {"path": output_path, **_mesh_info(mesh)}}

    def _decimate(self, params: dict) -> Iterator[dict]:
        mesh_path = self._mesh_path(params)
        output_path = params.get("output_path", "")
        if not output_path:
            raise RuntimeError("output_path gerekli")
        target_faces = params.get("target_faces")
        target_ratio = float(params.get("target_ratio", 0.5))

        mesh = _load_mesh(mesh_path)
        before = int(mesh.faces.shape[0])
        face_count = int(target_faces) if target_faces else max(4, int(before * target_ratio))

        yield {"event": "progress", "message": f"basitleştiriliyor ({before} -> {face_count} yüz)…"}
        try:
            simplified = mesh.simplify_quadric_decimation(face_count=face_count)
        except ImportError as exc:
            raise _import_error("fast_simplification bulunamadı", exc) from exc

        # Quadric decimation neredeyse her zaman küçük non-manifold artefaktlar
        # bırakır — best-effort onarım denenir (birçok durumda yeterli), ama
        # garanti DEĞİL: bazı geometriler trimesh'in hafif onarım araçlarıyla
        # (hatta canlı testte denenen pymeshlab'ın daha güçlü filtreleriyle
        # bile) tam watertight'a dönmüyor — bkz. docs/worker-envs.md. Sonuçtaki
        # `watertight` alanı her zaman gerçek durumu yansıtır, çağıran taraf
        # buna göre karar verir (ör. gerekirse `repair` op'unu ayrıca dener).
        if not simplified.is_watertight:
            from trimesh import repair as trimesh_repair

            yield {"event": "progress", "message": "basitleştirme sonrası onarılıyor…"}
            trimesh_repair.fix_winding(simplified)
            trimesh_repair.fix_normals(simplified)
            trimesh_repair.fill_holes(simplified)

        yield {"event": "progress", "message": "dışa aktarılıyor…"}
        self._export(simplified, output_path)
        yield {
            "event": "done",
            "result": {"path": output_path, "faces_before": before, **_mesh_info(simplified)},
        }

    def _repair(self, params: dict) -> Iterator[dict]:
        mesh_path = self._mesh_path(params)
        output_path = params.get("output_path", "")
        if not output_path:
            raise RuntimeError("output_path gerekli")
        from trimesh import repair as trimesh_repair

        mesh = _load_mesh(mesh_path)
        watertight_before = bool(mesh.is_watertight)

        yield {"event": "progress", "message": "normaller ve delikler onarılıyor…"}
        trimesh_repair.fix_winding(mesh)
        trimesh_repair.fix_normals(mesh)
        trimesh_repair.fill_holes(mesh)

        yield {"event": "progress", "message": "dışa aktarılıyor…"}
        self._export(mesh, output_path)
        yield {
            "event": "done",
            "result": {
                "path": output_path,
                "watertight_before": watertight_before,
                **_mesh_info(mesh),
            },
        }

    def _info(self, params: dict) -> Iterator[dict]:
        mesh_path = self._mesh_path(params)
        mesh = _load_mesh(mesh_path)
        yield {"event": "done", "result": {"path": mesh_path, **_mesh_info(mesh)}}

    def unload(self) -> None:
        pass

    def vram_estimate(self, params: dict) -> int:
        return 0  # CPU-only, VRAM gerektirmez
