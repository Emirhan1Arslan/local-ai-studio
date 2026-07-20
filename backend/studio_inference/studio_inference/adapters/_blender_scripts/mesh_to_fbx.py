"""Blender'ın KENDİ Python yorumlayıcısıyla `blender --background --python
mesh_to_fbx.py -- --input X --output Y.fbx` şeklinde çalıştırılır (bkz.
adapters/_blender_bridge.py) — worker env'inin torch/trimesh'i burada YOK,
yalnızca `bpy` var. Varsayılan sahneyi temizler, girdiyi (glb/gltf/obj/ply/
stl) içe aktarır, FBX olarak dışa aktarır.

trimesh FBX (Autodesk'in kapalı formatı) dışa aktaramıyor; pyassimp'in FBX
export'u güvenilmez (bkz. docs/worker-envs.md) — Blender'ın kendi olgun FBX
exporter'ı kullanılıyor, ayrıca Blender kendi ürettiği FBX'i sorunsuz geri
okuyabildiği için plan.md'nin DoD'si ("FBX Blender'da import edilir")
otomatik sağlanmış oluyor.
"""

import argparse
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args(argv)

bpy.ops.wm.read_factory_settings(use_empty=True)

ext = args.input.rsplit(".", 1)[-1].lower()
importers = {
    "glb": lambda: bpy.ops.import_scene.gltf(filepath=args.input),
    "gltf": lambda: bpy.ops.import_scene.gltf(filepath=args.input),
    "obj": lambda: bpy.ops.wm.obj_import(filepath=args.input),
    "ply": lambda: bpy.ops.wm.ply_import(filepath=args.input),
    "stl": lambda: bpy.ops.wm.stl_import(filepath=args.input),
}
if ext not in importers:
    raise SystemExit(f"desteklenmeyen girdi formatı: {ext!r} ({', '.join(importers)})")
importers[ext]()

bpy.ops.export_scene.fbx(filepath=args.output, use_selection=False)
print(f"FBX_BRIDGE_OK:{args.output}")
