import { OrbitControls, Stage } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { Suspense, useEffect, useRef, useState } from "react";
import { useLoader } from "@react-three/fiber";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import {
  apiUrl,
  createJob,
  fetchAssets,
  fetchJobs,
  fetchModels,
  uploadImageFile,
  type AssetInfo,
  type JobInfo,
  type ModelInfo,
} from "../api";
import { useStudio } from "../state";

// plan.md Faz 9 adım 1'in kademeli listesi: TripoSR (hafif, hızlı) ilk model —
// Stable Fast 3D/InstantMesh/TRELLIS/Hunyuan3D-2/OpenLRM-CRM henüz eklenmedi.
type Family = "triposr";
const FAMILIES: [Family, string][] = [["triposr", "TripoSR (hafif, hızlı)"]];

type ExportFormat = "glb" | "obj" | "stl" | "ply" | "fbx";
const EXPORT_FORMATS: ExportFormat[] = ["glb", "obj", "stl", "ply", "fbx"];

function outUrl(path: string): string {
  return apiUrl(`/outputs/${path.split(/[\\/]/).pop()}`);
}

function MeshModel({ url, wireframe }: { url: string; wireframe: boolean }) {
  const gltf = useLoader(GLTFLoader, url);
  useEffect(() => {
    gltf.scene.traverse((obj) => {
      const mesh = obj as unknown as { isMesh?: boolean; material?: { wireframe?: boolean } };
      if (mesh.isMesh && mesh.material) mesh.material.wireframe = wireframe;
    });
  }, [gltf, wireframe]);
  return <primitive object={gltf.scene} />;
}

export default function MeshPanel() {
  const { events, pendingMeshImage, consumeMeshRequest } = useStudio();

  // Üretim
  const [family, setFamily] = useState<Family>("triposr");
  const [meshModels, setMeshModels] = useState<ModelInfo[]>([]);
  const [modelPath, setModelPath] = useState("");
  const [srcImage, setSrcImage] = useState<{ path: string; url: string } | null>(null);
  const [parentAssetId, setParentAssetId] = useState("");
  const [removeBg, setRemoveBg] = useState(true);
  const [mcResolution, setMcResolution] = useState(256);

  const [jobs, setJobs] = useState<JobInfo[]>([]);
  const [results, setResults] = useState<AssetInfo[]>([]);
  const [selected, setSelected] = useState<AssetInfo | null>(null);
  const [wireframe, setWireframe] = useState(false);
  const [formError, setFormError] = useState("");
  const [toolBusy, setToolBusy] = useState(false);
  const [progress, setProgress] = useState<Record<string, { value: number; message: string }>>(
    {},
  );
  const lastEvent = useRef(0);

  const refresh = () => {
    fetchJobs()
      .then((all) => setJobs(all.filter((j) => j.kind === "mesh" || j.kind === "meshtool")))
      .catch(() => undefined);
    // "üret" (kind=mesh) ve "araçlar" (kind=meshtool convert/decimate/repair) tek galeride birleşir.
    Promise.all([fetchAssets("mesh", 24), fetchAssets("meshtool", 24)])
      .then(([a, b]) => {
        const all = [...a, ...b].sort((x, y) => (x.created_at < y.created_at ? 1 : -1));
        setResults(all);
        setSelected((cur) => cur ?? all[0] ?? null);
      })
      .catch(() => undefined);
  };

  useEffect(() => {
    refresh();
    fetchModels("3d").then(setMeshModels).catch(() => undefined);
    const t = setInterval(refresh, 4000);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Görüntü→3D köprüsü (plan.md Faz 9 adım 5): GeneratePanel'den "3D'ye gönder"
  // ile gelen görsel — Asset Graph'ta türev olarak bağlanması için parent_asset_id
  // üretim job'una geçiliyor (register_asset_for_job zaten bunu okuyor).
  useEffect(() => {
    if (!pendingMeshImage) return;
    setSrcImage({ path: pendingMeshImage.path, url: pendingMeshImage.url });
    setParentAssetId(pendingMeshImage.assetId);
    consumeMeshRequest();
  }, [pendingMeshImage, consumeMeshRequest]);

  useEffect(() => {
    let needsRefresh = false;
    for (let i = lastEvent.current; i < events.length; i++) {
      const e = events[i];
      if (e.topic === "job.progress") {
        const { id, value, message } = e.data as { id: string; value: number; message: string };
        setProgress((p) => ({ ...p, [id]: { value, message } }));
      } else if (e.topic.startsWith("job.")) {
        needsRefresh = true;
      }
    }
    lastEvent.current = events.length;
    if (needsRefresh) refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [events]);

  const running = jobs.find((j) => j.status === "running" || j.status === "queued");

  const pickImage = async (file: File) => {
    setFormError("");
    setParentAssetId(""); // elle seçilen görsel Asset Graph'a bağlı değil
    try {
      const up = await uploadImageFile(file);
      setSrcImage({ path: up.path, url: URL.createObjectURL(file) });
    } catch (e) {
      setFormError(e instanceof Error ? e.message : String(e));
    }
  };

  const generate = async () => {
    setFormError("");
    if (!srcImage) {
      setFormError("önce bir görsel seçin");
      return;
    }
    try {
      await createJob("mesh", {
        family,
        model_path: modelPath,
        image_path: srcImage.path,
        remove_background: removeBg,
        mc_resolution: mcResolution,
        ...(parentAssetId ? { parent_asset_id: parentAssetId } : {}),
      });
      refresh();
    } catch (e) {
      setFormError(e instanceof Error ? e.message : String(e));
    }
  };

  const meshPathOf = (a: AssetInfo): string => {
    const result = a.meta.result as { path?: string } | undefined;
    return result?.path ?? a.path;
  };

  const runTool = async (op: "convert" | "decimate" | "repair", format: ExportFormat) => {
    if (!selected) return;
    setFormError("");
    setToolBusy(true);
    try {
      await createJob("meshtool", {
        op,
        mesh_path: meshPathOf(selected),
        format,
        ...(op === "decimate" ? { target_ratio: 0.5 } : {}),
      });
      refresh();
    } catch (e) {
      setFormError(e instanceof Error ? e.message : String(e));
    } finally {
      setToolBusy(false);
    }
  };

  const selectedUrl = selected ? outUrl(meshPathOf(selected)) : "";
  const selectedIsGlb = selectedUrl.toLowerCase().endsWith(".glb");

  return (
    <div style={{ display: "flex", height: "100%", fontSize: 13 }}>
      <div
        style={{
          width: 280,
          padding: 12,
          borderRight: "1px solid #333",
          overflowY: "auto",
          display: "flex",
          flexDirection: "column",
          gap: 10,
        }}
      >
        <div>
          <label>Aile</label>
          <select
            value={family}
            onChange={(e) => setFamily(e.target.value as Family)}
            style={{ width: "100%" }}
          >
            {FAMILIES.map(([f, label]) => (
              <option key={f} value={f}>
                {label}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label>Model (boşsa HF'den iner)</label>
          <select
            value={modelPath}
            onChange={(e) => setModelPath(e.target.value)}
            style={{ width: "100%" }}
          >
            <option value="">— varsayılan —</option>
            {meshModels.map((m) => (
              <option key={m.id} value={m.path}>
                {m.name}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label>Görsel</label>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => e.target.files?.[0] && pickImage(e.target.files[0])}
          />
          {srcImage && (
            <img
              src={srcImage.url}
              alt=""
              style={{ width: "100%", marginTop: 6, borderRadius: 4 }}
            />
          )}
          {parentAssetId && (
            <div style={{ opacity: 0.6, marginTop: 4 }}>Üretim panelinden gönderildi</div>
          )}
        </div>

        <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <input type="checkbox" checked={removeBg} onChange={(e) => setRemoveBg(e.target.checked)} />
          arka planı kaldır (rembg)
        </label>

        <div>
          <label>Marching cubes çözünürlüğü: {mcResolution}</label>
          <input
            type="range"
            min={64}
            max={384}
            step={32}
            value={mcResolution}
            onChange={(e) => setMcResolution(Number(e.target.value))}
            style={{ width: "100%" }}
          />
        </div>

        <button onClick={generate} disabled={!!running}>
          {running ? "üretiliyor…" : "Üret"}
        </button>

        {running && progress[running.id] && (
          <div style={{ opacity: 0.8 }}>{progress[running.id].message}</div>
        )}
        {formError && <div style={{ color: "#e66" }}>{formError}</div>}

        <hr style={{ width: "100%", borderColor: "#333" }} />

        <div>Geçmiş</div>
        <div style={{ display: "flex", flexDirection: "column", gap: 4, overflowY: "auto" }}>
          {results.map((a) => (
            <button
              key={a.id}
              onClick={() => setSelected(a)}
              style={{
                textAlign: "left",
                background: selected?.id === a.id ? "#2a4" : undefined,
              }}
            >
              {meshPathOf(a).split(/[\\/]/).pop()}
            </button>
          ))}
        </div>
      </div>

      <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
        <div style={{ flex: 1, position: "relative", background: "#1a1a1a" }}>
          {selected && selectedIsGlb ? (
            <Canvas camera={{ position: [2, 2, 2], fov: 45 }}>
              <Suspense fallback={null}>
                <Stage environment="city" intensity={0.6} adjustCamera>
                  <MeshModel key={selectedUrl} url={selectedUrl} wireframe={wireframe} />
                </Stage>
              </Suspense>
              <OrbitControls makeDefault />
            </Canvas>
          ) : (
            <div style={{ padding: 20, opacity: 0.6 }}>
              {selected
                ? "bu format doğrudan önizlenemiyor (yalnızca GLB) — dışa aktarılan dosyayı indirin"
                : "henüz bir mesh üretilmedi"}
            </div>
          )}
        </div>

        <div
          style={{
            padding: 10,
            borderTop: "1px solid #333",
            display: "flex",
            gap: 8,
            alignItems: "center",
            flexWrap: "wrap",
          }}
        >
          <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <input
              type="checkbox"
              checked={wireframe}
              onChange={(e) => setWireframe(e.target.checked)}
            />
            wireframe
          </label>
          <span style={{ opacity: 0.6 }}>|</span>
          <span>dışa aktar:</span>
          {EXPORT_FORMATS.map((fmt) => (
            <button key={fmt} disabled={!selected || toolBusy} onClick={() => runTool("convert", fmt)}>
              {fmt.toUpperCase()}
            </button>
          ))}
          <span style={{ opacity: 0.6 }}>|</span>
          <button disabled={!selected || toolBusy} onClick={() => runTool("decimate", "glb")}>
            basitleştir (%50)
          </button>
          <button disabled={!selected || toolBusy} onClick={() => runTool("repair", "glb")}>
            onar
          </button>
          {selected && (
            <a href={selectedUrl} download style={{ marginLeft: "auto" }}>
              indir
            </a>
          )}
        </div>
      </div>
    </div>
  );
}
