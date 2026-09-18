import { OrbitControls, Stage } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { Suspense, useEffect, useMemo, useRef, useState } from "react";
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

// plan.md Faz 9 adım 1'in kademeli listesi TAMAMLANDI — yedi aile de canlı
// doğrulandı: TripoSR (hafif, hızlı), Stable Fast 3D (dokulu/tekstürlü),
// OpenLRM (vertex-renkli, yüksek çözünürlük), Hunyuan3D-2 (yalnızca shape —
// dokusuz, yoğun mesh), InstantMesh (LRM+FlexiCubes, zero123plus çoklu-
// görünüm difüzyonu), TRELLIS (sparse-voxel yapı + gerçek Gaussian-pişirmeli
// doku) ve CRM (iki aşamalı çoklu-görünüm difüzyonu + nvdiffrast UV
// pişirme). NVIDIA CUDA Toolkit 2026-07-20'de kuruldu (bkz.
// docs/worker-envs.md) — InstantMesh/TRELLIS/CRM'i bloke eden nvdiffrast/
// xformers/flash_attn engellerini kaldırdı (TRELLIS ve CRM'in kendi
// attention modüllerinde upstream'in 'sdpa' desenini genişlettik — bkz.
// vendor/trellis/NOTICE.md, vendor/crm/NOTICE.md). Hunyuan3D-2'nin texgen/
// paint (doku/boyama) kısmı ayrı, farklı bir engelle karşılaştı —
// upstream'in checkpoint'i kendi yükleme kodunun oluşturmadığı IP-Adapter
// anahtarları içeriyor (ayrıntı: vendor/hunyuan3d/NOTICE.md).
type Family = "triposr" | "sf3d" | "openlrm" | "hunyuan3d" | "instantmesh" | "trellis" | "crm";

const FAMILIES: { id: Family; label: string; modelNameHint: string }[] = [
  { id: "triposr", label: "TripoSR — hafif, hızlı (~10sn, tek renkli vertex)", modelNameHint: "TripoSR" },
  { id: "sf3d", label: "Stable Fast 3D — dokulu/tekstürlü (~15sn)", modelNameHint: "stable-fast-3d" },
  { id: "openlrm", label: "OpenLRM — yüksek çözünürlüklü vertex-renkli (~10sn + ilk kullanımda ek indirme)", modelNameHint: "openlrm" },
  { id: "hunyuan3d", label: "Hunyuan3D-2 — yoğun mesh (~9sn), opsiyonel Paint dokusu", modelNameHint: "hunyuan3d-dit" },
  { id: "instantmesh", label: "InstantMesh — vertex-renkli ya da gerçek UV doku (~12sn, yüksek VRAM)", modelNameHint: "instant_mesh" },
  { id: "trellis", label: "TRELLIS — gerçek dokulu, sparse-voxel (~28sn, yüksek VRAM)", modelNameHint: "ss_flow_img_dit" },
  { id: "crm", label: "CRM — gerçek dokulu, iki aşamalı çoklu-görünüm (~40sn, en yüksek VRAM)", modelNameHint: "CRM" },
];

const REMESH_OPTIONS: { id: string; label: string }[] = [
  { id: "none", label: "yok (ham çıktı)" },
  { id: "triangle", label: "üçgen" },
  { id: "quad", label: "dörtgen" },
];

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
  const [foregroundRatio, setForegroundRatio] = useState(0.85);
  // triposr'a özgü
  const [mcResolution, setMcResolution] = useState(256);
  // sf3d'ye özgü
  const [textureResolution, setTextureResolution] = useState(1024);
  const [remesh, setRemesh] = useState("none");
  const [targetVertexCount, setTargetVertexCount] = useState(-1);
  // openlrm'e özgü
  const [meshSize, setMeshSize] = useState(256);
  const [meshThres, setMeshThres] = useState(3.0);
  // hunyuan3d'ye özgü
  const [numInferenceSteps, setNumInferenceSteps] = useState(30);
  const [octreeResolution, setOctreeResolution] = useState(256);
  const [hyTexture, setHyTexture] = useState(false);
  // instantmesh'e özgü
  const [imSteps, setImSteps] = useState(50);
  const [imView, setImView] = useState<4 | 6>(6);
  const [imExportTexmap, setImExportTexmap] = useState(false);
  // trellis'e özgü
  const [trSteps, setTrSteps] = useState(25);
  const [trSimplify, setTrSimplify] = useState(0.95);
  const [trTextureSize, setTrTextureSize] = useState(1024);
  // crm'e özgü
  const [crmSteps, setCrmSteps] = useState(30);
  const [crmScale, setCrmScale] = useState(5.0);

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

  const activeFamily = FAMILIES.find((f) => f.id === family)!;

  // Model Browser'daki 3D modelleri seçili aileye göre filtrele (isim ipucuyla) —
  // yanlış aileye ait bir ağırlık dosyasının yanlışlıkla seçilmesini önler.
  const familyModels = useMemo(
    () => meshModels.filter((m) => m.name.toLowerCase().includes(activeFamily.modelNameHint.toLowerCase())),
    [meshModels, activeFamily],
  );

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

  // Aile değişince, o aileye ait olmayan bir model seçili kalmasın.
  useEffect(() => {
    setModelPath((cur) => (familyModels.some((m) => m.path === cur) ? cur : ""));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [family, meshModels]);

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
    const params: Record<string, unknown> = {
      family,
      model_path: modelPath,
      image_path: srcImage.path,
      remove_background: removeBg,
      foreground_ratio: foregroundRatio,
      ...(parentAssetId ? { parent_asset_id: parentAssetId } : {}),
    };
    if (family === "triposr") {
      params.mc_resolution = mcResolution;
    } else if (family === "sf3d") {
      params.texture_resolution = textureResolution;
      params.remesh = remesh;
      params.target_vertex_count = targetVertexCount;
    } else if (family === "openlrm") {
      params.mesh_size = meshSize;
      params.mesh_thres = meshThres;
    } else if (family === "hunyuan3d") {
      params.num_inference_steps = numInferenceSteps;
      params.octree_resolution = octreeResolution;
      params.texture = hyTexture;
    } else if (family === "instantmesh") {
      params.num_inference_steps = imSteps;
      params.view = imView;
      params.export_texmap = imExportTexmap;
    } else if (family === "trellis") {
      params.num_inference_steps = trSteps;
      params.simplify = trSimplify;
      params.texture_size = trTextureSize;
    } else {
      params.num_inference_steps = crmSteps;
      params.scale = crmScale;
    }
    try {
      await createJob("mesh", params);
      refresh();
    } catch (e) {
      setFormError(e instanceof Error ? e.message : String(e));
    }
  };

  const meshPathOf = (a: AssetInfo): string => {
    const result = a.meta.result as { path?: string } | undefined;
    return result?.path ?? a.path;
  };

  const familyOf = (a: AssetInfo): string =>
    String((a.meta.params?.family as string) ?? (a.meta.result?.family as string) ?? "");

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
          width: 300,
          padding: 12,
          borderRight: "1px solid #333",
          overflowY: "auto",
          display: "flex",
          flexDirection: "column",
          gap: 10,
        }}
      >
        <div>
          <label>Model ailesi</label>
          <select
            value={family}
            onChange={(e) => setFamily(e.target.value as Family)}
            style={{ width: "100%" }}
          >
            {FAMILIES.map((f) => (
              <option key={f.id} value={f.id}>
                {f.label}
              </option>
            ))}
          </select>
        </div>

        {family === "sf3d" && (
          <p style={{ opacity: 0.7, margin: 0 }}>
            Stable Fast 3D ağırlıkları <strong>Stability AI Community License</strong> ile
            gelir — araştırma/kişisel kullanım serbest; yıllık geliri 1M USD'yi aşan
            kuruluşlar için ayrı bir Enterprise License gerekir.
          </p>
        )}
        {family === "openlrm" && (
          <p style={{ opacity: 0.7, margin: 0 }}>
            OpenLRM ağırlıkları <strong>CC BY-NC 4.0</strong> ile gelir — yalnızca
            araştırma/kişisel kullanım, TİCARİ KULLANIM YOK. İlk kullanımda ayrıca
            ~330MB'lık bir DINOv2 kodlayıcı indirilir (Meta'nın sunucusundan, Model
            Manager'dan bağımsız).
          </p>
        )}
        {family === "hunyuan3d" && (
          <p style={{ opacity: 0.7, margin: 0 }}>
            Hunyuan3D-2 ağırlıkları <strong>Tencent Hunyuan 3D 2.0 Community
            License</strong> ile gelir — AB/UK/Güney Kore hariç her yerde serbest,
            aylık 1M aktif kullanıcıyı aşan ürünler için Tencent'ten ayrı lisans
            gerekir. Varsayılan çıktı <strong>shape (dokusuz mesh)</strong>;
            "Paint dokusu üret" açıksa delight + 6 görünümlü difüzyon + UV
            pişirme ile 2048² dokulu GLB üretilir — Model Manager'da
            hunyuan3d-delight-v2-0 ve hunyuan3d-paint-v2-0-turbo klasörleri
            gerekir, daha fazla VRAM ve süre ister (bkz. vendor/hunyuan3d/NOTICE.md).
          </p>
        )}
        {family === "instantmesh" && (
          <p style={{ opacity: 0.7, margin: 0 }}>
            Apache-2.0 (özel bir kullanım kısıtı yok). Diğer dört aileden{" "}
            <strong>belirgin daha fazla VRAM</strong> kullanır (canlı ölçüm:
            ~15.6GB) — zero123plus tam bir SD1.5-boyutlu difüzyon modeli
            taşıyor. İlk kullanımda ayrıca ~330MB'lık bir DINO kodlayıcı
            indirilir (Model Manager'dan bağımsız, HF'den).
          </p>
        )}
        {family === "trellis" && (
          <p style={{ opacity: 0.7, margin: 0 }}>
            TRELLIS'in kendisi MIT/Apache-2.0. Ama gerçek dokulu çıktı için
            kullanılan bir bileşen (Gaussian doku pişirme) <strong>Inria/MPII
            Gaussian-Splatting Lisansı</strong> ile gelir — yalnızca araştırma/
            değerlendirme amaçlı, TİCARİ KULLANIM YOK. Yedi aile arasında{" "}
            <strong>en yüksek VRAM kullananlardan biri</strong> (canlı ölçüm:
            ~11.6GB). İlk kullanımda ayrıca bir DINOv2 kodlayıcı indirilir
            (Model Manager'dan bağımsız, HF'den).
          </p>
        )}
        {family === "crm" && (
          <p style={{ opacity: 0.7, margin: 0 }}>
            MIT (özel bir kullanım kısıtı yok). İki büyük difüzyon ağırlığı
            (~6GB'ın her biri) aynı anda belleğe yükleniyor — ilk üretim
            isteği diğer ailelerden belirgin daha yavaş başlar. Yedi aile
            arasında <strong>en yüksek VRAM kullananlardan biri</strong>{" "}
            (canlı ölçüm: ~13.6GB).
          </p>
        )}

        <div>
          <label>Ağırlık (boşsa HF'den iner)</label>
          <select
            value={modelPath}
            onChange={(e) => setModelPath(e.target.value)}
            style={{ width: "100%" }}
          >
            <option value="">— varsayılan ({activeFamily.modelNameHint}) —</option>
            {familyModels.map((m) => (
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

        {family === "trellis" ? (
          <p style={{ opacity: 0.6, margin: 0 }}>
            TRELLIS arka plan temizliğini kendi ön işlemesinde (DINOv2 için gereken
            518x518 kırpma ile birlikte) otomatik yapar — ayrı bir ayar yok.
          </p>
        ) : (
          <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <input type="checkbox" checked={removeBg} onChange={(e) => setRemoveBg(e.target.checked)} />
            arka planı kaldır (rembg)
          </label>
        )}
        {removeBg && family !== "trellis" && family !== "crm" && (
          <div>
            <label>Önplan oranı: {foregroundRatio.toFixed(2)}</label>
            <input
              type="range"
              min={0.5}
              max={1}
              step={0.05}
              value={foregroundRatio}
              onChange={(e) => setForegroundRatio(Number(e.target.value))}
              style={{ width: "100%" }}
            />
          </div>
        )}

        {family === "triposr" && (
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
        )}
        {family === "sf3d" && (
          <>
            <div>
              <label>Doku çözünürlüğü: {textureResolution}px</label>
              <input
                type="range"
                min={512}
                max={2048}
                step={512}
                value={textureResolution}
                onChange={(e) => setTextureResolution(Number(e.target.value))}
                style={{ width: "100%" }}
              />
            </div>
            <div>
              <label>Yeniden örgüleme (remesh)</label>
              <select
                value={remesh}
                onChange={(e) => setRemesh(e.target.value)}
                style={{ width: "100%" }}
              >
                {REMESH_OPTIONS.map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.label}
                  </option>
                ))}
              </select>
            </div>
            {remesh !== "none" && (
              <div>
                <label>Hedef vertex sayısı</label>
                <input
                  type="number"
                  min={100}
                  value={targetVertexCount === -1 ? "" : targetVertexCount}
                  placeholder="otomatik"
                  onChange={(e) =>
                    setTargetVertexCount(e.target.value ? Number(e.target.value) : -1)
                  }
                  style={{ width: "100%" }}
                />
              </div>
            )}
          </>
        )}
        {family === "openlrm" && (
          <>
            <div>
              <label>Marching cubes çözünürlüğü: {meshSize}</label>
              <input
                type="range"
                min={128}
                max={384}
                step={64}
                value={meshSize}
                onChange={(e) => setMeshSize(Number(e.target.value))}
                style={{ width: "100%" }}
              />
              <p style={{ opacity: 0.6, margin: "2px 0 0" }}>
                yüksek değerler çok daha yoğun mesh üretir (384 ≈ 289K vertex)
              </p>
            </div>
            <div>
              <label>Isosurface eşiği: {meshThres.toFixed(1)}</label>
              <input
                type="range"
                min={0.5}
                max={10}
                step={0.5}
                value={meshThres}
                onChange={(e) => setMeshThres(Number(e.target.value))}
                style={{ width: "100%" }}
              />
            </div>
          </>
        )}
        {family === "hunyuan3d" && (
          <>
            <div>
              <label>Diffusion adım sayısı: {numInferenceSteps}</label>
              <input
                type="range"
                min={10}
                max={50}
                step={5}
                value={numInferenceSteps}
                onChange={(e) => setNumInferenceSteps(Number(e.target.value))}
                style={{ width: "100%" }}
              />
            </div>
            <div>
              <label>Octree çözünürlüğü: {octreeResolution}</label>
              <input
                type="range"
                min={128}
                max={384}
                step={64}
                value={octreeResolution}
                onChange={(e) => setOctreeResolution(Number(e.target.value))}
                style={{ width: "100%" }}
              />
              <p style={{ opacity: 0.6, margin: "2px 0 0" }}>
                yüksek değerler çok daha yoğun mesh üretir (384 ≈ 584K vertex, ~6GB VRAM)
              </p>
            </div>
            <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <input
                type="checkbox"
                checked={hyTexture}
                onChange={(e) => setHyTexture(e.target.checked)}
              />
              Paint dokusu üret (delight + 6 görünüm difüzyonu + UV pişirme, 2048²)
            </label>
            {hyTexture && (
              <p style={{ opacity: 0.6, margin: 0 }}>
                mesh önce 40K yüze indirgenir; GLB'ye gömülü doku üretilir (OBJ: mtl+png)
              </p>
            )}
          </>
        )}
        {family === "instantmesh" && (
          <>
            <div>
              <label>Çoklu-görünüm difüzyon adım sayısı: {imSteps}</label>
              <input
                type="range"
                min={10}
                max={75}
                step={5}
                value={imSteps}
                onChange={(e) => setImSteps(Number(e.target.value))}
                style={{ width: "100%" }}
              />
            </div>
            <div>
              <label>Görünüm sayısı</label>
              <select
                value={imView}
                onChange={(e) => setImView(Number(e.target.value) as 4 | 6)}
                style={{ width: "100%" }}
              >
                <option value={6}>6 (daha kaliteli)</option>
                <option value={4}>4 (daha hızlı)</option>
              </select>
            </div>
            <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <input
                type="checkbox"
                checked={imExportTexmap}
                onChange={(e) => setImExportTexmap(e.target.checked)}
              />
              gerçek UV dokusu üret (yavaş — nvdiffrast+xatlas)
            </label>
            {!imExportTexmap && (
              <p style={{ opacity: 0.6, margin: 0 }}>kapalıyken vertex-renkli, hızlı çıktı (upstream varsayılanı)</p>
            )}
          </>
        )}
        {family === "trellis" && (
          <>
            <div>
              <label>Örnekleme adım sayısı: {trSteps}</label>
              <input
                type="range"
                min={10}
                max={50}
                step={5}
                value={trSteps}
                onChange={(e) => setTrSteps(Number(e.target.value))}
                style={{ width: "100%" }}
              />
              <p style={{ opacity: 0.6, margin: "2px 0 0" }}>
                hem sparse-yapı hem structured-latent örneklemesi için (upstream vars. 25)
              </p>
            </div>
            <div>
              <label>Basitleştirme oranı: {trSimplify.toFixed(2)}</label>
              <input
                type="range"
                min={0}
                max={0.99}
                step={0.05}
                value={trSimplify}
                onChange={(e) => setTrSimplify(Number(e.target.value))}
                style={{ width: "100%" }}
              />
              <p style={{ opacity: 0.6, margin: "2px 0 0" }}>
                kaldırılacak yüz oranı — ham çıktı çok yoğun, upstream vars. 0.95
              </p>
            </div>
            <div>
              <label>Doku çözünürlüğü: {trTextureSize}px</label>
              <input
                type="range"
                min={512}
                max={2048}
                step={512}
                value={trTextureSize}
                onChange={(e) => setTrTextureSize(Number(e.target.value))}
                style={{ width: "100%" }}
              />
            </div>
          </>
        )}
        {family === "crm" && (
          <>
            <div>
              <label>Difüzyon adım sayısı: {crmSteps}</label>
              <input
                type="range"
                min={10}
                max={50}
                step={5}
                value={crmSteps}
                onChange={(e) => setCrmSteps(Number(e.target.value))}
                style={{ width: "100%" }}
              />
              <p style={{ opacity: 0.6, margin: "2px 0 0" }}>
                pixel + CCM/XYZ aşamalarının İKİSİ için de (upstream vars. 50)
              </p>
            </div>
            <div>
              <label>CFG ölçeği: {crmScale.toFixed(1)}</label>
              <input
                type="range"
                min={1}
                max={10}
                step={0.5}
                value={crmScale}
                onChange={(e) => setCrmScale(Number(e.target.value))}
                style={{ width: "100%" }}
              />
            </div>
          </>
        )}

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
          {results.map((a) => {
            const fam = familyOf(a);
            return (
              <button
                key={a.id}
                onClick={() => setSelected(a)}
                style={{
                  textAlign: "left",
                  background: selected?.id === a.id ? "#2a4" : undefined,
                }}
                title={fam ? `aile: ${fam}` : undefined}
              >
                {meshPathOf(a).split(/[\\/]/).pop()}
              </button>
            );
          })}
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
