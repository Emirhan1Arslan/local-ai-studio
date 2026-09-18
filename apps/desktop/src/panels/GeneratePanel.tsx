import { useEffect, useRef, useState } from "react";
import {
  apiUrl,
  createJob,
  createJobBatch,
  fetchAssets,
  fetchJobs,
  fetchModels,
  regenerateAsset,
  uploadDataUrl,
  uploadImageFile,
  type AssetInfo,
  type JobInfo,
  type ModelInfo,
} from "../api";
import { useStudio } from "../state";

interface LoraPick {
  path: string;
  name: string;
  weight: number;
}

interface InitPick {
  path: string; // backend dosya yolu (iş parametresi)
  url: string; // tarayıcı önizleme URL'i
  assetId?: string; // galeriden geldiyse: Asset Graph parent bağı
}

type Mode = "txt2img" | "img2img" | "inpaint" | "outpaint";

const MODE_LABELS: Record<Mode, string> = {
  txt2img: "metin→görsel",
  img2img: "görsel→görsel",
  inpaint: "inpaint",
  outpaint: "outpaint",
};

const SAMPLERS: [string, string][] = [
  ["", "sampler: model varsayılanı"],
  ["euler", "Euler"],
  ["euler_a", "Euler a"],
  ["dpmpp_2m", "DPM++ 2M"],
  ["dpmpp_2m_karras", "DPM++ 2M Karras"],
  ["dpmpp_sde_karras", "DPM++ SDE Karras"],
  ["unipc", "UniPC"],
  ["ddim", "DDIM"],
  ["heun", "Heun"],
  ["lms", "LMS"],
];

function assetUrl(a: AssetInfo): string {
  return apiUrl(`/outputs/${a.path.split(/[\\/]/).pop()}`);
}

const snap = (v: number) => Math.min(2048, Math.max(512, Math.round(v / 8) * 8));

// Yeni nesil aileler ayrı worker env'inde (envs/imagex) çalışır — iş türü "imagex".
// Model Manager `family` alanını klasörden çıkarır (studio_models/metadata.py::group_family).
const NEW_GEN_FAMILIES = new Set(["zimage", "hidream_o1"]);
const NEW_GEN_DEFAULTS: Record<string, { steps: number; cfg: number; label: string }> = {
  zimage: { steps: 9, cfg: 0, label: "Z-Image-Turbo — 9 adım, guidance 0 (Apache-2.0)" },
  hidream_o1: { steps: 28, cfg: 0, label: "HiDream-O1 — Dev 28 adım / guidance 0, Full 50 / 5 (MIT)" },
};
function newGenDefaults(m: ModelInfo | undefined): { steps: number; cfg: number; label: string } | null {
  if (!m || !NEW_GEN_FAMILIES.has(m.family)) return null;
  const d = NEW_GEN_DEFAULTS[m.family];
  if (m.family === "hidream_o1" && !m.name.toLowerCase().includes("dev")) return { ...d, steps: 50, cfg: 5 };
  return d;
}

function MaskEditor({
  src,
  canvasRef,
}: {
  src: string;
  canvasRef: React.RefObject<HTMLCanvasElement>;
}) {
  const [brush, setBrush] = useState(48);
  const imgRef = useRef<HTMLImageElement>(null);
  const drawing = useRef(false);

  const syncSize = () => {
    const img = imgRef.current;
    const c = canvasRef.current;
    if (img && c && (c.width !== img.naturalWidth || c.height !== img.naturalHeight)) {
      c.width = img.naturalWidth;
      c.height = img.naturalHeight;
    }
  };

  const paint = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const c = canvasRef.current;
    if (!c) return;
    const rect = c.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * c.width;
    const y = ((e.clientY - rect.top) / rect.height) * c.height;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "#fff";
    ctx.beginPath();
    ctx.arc(x, y, brush, 0, Math.PI * 2);
    ctx.fill();
  };

  return (
    <div className="mask-wrap">
      <div className="mask-stage">
        <img ref={imgRef} src={src} onLoad={syncSize} alt="" draggable={false} />
        <canvas
          ref={canvasRef}
          onPointerDown={(e) => {
            drawing.current = true;
            e.currentTarget.setPointerCapture(e.pointerId);
            paint(e);
          }}
          onPointerMove={(e) => drawing.current && paint(e)}
          onPointerUp={() => (drawing.current = false)}
          onPointerLeave={() => (drawing.current = false)}
        />
      </div>
      <div className="dl-row">
        <label className="gen-num">
          fırça{" "}
          <input
            type="number"
            min={4}
            max={256}
            value={brush}
            onChange={(e) => setBrush(Number(e.target.value))}
          />
        </label>
        <button
          type="button"
          onClick={() => {
            const c = canvasRef.current;
            c?.getContext("2d")?.clearRect(0, 0, c.width, c.height);
          }}
        >
          maskeyi temizle
        </button>
        <span className="muted">beyaza boyanan alan yeniden çizilir</span>
      </div>
    </div>
  );
}

export default function GeneratePanel() {
  const { events, connected, sendToMesh } = useStudio();
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loraModels, setLoraModels] = useState<ModelInfo[]>([]);
  const [loras, setLoras] = useState<LoraPick[]>([]);
  const [batch, setBatch] = useState(1);
  const [modelPath, setModelPath] = useState("");
  const [prompt, setPrompt] = useState("");
  const [negative, setNegative] = useState("");
  const [size, setSize] = useState("1024x1024");
  const [customW, setCustomW] = useState(1024);
  const [customH, setCustomH] = useState(1024);
  const [steps, setSteps] = useState(30);
  const [cfg, setCfg] = useState(7);
  const [seed, setSeed] = useState(-1);
  const [sampler, setSampler] = useState("");
  const [vram, setVram] = useState("max");
  const [mode, setMode] = useState<Mode>("txt2img");
  const [init, setInit] = useState<InitPick | null>(null);
  const [strength, setStrength] = useState(0.75);
  const [op, setOp] = useState({ left: 0, right: 256, top: 0, bottom: 0 });
  const [cnModels, setCnModels] = useState<ModelInfo[]>([]);
  const [cnPick, setCnPick] = useState(""); // "" kapalı · model yolu · "__custom"
  const [cnModel, setCnModel] = useState(""); // elle girilen yol / HF repo id
  const [cnImage, setCnImage] = useState<{ path: string; name: string } | null>(null);
  const [cnPrep, setCnPrep] = useState("canny");
  const [cnScale, setCnScale] = useState(0.8);
  const [ipImage, setIpImage] = useState<{ path: string; name: string } | null>(null);
  const [ipScale, setIpScale] = useState(0.6);
  const [imageJobs, setImageJobs] = useState<JobInfo[]>([]);
  const [assets, setAssets] = useState<AssetInfo[]>([]);
  const [progress, setProgress] = useState<
    Record<string, { value: number; message: string; preview?: string }>
  >({});
  const [formError, setFormError] = useState("");
  const [busy, setBusy] = useState(false);
  const lastEvent = useRef(0);
  const maskCanvasRef = useRef<HTMLCanvasElement>(null);

  const refreshJobs = () => {
    fetchJobs()
      .then((all) => setImageJobs(all.filter((j) => j.kind === "image" || j.kind === "imagex")))
      .catch(() => undefined);
    fetchAssets("image").then(setAssets).catch(() => undefined);
  };

  useEffect(() => {
    if (!connected) return;
    fetchModels()
      .then((all) => {
        const cands = all.filter(
          (m) => (m.kind === "checkpoint" || m.kind === "image") && m.format === "safetensors",
        );
        setModels(cands);
        if (cands.length > 0 && !modelPath) setModelPath(cands[0].path);
        setLoraModels(all.filter((m) => m.kind === "lora" && m.format === "safetensors"));
        setCnModels(all.filter((m) => m.kind === "controlnet" && m.format === "safetensors"));
      })
      .catch(() => setModels([]));
    refreshJobs();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [connected]);

  const selectedModel = models.find((m) => m.path === modelPath);
  const newGen = newGenDefaults(selectedModel);

  // Aile değişince adım/guidance varsayılanlarını modele uydur (SDXL'e dönüşte 30/7).
  useEffect(() => {
    const d = newGenDefaults(models.find((m) => m.path === modelPath));
    if (d) {
      setSteps(d.steps);
      setCfg(d.cfg);
    } else if (NEW_GEN_FAMILIES.size) {
      setSteps((s) => (s === 9 || s === 28 || s === 50 ? 30 : s));
      setCfg((c) => (c === 0 || c === 5 ? 7 : c));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelPath]);

  useEffect(() => {
    let needsRefresh = false;
    for (let i = lastEvent.current; i < events.length; i++) {
      const e = events[i];
      if (e.topic === "job.progress") {
        const { id, value, message, preview } = e.data as {
          id: string;
          value: number;
          message: string;
          preview?: string;
        };
        setProgress((p) => ({
          ...p,
          [id]: { value, message, preview: preview ?? p[id]?.preview },
        }));
      } else if (e.topic.startsWith("job.")) {
        needsRefresh = true;
      }
    }
    lastEvent.current = events.length;
    if (needsRefresh) refreshJobs();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [events]);

  const running = imageJobs.find((j) => j.status === "running" || j.status === "queued");

  // Kaynak görsel seçildiğinde çıktı boyutunu görselin (8'e yuvarlanmış) boyutuna çek
  const adoptInit = (pick: InitPick) => {
    setInit(pick);
    const probe = new Image();
    probe.onload = () => {
      setCustomW(snap(probe.naturalWidth));
      setCustomH(snap(probe.naturalHeight));
      setSize("custom");
    };
    probe.src = pick.url;
  };

  const pickInitFile = async (file: File) => {
    setFormError("");
    try {
      const up = await uploadImageFile(file);
      adoptInit({ path: up.path, url: URL.createObjectURL(file) });
    } catch (e) {
      setFormError(e instanceof Error ? e.message : String(e));
    }
  };

  const exportMask = (): string | null => {
    const c = maskCanvasRef.current;
    if (!c || !c.width) return null;
    const out = document.createElement("canvas");
    out.width = c.width;
    out.height = c.height;
    const ctx = out.getContext("2d");
    if (!ctx) return null;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, out.width, out.height);
    ctx.drawImage(c, 0, 0);
    return out.toDataURL("image/png");
  };

  const generate = async (ev: React.FormEvent) => {
    ev.preventDefault();
    setFormError("");
    const [w, h] =
      size === "custom" ? [snap(customW), snap(customH)] : size.split("x").map(Number);
    const params: Record<string, unknown> = {
      model_path: modelPath,
      prompt,
      negative_prompt: negative,
      width: w,
      height: h,
      steps,
      cfg,
      seed,
      ...(loras.length > 0 && {
        loras: loras.map((l) => ({ path: l.path, weight: l.weight })),
      }),
    };
    const kind = newGen ? "imagex" : "image";
    if (newGen) {
      // imagex aileleri: sampler/LoRA/ControlNet/IP-Adapter SDXL'e özgü; inpaint/outpaint yok.
      delete params.loras;
      if (mode === "inpaint" || mode === "outpaint") {
        setFormError("bu model ailesi yalnız txt2img ve img2img/edit destekliyor");
        return;
      }
      if (vram !== "max") params.vram_mode = "balanced";
    } else {
      if (sampler) params.sampler = sampler;
      if (vram !== "max") params.vram_mode = vram;
    }
    if (mode !== "txt2img") {
      if (!init) {
        setFormError("kaynak görsel seçin (dosya yükleyin ya da galeriden 'kaynak al')");
        return;
      }
      // HiDream-O1 img2img yerine referans görselli düzenleme ("edit") yapar.
      params.mode = newGen && selectedModel?.family === "hidream_o1" && mode === "img2img" ? "edit" : mode;
      params.init_image = init.path;
      params.strength = strength;
      if (init.assetId) params.parent_asset_id = init.assetId;
    }
    setBusy(true);
    try {
      if (mode === "inpaint") {
        const maskUrl = exportMask();
        if (!maskUrl) {
          setFormError("maske çizin (kaynak görselin üzerine boyayın)");
          return;
        }
        const up = await uploadDataUrl("mask.png", maskUrl);
        params.mask_image = up.path;
      }
      if (mode === "outpaint") params.outpaint = { ...op };
      const cnEffective = cnPick === "__custom" ? cnModel.trim() : cnPick;
      if (!newGen && mode === "txt2img" && cnEffective && cnImage) {
        params.controlnets = [
          { model: cnEffective, image: cnImage.path, scale: cnScale, preprocess: cnPrep },
        ];
      }
      if (!newGen && mode === "txt2img" && ipImage) {
        params.ip_adapter = { image: ipImage.path, scale: ipScale };
      }
      if (batch > 1) await createJobBatch(kind, params, batch);
      else await createJob(kind, params);
      refreshJobs();
    } catch (e) {
      setFormError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const runningPreview = running ? progress[running.id]?.preview : undefined;

  return (
    <div className="panel-body">
      <form className="dl-form" onSubmit={generate}>
        <div className="mode-tabs">
          {(Object.keys(MODE_LABELS) as Mode[]).map((m) => (
            <button
              key={m}
              type="button"
              className={mode === m ? "active" : ""}
              onClick={() => setMode(m)}
            >
              {MODE_LABELS[m]}
            </button>
          ))}
        </div>
        <select value={modelPath} onChange={(e) => setModelPath(e.target.value)} required>
          <option value="" disabled>
            {models.length === 0 ? "checkpoint bulunamadı — İndirmeler panelinden indirin" : "model seç"}
          </option>
          {models.map((m) => (
            <option key={m.id} value={m.path}>
              {m.name}
              {NEW_GEN_FAMILIES.has(m.family) ? " · yeni nesil" : ""}
            </option>
          ))}
        </select>
        {newGen && (
          <p className="muted" style={{ margin: 0 }}>
            {newGen.label}. Ayrı worker env'inde (envs/imagex) çalışır; sampler/LoRA/ControlNet/
            IP-Adapter ve inpaint/outpaint bu ailede yok, negative prompt yalnız guidance &gt; 1 iken etkili.
          </p>
        )}
        <textarea
          placeholder="prompt"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={3}
          required
        />
        <textarea
          placeholder="negatif prompt (opsiyonel)"
          value={negative}
          onChange={(e) => setNegative(e.target.value)}
          rows={2}
        />

        {mode !== "txt2img" && (
          <div className="dl-row">
            <label className="file-pick">
              kaynak görsel…
              <input
                type="file"
                accept="image/png,image/jpeg,image/webp"
                onChange={(e) => e.target.files?.[0] && pickInitFile(e.target.files[0])}
              />
            </label>
            {init && <img className="init-thumb" src={init.url} alt="" />}
            {init && (
              <button type="button" onClick={() => setInit(null)}>
                kaldır
              </button>
            )}
            <label className="gen-num">
              güç{" "}
              <input
                type="number"
                min={0.05}
                max={1}
                step={0.05}
                value={strength}
                onChange={(e) => setStrength(Number(e.target.value))}
              />
            </label>
          </div>
        )}
        {mode === "inpaint" && init && <MaskEditor src={init.url} canvasRef={maskCanvasRef} />}
        {mode === "outpaint" && (
          <div className="dl-row">
            {(["left", "right", "top", "bottom"] as const).map((k) => (
              <label className="gen-num" key={k}>
                {{ left: "sol", right: "sağ", top: "üst", bottom: "alt" }[k]}{" "}
                <input
                  type="number"
                  min={0}
                  max={1024}
                  step={8}
                  value={op[k]}
                  onChange={(e) => setOp({ ...op, [k]: Number(e.target.value) })}
                />
              </label>
            ))}
            <span className="muted">piksel cinsinden tuval genişletme</span>
          </div>
        )}

        <div className="dl-row">
          {mode !== "outpaint" && (
            <select value={size} onChange={(e) => setSize(e.target.value)}>
              <option>1024x1024</option>
              <option>896x1152</option>
              <option>1152x896</option>
              <option>768x1344</option>
              <option>1344x768</option>
              <option value="custom">özel…</option>
            </select>
          )}
          {mode !== "outpaint" && size === "custom" && (
            <>
              <label className="gen-num">
                G{" "}
                <input
                  type="number"
                  min={512}
                  max={2048}
                  step={8}
                  value={customW}
                  onChange={(e) => setCustomW(Number(e.target.value))}
                />
              </label>
              <label className="gen-num">
                Y{" "}
                <input
                  type="number"
                  min={512}
                  max={2048}
                  step={8}
                  value={customH}
                  onChange={(e) => setCustomH(Number(e.target.value))}
                />
              </label>
            </>
          )}
          <select value={sampler} onChange={(e) => setSampler(e.target.value)}>
            {SAMPLERS.map(([v, label]) => (
              <option key={v} value={v}>
                {label}
              </option>
            ))}
          </select>
          <label className="gen-num">
            adım{" "}
            <input
              type="number"
              min={1}
              max={100}
              value={steps}
              onChange={(e) => setSteps(Number(e.target.value))}
            />
          </label>
          <label className="gen-num">
            CFG{" "}
            <input
              type="number"
              min={0}
              max={20}
              step={0.5}
              value={cfg}
              onChange={(e) => setCfg(Number(e.target.value))}
            />
          </label>
          <label className="gen-num">
            seed{" "}
            <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
          </label>
          <label className="gen-num">
            batch{" "}
            <input
              type="number"
              min={1}
              max={16}
              value={batch}
              onChange={(e) => setBatch(Number(e.target.value))}
            />
          </label>
          <button type="submit" disabled={!connected || !modelPath || !!running || busy}>
            {running ? "Üretiliyor…" : batch > 1 ? `Üret ×${batch}` : "Üret"}
          </button>
        </div>

        <details className="adv">
          <summary>gelişmiş — VRAM · ControlNet · IP-Adapter</summary>
          <div className="dl-row">
            <label className="gen-num">
              VRAM{" "}
              <select value={vram} onChange={(e) => setVram(e.target.value)}>
                <option value="max">tam GPU</option>
                <option value="balanced">dengeli (model offload)</option>
                <option value="low">düşük VRAM (sequential)</option>
              </select>
            </label>
            {vram !== "max" && <span className="muted">offload modunda yalnız metin→görsel çalışır</span>}
          </div>
          {mode === "txt2img" && (
            <>
              <div className="dl-row">
                <select value={cnPick} onChange={(e) => setCnPick(e.target.value)}>
                  <option value="">ControlNet yok</option>
                  {cnModels.map((m) => (
                    <option key={m.id} value={m.path}>
                      {m.name}
                    </option>
                  ))}
                  <option value="__custom">elle gir…</option>
                </select>
                {cnPick === "__custom" && (
                  <input
                    type="text"
                    placeholder="model yolu ya da HF repo (ör. diffusers/controlnet-canny-sdxl-1.0)"
                    value={cnModel}
                    onChange={(e) => setCnModel(e.target.value)}
                  />
                )}
                {cnPick !== "" && (
                  <>
                    <select value={cnPrep} onChange={(e) => setCnPrep(e.target.value)}>
                      <option value="canny">canny çıkar</option>
                      <option value="none">hazır koşul görseli</option>
                    </select>
                    <label className="gen-num">
                      etki{" "}
                      <input
                        type="number"
                        min={0}
                        max={2}
                        step={0.05}
                        value={cnScale}
                        onChange={(e) => setCnScale(Number(e.target.value))}
                      />
                    </label>
                    <label className="file-pick">
                      {cnImage ? cnImage.name : "koşul görseli…"}
                      <input
                        type="file"
                        accept="image/png,image/jpeg,image/webp"
                        onChange={async (e) => {
                          const f = e.target.files?.[0];
                          if (!f) return;
                          try {
                            const up = await uploadImageFile(f);
                            setCnImage({ path: up.path, name: f.name });
                          } catch (err) {
                            setFormError(err instanceof Error ? err.message : String(err));
                          }
                        }}
                      />
                    </label>
                  </>
                )}
              </div>
              <div className="dl-row">
                <label className="file-pick">
                  {ipImage ? ipImage.name : "IP-Adapter referans görseli…"}
                  <input
                    type="file"
                    accept="image/png,image/jpeg,image/webp"
                    onChange={async (e) => {
                      const f = e.target.files?.[0];
                      if (!f) return;
                      try {
                        const up = await uploadImageFile(f);
                        setIpImage({ path: up.path, name: f.name });
                      } catch (err) {
                        setFormError(err instanceof Error ? err.message : String(err));
                      }
                    }}
                  />
                </label>
                {ipImage && (
                  <>
                    <label className="gen-num">
                      etki{" "}
                      <input
                        type="number"
                        min={0}
                        max={2}
                        step={0.05}
                        value={ipScale}
                        onChange={(e) => setIpScale(Number(e.target.value))}
                      />
                    </label>
                    <button type="button" onClick={() => setIpImage(null)}>
                      kaldır
                    </button>
                  </>
                )}
              </div>
            </>
          )}
        </details>

        {loraModels.length > 0 && (
          <div className="dl-row">
            <select
              value=""
              onChange={(e) => {
                const m = loraModels.find((x) => x.path === e.target.value);
                if (m && !loras.some((l) => l.path === m.path)) {
                  setLoras([...loras, { path: m.path, name: m.name, weight: 1.0 }]);
                }
              }}
            >
              <option value="" disabled>
                LoRA ekle…
              </option>
              {loraModels.map((m) => (
                <option key={m.id} value={m.path}>
                  {m.name}
                </option>
              ))}
            </select>
            {loras.map((l, i) => (
              <span key={l.path} className="lora-chip">
                {l.name}
                <input
                  type="number"
                  min={0}
                  max={2}
                  step={0.05}
                  value={l.weight}
                  onChange={(e) => {
                    const next = [...loras];
                    next[i] = { ...l, weight: Number(e.target.value) };
                    setLoras(next);
                  }}
                />
                <button type="button" onClick={() => setLoras(loras.filter((x) => x.path !== l.path))}>
                  ✕
                </button>
              </span>
            ))}
          </div>
        )}
        {formError && <p className="err-text">{formError}</p>}
      </form>

      {running && (
        <div className="dl-item">
          <div className="dl-head">
            <span className="dl-name">
              {running.status === "queued" ? "kuyrukta" : progress[running.id]?.message ?? "başlıyor…"}
            </span>
          </div>
          <div className="dl-bar">
            <div
              className="dl-fill"
              style={{ width: `${((progress[running.id]?.value ?? 0) * 100).toFixed(0)}%` }}
            />
          </div>
          {runningPreview && <img className="live-preview" src={runningPreview} alt="" />}
        </div>
      )}

      <div className="gallery">
        {assets.map((a) => {
          const url = assetUrl(a);
          const promptText = String(a.meta.params?.prompt ?? "");
          return (
            <figure key={a.id} className="gallery-item">
              <a href={url} target="_blank" rel="noreferrer" title={promptText}>
                <img src={url} alt="" loading="lazy" />
              </a>
              <div className="gallery-actions">
                <button
                  className="regen"
                  title="Aynı parametrelerle yeniden üret"
                  onClick={() => regenerateAsset(a.id).then(refreshJobs).catch(() => undefined)}
                  disabled={!!running}
                >
                  ↻ yeniden üret
                </button>
                <button
                  className="regen"
                  title="Bu görseli img2img/inpaint kaynağı yap"
                  onClick={() => {
                    adoptInit({ path: a.path, url, assetId: a.id });
                    if (mode === "txt2img") setMode("img2img");
                  }}
                >
                  ⤴ kaynak al
                </button>
                <button
                  className="regen"
                  title="Bu görselden 3D mesh üret (3D panelinde açılır)"
                  onClick={() => sendToMesh({ path: a.path, url, assetId: a.id })}
                >
                  ➜ 3D'ye gönder
                </button>
              </div>
            </figure>
          );
        })}
      </div>
      {assets.length === 0 && !running && (
        <p className="muted">Henüz üretim yok. Bir checkpoint seçip prompt yazın.</p>
      )}
    </div>
  );
}
