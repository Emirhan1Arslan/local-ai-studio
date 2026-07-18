import { useEffect, useRef, useState } from "react";
import {
  createJob,
  createJobBatch,
  fetchAssets,
  fetchJobs,
  fetchModels,
  regenerateAsset,
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

function assetUrl(a: AssetInfo): string {
  return `/outputs/${a.path.split(/[\\/]/).pop()}`;
}

export default function GeneratePanel() {
  const { events, connected } = useStudio();
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
  const [imageJobs, setImageJobs] = useState<JobInfo[]>([]);
  const [assets, setAssets] = useState<AssetInfo[]>([]);
  const [progress, setProgress] = useState<Record<string, { value: number; message: string }>>({});
  const [formError, setFormError] = useState("");
  const lastEvent = useRef(0);

  const refreshJobs = () => {
    fetchJobs().then((all) => setImageJobs(all.filter((j) => j.kind === "image"))).catch(() => undefined);
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
      })
      .catch(() => setModels([]));
    refreshJobs();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [connected]);

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
    if (needsRefresh) refreshJobs();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [events]);

  const running = imageJobs.find((j) => j.status === "running" || j.status === "queued");

  const generate = async (ev: React.FormEvent) => {
    ev.preventDefault();
    setFormError("");
    // 8'in katına yuvarla (VAE zorunluluğu); 512–2048 aralığına sıkıştır
    const snap = (v: number) => Math.min(2048, Math.max(512, Math.round(v / 8) * 8));
    const [w, h] =
      size === "custom" ? [snap(customW), snap(customH)] : size.split("x").map(Number);
    const params = {
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
    try {
      if (batch > 1) await createJobBatch("image", params, batch);
      else await createJob("image", params);
      refreshJobs();
    } catch (e) {
      setFormError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div className="panel-body">
      <form className="dl-form" onSubmit={generate}>
        <select value={modelPath} onChange={(e) => setModelPath(e.target.value)} required>
          <option value="" disabled>
            {models.length === 0 ? "checkpoint bulunamadı — İndirmeler panelinden indirin" : "model seç"}
          </option>
          {models.map((m) => (
            <option key={m.id} value={m.path}>
              {m.name}
            </option>
          ))}
        </select>
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
        <div className="dl-row">
          <select value={size} onChange={(e) => setSize(e.target.value)}>
            <option>1024x1024</option>
            <option>896x1152</option>
            <option>1152x896</option>
            <option>768x1344</option>
            <option>1344x768</option>
            <option value="custom">özel…</option>
          </select>
          {size === "custom" && (
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
          <button type="submit" disabled={!connected || !modelPath || !!running}>
            {running ? "Üretiliyor…" : batch > 1 ? `Üret ×${batch}` : "Üret"}
          </button>
        </div>
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
            <span className="dl-name">{running.status === "queued" ? "kuyrukta" : progress[running.id]?.message ?? "başlıyor…"}</span>
          </div>
          <div className="dl-bar">
            <div
              className="dl-fill"
              style={{ width: `${((progress[running.id]?.value ?? 0) * 100).toFixed(0)}%` }}
            />
          </div>
        </div>
      )}

      <div className="gallery">
        {assets.map((a) => {
          const url = assetUrl(a);
          const prompt = String(a.meta.params?.prompt ?? "");
          return (
            <figure key={a.id} className="gallery-item">
              <a href={url} target="_blank" rel="noreferrer" title={prompt}>
                <img src={url} alt="" loading="lazy" />
              </a>
              <button
                className="regen"
                title="Aynı parametrelerle yeniden üret"
                onClick={() => regenerateAsset(a.id).then(refreshJobs).catch(() => undefined)}
                disabled={!!running}
              >
                ↻ yeniden üret
              </button>
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
