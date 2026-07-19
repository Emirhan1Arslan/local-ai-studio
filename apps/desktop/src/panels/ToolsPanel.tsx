import { useEffect, useRef, useState } from "react";
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

type Op = "bg_remove" | "upscale" | "caption" | "ocr";

const OP_LABELS: Record<Op, string> = {
  bg_remove: "arka plan kaldır",
  upscale: "büyüt",
  caption: "başlıkla",
  ocr: "metin oku",
};

const IMAGE_OPS = new Set<string>(["bg_remove", "upscale"]);

interface SourcePick {
  path: string;
  url: string;
  assetId?: string;
}

function outUrl(path: string): string {
  return apiUrl(`/outputs/${path.split(/[\\/]/).pop()}`);
}

/** Bir asset'in tuvalde gösterilebilir bir görsel çıktısı var mı (üretim ya da bg_remove/upscale)? */
function isImageAsset(a: AssetInfo): boolean {
  if (a.kind === "image") return true;
  const op = a.meta.params?.op as string | undefined;
  return a.kind === "imgtool" && !!op && IMAGE_OPS.has(op);
}

export default function ToolsPanel() {
  const { events, connected } = useStudio();
  const [op, setOp] = useState<Op>("bg_remove");
  const [source, setSource] = useState<SourcePick | null>(null);
  const [sourceCandidates, setSourceCandidates] = useState<AssetInfo[]>([]);
  const [upscaleModels, setUpscaleModels] = useState<ModelInfo[]>([]);
  const [upscaleModel, setUpscaleModel] = useState("");
  const [bgModel, setBgModel] = useState("");
  const [captionModel, setCaptionModel] = useState("");
  const [captionTask, setCaptionTask] = useState("short");
  const [ocrLangs, setOcrLangs] = useState("en");
  const [jobs, setJobs] = useState<JobInfo[]>([]);
  const [results, setResults] = useState<AssetInfo[]>([]);
  const [progress, setProgress] = useState<Record<string, { value: number; message: string }>>({});
  const [formError, setFormError] = useState("");
  const lastEvent = useRef(0);

  const refresh = () => {
    fetchJobs().then((all) => setJobs(all.filter((j) => j.kind === "imgtool"))).catch(() => undefined);
    fetchAssets("imgtool", 24).then(setResults).catch(() => undefined);
    fetchAssets(undefined, 40)
      .then((all) => setSourceCandidates(all.filter(isImageAsset)))
      .catch(() => undefined);
  };

  useEffect(() => {
    if (!connected) return;
    fetchModels("upscale").then(setUpscaleModels).catch(() => setUpscaleModels([]));
    refresh();
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
    if (needsRefresh) refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [events]);

  const running = jobs.find((j) => j.status === "running" || j.status === "queued");

  const pickFile = async (file: File) => {
    setFormError("");
    try {
      const up = await uploadImageFile(file);
      setSource({ path: up.path, url: URL.createObjectURL(file) });
    } catch (e) {
      setFormError(e instanceof Error ? e.message : String(e));
    }
  };

  const useAsset = (a: AssetInfo) => {
    setSource({ path: a.path, url: outUrl(a.path), assetId: a.id });
  };

  const run = async (ev: React.FormEvent) => {
    ev.preventDefault();
    setFormError("");
    if (!source) {
      setFormError("önce bir kaynak görsel seçin (yükleyin ya da aşağıdan seçin)");
      return;
    }
    const params: Record<string, unknown> = { op, image_path: source.path };
    if (source.assetId) params.parent_asset_id = source.assetId;
    if (op === "bg_remove" && bgModel) params.model_path = bgModel;
    if (op === "upscale") {
      if (!upscaleModel) {
        setFormError("upscale için Model Browser'dan bir ESRGAN modeli seçin");
        return;
      }
      params.model_path = upscaleModel;
    }
    if (op === "caption") {
      params.task = captionTask;
      if (captionModel) params.model_path = captionModel;
    }
    if (op === "ocr") {
      params.langs = ocrLangs.split(",").map((s) => s.trim()).filter(Boolean);
    }
    try {
      await createJob("imgtool", params);
      refresh();
    } catch (e) {
      setFormError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div className="panel-body">
      <form className="dl-form" onSubmit={run}>
        <div className="mode-tabs">
          {(Object.keys(OP_LABELS) as Op[]).map((o) => (
            <button key={o} type="button" className={op === o ? "active" : ""} onClick={() => setOp(o)}>
              {OP_LABELS[o]}
            </button>
          ))}
        </div>

        <div className="dl-row">
          <label className="file-pick">
            kaynak görsel yükle…
            <input
              type="file"
              accept="image/png,image/jpeg,image/webp"
              onChange={(e) => e.target.files?.[0] && pickFile(e.target.files[0])}
            />
          </label>
          {source && <img className="init-thumb" src={source.url} alt="" />}
          {source && (
            <button type="button" onClick={() => setSource(null)}>
              kaldır
            </button>
          )}
        </div>

        {sourceCandidates.length > 0 && (
          <div className="tool-source-strip">
            {sourceCandidates.map((a) => (
              <img
                key={a.id}
                src={outUrl(a.path)}
                alt=""
                className={source?.assetId === a.id ? "picked" : ""}
                onClick={() => useAsset(a)}
                title="kaynak olarak seç"
              />
            ))}
          </div>
        )}

        {op === "bg_remove" && (
          <div className="dl-row">
            <input
              type="text"
              placeholder="model (opsiyonel — varsayılan: ZhengPeng7/BiRefNet)"
              value={bgModel}
              onChange={(e) => setBgModel(e.target.value)}
            />
          </div>
        )}
        {op === "upscale" && (
          <div className="dl-row">
            <select value={upscaleModel} onChange={(e) => setUpscaleModel(e.target.value)}>
              <option value="" disabled>
                {upscaleModels.length === 0
                  ? "upscale modeli bulunamadı — İndirmeler panelinden indirin"
                  : "model seç"}
              </option>
              {upscaleModels.map((m) => (
                <option key={m.id} value={m.path}>
                  {m.name}
                </option>
              ))}
            </select>
          </div>
        )}
        {op === "caption" && (
          <div className="dl-row">
            <select value={captionTask} onChange={(e) => setCaptionTask(e.target.value)}>
              <option value="short">kısa başlık</option>
              <option value="detailed">detaylı açıklama</option>
              <option value="objects">nesne tespiti</option>
            </select>
            <input
              type="text"
              placeholder="model (opsiyonel — varsayılan: microsoft/Florence-2-base)"
              value={captionModel}
              onChange={(e) => setCaptionModel(e.target.value)}
            />
          </div>
        )}
        {op === "ocr" && (
          <div className="dl-row">
            <input
              type="text"
              placeholder="diller, virgülle (ör. en,tr)"
              value={ocrLangs}
              onChange={(e) => setOcrLangs(e.target.value)}
            />
          </div>
        )}

        <div className="dl-row">
          <button type="submit" disabled={!connected || !source || !!running}>
            {running ? "çalışıyor…" : "çalıştır"}
          </button>
        </div>
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
        </div>
      )}

      <div className="gallery">
        {results.map((a) => {
          const resultOp = String(a.meta.params?.op ?? "");
          const isImg = IMAGE_OPS.has(resultOp);
          const text = String(a.meta.result?.caption ?? a.meta.result?.text ?? "");
          return (
            <figure key={a.id} className="gallery-item">
              {isImg ? (
                <a href={outUrl(a.path)} target="_blank" rel="noreferrer">
                  <img src={outUrl(a.path)} alt="" loading="lazy" />
                </a>
              ) : (
                <div className="tool-text-card" title={text}>
                  <span className="tool-text-op">{OP_LABELS[resultOp as Op] ?? resultOp}</span>
                  <p>{text || "(boş)"}</p>
                </div>
              )}
              <div className="gallery-actions">
                {isImg && (
                  <button className="regen" onClick={() => useAsset(a)} disabled={!!running}>
                    ⤴ kaynak yap
                  </button>
                )}
              </div>
            </figure>
          );
        })}
      </div>
      {results.length === 0 && !running && (
        <p className="muted">Henüz sonuç yok. Bir görsel seçip araç çalıştırın.</p>
      )}
    </div>
  );
}
