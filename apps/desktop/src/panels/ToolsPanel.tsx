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

type Op =
  | "bg_remove"
  | "upscale"
  | "caption"
  | "ocr"
  | "segment"
  | "crop"
  | "adjust"
  | "face_restore";
type SegMode = "point" | "box" | "text";

const OP_LABELS: Record<Op, string> = {
  bg_remove: "arka plan kaldır",
  upscale: "büyüt",
  caption: "başlıkla",
  ocr: "metin oku",
  segment: "segmentle",
  crop: "kırp",
  adjust: "ayarla",
  face_restore: "yüz onar",
};

const IMAGE_OPS = new Set<string>([
  "bg_remove",
  "upscale",
  "segment",
  "crop",
  "adjust",
  "face_restore",
]);
const ADJUST_KEYS = ["brightness", "contrast", "saturation", "sharpness"] as const;
const ADJUST_LABELS: Record<(typeof ADJUST_KEYS)[number], string> = {
  brightness: "parlaklık",
  contrast: "kontrast",
  saturation: "doygunluk",
  sharpness: "keskinlik",
};

interface SourcePick {
  path: string;
  url: string;
  assetId?: string;
}

interface Box {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

function outUrl(path: string): string {
  return apiUrl(`/outputs/${path.split(/[\\/]/).pop()}`);
}

/** Bir asset'in tuvalde gösterilebilir bir görsel çıktısı var mı (üretim ya da görsel-çıktılı imgtool op'ları)? */
function isImageAsset(a: AssetInfo): boolean {
  if (a.kind === "image") return true;
  const op = a.meta.params?.op as string | undefined;
  return a.kind === "imgtool" && !!op && IMAGE_OPS.has(op);
}

/** Fare/dokunmatik ile tek bir dikdörtgen seçimi — crop ve segment(box) tarafından paylaşılır. */
function BoxPicker({
  src,
  box,
  onChange,
}: {
  src: string;
  box: Box | null;
  onChange: (b: Box) => void;
}) {
  const imgRef = useRef<HTMLImageElement>(null);
  const [nat, setNat] = useState({ w: 1, h: 1 });
  const dragStart = useRef<{ x: number; y: number } | null>(null);

  const toNatural = (clientX: number, clientY: number) => {
    const rect = imgRef.current!.getBoundingClientRect();
    const x = ((clientX - rect.left) / rect.width) * nat.w;
    const y = ((clientY - rect.top) / rect.height) * nat.h;
    return { x: Math.max(0, Math.min(nat.w, x)), y: Math.max(0, Math.min(nat.h, y)) };
  };

  return (
    <div className="box-picker">
      <img
        ref={imgRef}
        src={src}
        alt=""
        draggable={false}
        onLoad={(e) => setNat({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
        onPointerDown={(e) => {
          const p = toNatural(e.clientX, e.clientY);
          dragStart.current = p;
          onChange({ x1: p.x, y1: p.y, x2: p.x, y2: p.y });
        }}
        onPointerMove={(e) => {
          if (!dragStart.current) return;
          const p = toNatural(e.clientX, e.clientY);
          onChange({
            x1: Math.min(dragStart.current.x, p.x),
            y1: Math.min(dragStart.current.y, p.y),
            x2: Math.max(dragStart.current.x, p.x),
            y2: Math.max(dragStart.current.y, p.y),
          });
        }}
        onPointerUp={() => (dragStart.current = null)}
      />
      {box && nat.w > 1 && (
        <div
          className="box-overlay"
          style={{
            left: `${(box.x1 / nat.w) * 100}%`,
            top: `${(box.y1 / nat.h) * 100}%`,
            width: `${((box.x2 - box.x1) / nat.w) * 100}%`,
            height: `${((box.y2 - box.y1) / nat.h) * 100}%`,
          }}
        />
      )}
    </div>
  );
}

/** Model Browser'dan seç + "elle gir…" ile serbest HF repo id/yol — Model Browser'a
 * henüz indirilmemiş varsayılan modeller (BiRefNet, Florence-2, SAM, Grounding DINO)
 * için de çalışır; boş bırakılırsa adapter'ın kendi varsayılanı kullanılır. */
function ModelPick({
  models,
  pick,
  onPick,
  custom,
  onCustom,
  placeholder,
  emptyLabel,
}: {
  models: ModelInfo[];
  pick: string;
  onPick: (v: string) => void;
  custom: string;
  onCustom: (v: string) => void;
  placeholder: string;
  emptyLabel: string;
}) {
  return (
    <>
      <select value={pick} onChange={(e) => onPick(e.target.value)}>
        <option value="">{emptyLabel}</option>
        {models.map((m) => (
          <option key={m.id} value={m.path}>
            {m.name}
          </option>
        ))}
        <option value="__custom">elle gir…</option>
      </select>
      {pick === "__custom" && (
        <input
          type="text"
          placeholder={placeholder}
          value={custom}
          onChange={(e) => onCustom(e.target.value)}
        />
      )}
    </>
  );
}

/** Tıklayarak ön plan noktaları işaretleme — segment(point) için. */
function PointPicker({
  src,
  points,
  onChange,
}: {
  src: string;
  points: [number, number][];
  onChange: (p: [number, number][]) => void;
}) {
  const imgRef = useRef<HTMLImageElement>(null);
  const [nat, setNat] = useState({ w: 1, h: 1 });

  return (
    <div className="box-picker">
      <img
        ref={imgRef}
        src={src}
        alt=""
        draggable={false}
        onLoad={(e) => setNat({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
        onClick={(e) => {
          const rect = imgRef.current!.getBoundingClientRect();
          const x = ((e.clientX - rect.left) / rect.width) * nat.w;
          const y = ((e.clientY - rect.top) / rect.height) * nat.h;
          onChange([...points, [x, y]]);
        }}
      />
      {nat.w > 1 &&
        points.map(([x, y], i) => (
          <div
            key={i}
            className="point-dot"
            style={{ left: `${(x / nat.w) * 100}%`, top: `${(y / nat.h) * 100}%` }}
          />
        ))}
    </div>
  );
}

export default function ToolsPanel() {
  const { events, connected } = useStudio();
  const [op, setOp] = useState<Op>("bg_remove");
  const [source, setSource] = useState<SourcePick | null>(null);
  const [sourceCandidates, setSourceCandidates] = useState<AssetInfo[]>([]);
  const [upscaleModels, setUpscaleModels] = useState<ModelInfo[]>([]);
  const [upscaleModel, setUpscaleModel] = useState("");
  const [faceModels, setFaceModels] = useState<ModelInfo[]>([]);
  const [faceModel, setFaceModel] = useState("");
  const [faceWeight, setFaceWeight] = useState(0.5);
  const [bgModels, setBgModels] = useState<ModelInfo[]>([]);
  const [bgPick, setBgPick] = useState("");
  const [bgCustom, setBgCustom] = useState("");
  const [captionModels, setCaptionModels] = useState<ModelInfo[]>([]);
  const [captionPick, setCaptionPick] = useState("");
  const [captionCustom, setCaptionCustom] = useState("");
  const [captionTask, setCaptionTask] = useState("short");
  const [ocrLangs, setOcrLangs] = useState("en");
  const [segModels, setSegModels] = useState<ModelInfo[]>([]);
  const [samPick, setSamPick] = useState("");
  const [samCustom, setSamCustom] = useState("");
  const [dinoPick, setDinoPick] = useState("");
  const [dinoCustom, setDinoCustom] = useState("");
  const [segMode, setSegMode] = useState<SegMode>("text");
  const [segText, setSegText] = useState("");
  const [segBox, setSegBox] = useState<Box | null>(null);
  const [segPoints, setSegPoints] = useState<[number, number][]>([]);
  const [segCutout, setSegCutout] = useState(false);
  const [cropBox, setCropBox] = useState<Box | null>(null);
  const [adjustVals, setAdjustVals] = useState<Record<string, number>>({
    brightness: 1,
    contrast: 1,
    saturation: 1,
    sharpness: 1,
  });
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
    fetchModels("face_restoration").then(setFaceModels).catch(() => setFaceModels([]));
    fetchModels("background_removal").then(setBgModels).catch(() => setBgModels([]));
    fetchModels("captioning").then(setCaptionModels).catch(() => setCaptionModels([]));
    fetchModels("segmentation").then(setSegModels).catch(() => setSegModels([]));
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
      setSegBox(null);
      setSegPoints([]);
      setCropBox(null);
    } catch (e) {
      setFormError(e instanceof Error ? e.message : String(e));
    }
  };

  const useAsset = (a: AssetInfo) => {
    setSource({ path: a.path, url: outUrl(a.path), assetId: a.id });
    setSegBox(null);
    setSegPoints([]);
    setCropBox(null);
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
    const bgModel = bgPick === "__custom" ? bgCustom.trim() : bgPick;
    if (op === "bg_remove" && bgModel) params.model_path = bgModel;
    if (op === "upscale") {
      if (!upscaleModel) {
        setFormError("upscale için Model Browser'dan bir ESRGAN modeli seçin");
        return;
      }
      params.model_path = upscaleModel;
    }
    const captionModel = captionPick === "__custom" ? captionCustom.trim() : captionPick;
    if (op === "caption") {
      params.task = captionTask;
      if (captionModel) params.model_path = captionModel;
    }
    if (op === "ocr") {
      params.langs = ocrLangs.split(",").map((s) => s.trim()).filter(Boolean);
    }
    if (op === "segment") {
      params.mode = segMode;
      params.cutout = segCutout;
      const samModel = samPick === "__custom" ? samCustom.trim() : samPick;
      const dinoModel = dinoPick === "__custom" ? dinoCustom.trim() : dinoPick;
      if (samModel) params.sam_model = samModel;
      if (dinoModel) params.dino_model = dinoModel;
      if (segMode === "text") {
        if (!segText.trim()) {
          setFormError('metin girin (ör. "kırmızı araba")');
          return;
        }
        params.text_prompt = segText.trim();
      } else if (segMode === "box") {
        if (!segBox) {
          setFormError("bir kutu çizin");
          return;
        }
        params.box = [segBox.x1, segBox.y1, segBox.x2, segBox.y2];
      } else {
        if (segPoints.length === 0) {
          setFormError("en az bir nokta işaretleyin");
          return;
        }
        params.points = segPoints;
      }
    }
    if (op === "crop") {
      if (!cropBox) {
        setFormError("bir kırpma alanı çizin");
        return;
      }
      params.box = [cropBox.x1, cropBox.y1, cropBox.x2, cropBox.y2];
    }
    if (op === "adjust") {
      for (const [k, v] of Object.entries(adjustVals)) {
        if (v !== 1) params[k] = v;
      }
    }
    if (op === "face_restore") {
      if (!faceModel) {
        setFormError("face_restore için Model Browser'dan bir GFPGAN modeli seçin");
        return;
      }
      params.model_path = faceModel;
      params.weight = faceWeight;
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
          {source && !((op === "crop") || (op === "segment" && segMode !== "text")) && (
            <img className="init-thumb" src={source.url} alt="" />
          )}
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
            <ModelPick
              models={bgModels}
              pick={bgPick}
              onPick={setBgPick}
              custom={bgCustom}
              onCustom={setBgCustom}
              placeholder="HF repo id (ör. ZhengPeng7/BiRefNet)"
              emptyLabel="varsayılan: ZhengPeng7/BiRefNet"
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
            <ModelPick
              models={captionModels}
              pick={captionPick}
              onPick={setCaptionPick}
              custom={captionCustom}
              onCustom={setCaptionCustom}
              placeholder="HF repo id (ör. microsoft/Florence-2-base)"
              emptyLabel="varsayılan: microsoft/Florence-2-base"
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
            <span className="muted">çıktı: json + txt + hOCR</span>
          </div>
        )}

        {op === "segment" && (
          <>
            <div className="dl-row">
              <select value={segMode} onChange={(e) => setSegMode(e.target.value as SegMode)}>
                <option value="text">metinle (açık kelime dağarcığı)</option>
                <option value="box">kutu çizerek</option>
                <option value="point">nokta işaretleyerek</option>
              </select>
              <label className="gen-num">
                <input
                  type="checkbox"
                  checked={segCutout}
                  onChange={(e) => setSegCutout(e.target.checked)}
                />{" "}
                kesim (RGBA) de üret
              </label>
            </div>
            <div className="dl-row">
              <span className="muted">SAM</span>
              <ModelPick
                models={segModels}
                pick={samPick}
                onPick={setSamPick}
                custom={samCustom}
                onCustom={setSamCustom}
                placeholder="HF repo id (ör. facebook/sam-vit-base)"
                emptyLabel="varsayılan: facebook/sam-vit-base"
              />
            </div>
            {segMode === "text" && (
              <>
                <div className="dl-row">
                  <span className="muted">Grounding DINO</span>
                  <ModelPick
                    models={segModels}
                    pick={dinoPick}
                    onPick={setDinoPick}
                    custom={dinoCustom}
                    onCustom={setDinoCustom}
                    placeholder="HF repo id (ör. IDEA-Research/grounding-dino-tiny)"
                    emptyLabel="varsayılan: IDEA-Research/grounding-dino-tiny"
                  />
                </div>
                <div className="dl-row">
                  <input
                    type="text"
                    placeholder='ör. "kırmızı araba"'
                    value={segText}
                    onChange={(e) => setSegText(e.target.value)}
                  />
                </div>
              </>
            )}
            {segMode === "box" && source && (
              <BoxPicker src={source.url} box={segBox} onChange={setSegBox} />
            )}
            {segMode === "point" && source && (
              <>
                <PointPicker src={source.url} points={segPoints} onChange={setSegPoints} />
                <div className="dl-row">
                  <span className="muted">{segPoints.length} nokta işaretlendi</span>
                  <button type="button" onClick={() => setSegPoints([])}>
                    noktaları temizle
                  </button>
                </div>
              </>
            )}
          </>
        )}
        {op === "crop" && source && <BoxPicker src={source.url} box={cropBox} onChange={setCropBox} />}
        {op === "face_restore" && (
          <div className="dl-row">
            <select value={faceModel} onChange={(e) => setFaceModel(e.target.value)}>
              <option value="" disabled>
                {faceModels.length === 0
                  ? "GFPGAN modeli bulunamadı — İndirmeler panelinden indirin"
                  : "model seç"}
              </option>
              {faceModels.map((m) => (
                <option key={m.id} value={m.path}>
                  {m.name}
                </option>
              ))}
            </select>
            <label className="gen-num">
              güç{" "}
              <input
                type="number"
                min={0}
                max={1}
                step={0.05}
                value={faceWeight}
                onChange={(e) => setFaceWeight(Number(e.target.value))}
              />
            </label>
          </div>
        )}
        {op === "adjust" && (
          <div className="dl-row">
            {ADJUST_KEYS.map((k) => (
              <label className="gen-num" key={k}>
                {ADJUST_LABELS[k]}{" "}
                <input
                  type="number"
                  min={0}
                  max={3}
                  step={0.1}
                  value={adjustVals[k]}
                  onChange={(e) => setAdjustVals({ ...adjustVals, [k]: Number(e.target.value) })}
                />
              </label>
            ))}
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
