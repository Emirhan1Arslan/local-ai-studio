import { useEffect, useRef, useState } from "react";
import {
  apiUrl,
  createJob,
  fetchAssets,
  fetchJobs,
  fetchModels,
  uploadImageFile,
  uploadVideoFile,
  type AssetInfo,
  type JobInfo,
  type ModelInfo,
} from "../api";
import { useStudio } from "../state";

type Mode = "generate" | "edit";
type GenMode = "t2v" | "i2v";
type EditOp = "trim" | "concat" | "speed" | "add_audio" | "convert" | "upscale" | "interpolate";
type Family = "ltx" | "wan" | "cogvideox" | "hunyuan" | "mochi";

// plan.md Faz 8 adım 1'in kademeli listesi: LTX-Video → Wan 2.x → CogVideoX →
// Hunyuan Video → Mochi (Pyramid Flow/SkyReels diffusers'ın yerleşik pipeline'ları
// arasında yok, bu dilimde uygulanmadı). Varsayılanlar backend'deki video.py'nin
// _FAMILIES tablosuyla birebir eşleşir.
const FAMILIES: [Family, string][] = [
  ["ltx", "LTX-Video (en hızlı, düşük VRAM)"],
  ["wan", "Wan 2.1"],
  ["cogvideox", "CogVideoX"],
  ["hunyuan", "Hunyuan Video (ağır)"],
  ["mochi", "Mochi (en ağır)"],
];

const FAMILY_DEFAULTS: Record<
  Family,
  { width: number; height: number; numFrames: number; steps: number; cfg: number; i2v: boolean }
> = {
  ltx: { width: 704, height: 480, numFrames: 121, steps: 30, cfg: 3.0, i2v: true },
  wan: { width: 832, height: 480, numFrames: 81, steps: 50, cfg: 5.0, i2v: true },
  cogvideox: { width: 720, height: 480, numFrames: 49, steps: 50, cfg: 6.0, i2v: true },
  hunyuan: { width: 960, height: 544, numFrames: 65, steps: 30, cfg: 6.0, i2v: false },
  mochi: { width: 848, height: 480, numFrames: 19, steps: 64, cfg: 4.5, i2v: false },
};

function outUrl(path: string): string {
  return apiUrl(`/outputs/${path.split(/[\\/]/).pop()}`);
}

export default function VideoPanel() {
  const { events, connected } = useStudio();
  const [mode, setMode] = useState<Mode>("generate");

  // Üretim (kademeli aile listesi)
  const [family, setFamily] = useState<Family>("ltx");
  const [genMode, setGenMode] = useState<GenMode>("t2v");
  const [videoModels, setVideoModels] = useState<ModelInfo[]>([]);
  const [modelPath, setModelPath] = useState("");
  const [prompt, setPrompt] = useState("");
  const [negative, setNegative] = useState("");
  const [initImage, setInitImage] = useState<{ path: string; url: string } | null>(null);
  const [width, setWidth] = useState(704);
  const [height, setHeight] = useState(480);
  const [numFrames, setNumFrames] = useState(121);
  const [fps, setFps] = useState(25);
  const [steps, setSteps] = useState(30);
  const [cfg, setCfg] = useState(3.0);
  const [seed, setSeed] = useState(-1);
  const [vram, setVram] = useState("max");

  // Düzenleme (ffmpeg)
  const [editOp, setEditOp] = useState<EditOp>("trim");
  const [srcVideo, setSrcVideo] = useState<{ path: string; name: string; previewUrl: string } | null>(
    null,
  );
  const [srcDuration, setSrcDuration] = useState(0);
  const [concatVideos, setConcatVideos] = useState<{ path: string; name: string }[]>([]);
  const [trimStart, setTrimStart] = useState(0);
  const [trimEnd, setTrimEnd] = useState<number | "">("");
  const [speedFactor, setSpeedFactor] = useState(1.5);
  const [audioFile, setAudioFile] = useState<{ path: string; name: string } | null>(null);
  const [audioMode, setAudioMode] = useState<"replace" | "mix">("replace");
  const [convertFormat, setConvertFormat] = useState("mp4");
  const [upscaleModels, setUpscaleModels] = useState<ModelInfo[]>([]);
  const [upscaleModelPath, setUpscaleModelPath] = useState("");
  const [rifeModels, setRifeModels] = useState<ModelInfo[]>([]);
  const [rifeModelPath, setRifeModelPath] = useState("");
  const [interpMulti, setInterpMulti] = useState(2);
  const [quantization, setQuantization] = useState("none");

  const [jobs, setJobs] = useState<JobInfo[]>([]);
  const [results, setResults] = useState<AssetInfo[]>([]);
  const [progress, setProgress] = useState<
    Record<string, { value: number; message: string; preview?: string }>
  >({});
  const [formError, setFormError] = useState("");
  const lastEvent = useRef(0);
  const scrubRef = useRef<HTMLVideoElement>(null);

  const refresh = () => {
    fetchJobs()
      .then((all) => setJobs(all.filter((j) => j.kind === "video" || j.kind === "videotool")))
      .catch(() => undefined);
    // "üret" (kind=video) ve "düzenle" (kind=videotool) tek galeride birleşir.
    Promise.all([fetchAssets("video", 24), fetchAssets("videotool", 24)])
      .then(([a, b]) =>
        setResults(
          [...a, ...b]
            .sort((x, y) => y.created_at.localeCompare(x.created_at))
            .slice(0, 24),
        ),
      )
      .catch(() => undefined);
  };

  useEffect(() => {
    if (!connected) return;
    refresh();
    fetchModels("video").then(setVideoModels).catch(() => setVideoModels([]));
    fetchModels("upscale").then(setUpscaleModels).catch(() => setUpscaleModels([]));
    fetchModels("video")
      .then((all) => setRifeModels(all.filter((m) => m.name.toLowerCase().includes("rife"))))
      .catch(() => setRifeModels([]));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [connected]);

  // Model Browser'daki her video modeli ayrı bir aileye ait (LTX-Video, CogVideoX-2b,
  // Wan2.1-T2V-1.3B-Diffusers, ...) — isimden basit bir ipucuyla mevcut aileye filtrelenir.
  const familyModels = videoModels.filter((m) => m.name.toLowerCase().includes(family));

  const selectFamily = (f: Family) => {
    setFamily(f);
    setModelPath("");
    const d = FAMILY_DEFAULTS[f];
    setWidth(d.width);
    setHeight(d.height);
    setNumFrames(d.numFrames);
    setSteps(d.steps);
    setCfg(d.cfg);
    if (!d.i2v) setGenMode("t2v");
  };

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
    if (needsRefresh) refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [events]);

  const running = jobs.find((j) => j.status === "running" || j.status === "queued");
  const runningPreview = running ? progress[running.id] : undefined;

  const pickInitImage = async (file: File) => {
    setFormError("");
    try {
      const up = await uploadImageFile(file);
      setInitImage({ path: up.path, url: URL.createObjectURL(file) });
    } catch (e) {
      setFormError(e instanceof Error ? e.message : String(e));
    }
  };

  const pickSrcVideo = async (file: File) => {
    setFormError("");
    // Tarayıcının kendi <video> öğesiyle kare-kare önizleme için yerel blob URL —
    // yükleme bitmeden de anında scrub edilebilir (ayrı bir sunucu isteği gerekmez).
    const previewUrl = URL.createObjectURL(file);
    setSrcVideo({ path: "", name: file.name, previewUrl });
    setTrimStart(0);
    setTrimEnd("");
    try {
      const up = await uploadVideoFile(file);
      setSrcVideo({ path: up.path, name: file.name, previewUrl });
    } catch (e) {
      setFormError(e instanceof Error ? e.message : String(e));
    }
  };

  const pickConcatVideo = async (file: File) => {
    setFormError("");
    try {
      const up = await uploadVideoFile(file);
      setConcatVideos((v) => [...v, { path: up.path, name: file.name }]);
    } catch (e) {
      setFormError(e instanceof Error ? e.message : String(e));
    }
  };

  const pickAudio = async (file: File) => {
    setFormError("");
    try {
      const up = await uploadVideoFile(file);
      setAudioFile({ path: up.path, name: file.name });
    } catch (e) {
      setFormError(e instanceof Error ? e.message : String(e));
    }
  };

  const run = async (ev: React.FormEvent) => {
    ev.preventDefault();
    setFormError("");

    if (mode === "generate") {
      if (!prompt.trim()) {
        setFormError("prompt yazın");
        return;
      }
      if (genMode === "i2v" && !initImage) {
        setFormError("i2v için bir kaynak görsel yükleyin");
        return;
      }
      const params: Record<string, unknown> = {
        family,
        mode: genMode,
        prompt: prompt.trim(),
        negative_prompt: negative.trim(),
        width,
        height,
        num_frames: numFrames,
        fps,
        steps,
        cfg,
        seed,
      };
      if (modelPath) params.model_path = modelPath;
      if (vram !== "max") params.vram_mode = vram;
      if (quantization !== "none") params.quantization = quantization;
      if (genMode === "i2v" && initImage) params.init_image = initImage.path;
      try {
        await createJob("video", params);
        refresh();
      } catch (e) {
        setFormError(e instanceof Error ? e.message : String(e));
      }
      return;
    }

    // düzenleme
    const params: Record<string, unknown> = { op: editOp };
    if (editOp === "concat") {
      if (concatVideos.length < 2) {
        setFormError("concat için en az 2 video yükleyin");
        return;
      }
      params.video_paths = concatVideos.map((v) => v.path);
    } else if (editOp === "add_audio") {
      if (!srcVideo?.path || !audioFile) {
        setFormError("hem video hem ses dosyası yükleyin (yükleme bitene kadar bekleyin)");
        return;
      }
      params.video_path = srcVideo.path;
      params.audio_path = audioFile.path;
      params.mode = audioMode;
    } else {
      if (!srcVideo?.path) {
        setFormError("bir video dosyası yükleyin (yükleme bitene kadar bekleyin)");
        return;
      }
      params.video_path = srcVideo.path;
      if (editOp === "trim") {
        params.start = trimStart;
        if (trimEnd !== "") params.end = trimEnd;
      } else if (editOp === "speed") {
        params.factor = speedFactor;
      } else if (editOp === "convert") {
        params.format = convertFormat;
      } else if (editOp === "upscale") {
        if (!upscaleModelPath) {
          setFormError("upscale için Model Browser'dan bir ESRGAN modeli seçin");
          return;
        }
        params.model_path = upscaleModelPath;
      } else if (editOp === "interpolate") {
        if (!rifeModelPath) {
          setFormError("kare interpolasyonu için RIFE modelini seçin (models/video/RIFE-4.25)");
          return;
        }
        params.model_path = rifeModelPath;
        params.multi = interpMulti;
      }
    }
    try {
      await createJob("videotool", params);
      refresh();
    } catch (e) {
      setFormError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div className="panel-body">
      <form className="dl-form" onSubmit={run}>
        <div className="mode-tabs">
          <button type="button" className={mode === "generate" ? "active" : ""} onClick={() => setMode("generate")}>
            üret
          </button>
          <button type="button" className={mode === "edit" ? "active" : ""} onClick={() => setMode("edit")}>
            düzenle (ffmpeg)
          </button>
        </div>

        {mode === "generate" && (
          <>
            <div className="dl-row">
              <select value={family} onChange={(e) => selectFamily(e.target.value as Family)}>
                {FAMILIES.map(([f, label]) => (
                  <option key={f} value={f}>
                    {label}
                  </option>
                ))}
              </select>
            </div>
            <div className="mode-tabs">
              <button type="button" className={genMode === "t2v" ? "active" : ""} onClick={() => setGenMode("t2v")}>
                metinden video
              </button>
              {FAMILY_DEFAULTS[family].i2v && (
                <button type="button" className={genMode === "i2v" ? "active" : ""} onClick={() => setGenMode("i2v")}>
                  görselden video
                </button>
              )}
            </div>
            {genMode === "i2v" && (
              <div className="dl-row">
                <label className="file-pick">
                  kaynak görsel…
                  <input
                    type="file"
                    accept="image/png,image/jpeg,image/webp"
                    onChange={(e) => e.target.files?.[0] && pickInitImage(e.target.files[0])}
                  />
                </label>
                {initImage && <img className="init-thumb" src={initImage.url} alt="" />}
                {initImage && (
                  <button type="button" onClick={() => setInitImage(null)}>
                    kaldır
                  </button>
                )}
              </div>
            )}
            <textarea placeholder="prompt" value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={3} required />
            <textarea
              placeholder="negatif prompt (opsiyonel)"
              value={negative}
              onChange={(e) => setNegative(e.target.value)}
              rows={2}
            />
            <div className="dl-row">
              <select value={modelPath} onChange={(e) => setModelPath(e.target.value)}>
                <option value="">varsayılan (ilk kullanımda indirir)</option>
                {familyModels.map((m) => (
                  <option key={m.id} value={m.path}>
                    {m.name} — Model Browser
                  </option>
                ))}
              </select>
            </div>
            <div className="dl-row">
              <label className="gen-num">
                G{" "}
                <input type="number" min={256} max={1280} step={32} value={width} onChange={(e) => setWidth(Number(e.target.value))} />
              </label>
              <label className="gen-num">
                Y{" "}
                <input type="number" min={256} max={1280} step={32} value={height} onChange={(e) => setHeight(Number(e.target.value))} />
              </label>
              <label className="gen-num">
                kare{" "}
                <input
                  type="number"
                  min={9}
                  max={257}
                  step={8}
                  value={numFrames}
                  onChange={(e) => setNumFrames(Number(e.target.value))}
                />
              </label>
              <label className="gen-num">
                fps{" "}
                <input type="number" min={8} max={60} value={fps} onChange={(e) => setFps(Number(e.target.value))} />
              </label>
              <span className="muted">~{(numFrames / fps).toFixed(1)}sn klip</span>
            </div>
            <div className="dl-row">
              <label className="gen-num">
                adım{" "}
                <input type="number" min={1} max={100} value={steps} onChange={(e) => setSteps(Number(e.target.value))} />
              </label>
              <label className="gen-num">
                CFG{" "}
                <input type="number" min={0} max={20} step={0.5} value={cfg} onChange={(e) => setCfg(Number(e.target.value))} />
              </label>
              <label className="gen-num">
                seed{" "}
                <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
              </label>
              <label className="gen-num">
                VRAM{" "}
                <select value={vram} onChange={(e) => setVram(e.target.value)}>
                  <option value="max">tam GPU</option>
                  <option value="balanced">dengeli (model offload)</option>
                  <option value="low">düşük VRAM (sequential)</option>
                </select>
              </label>
              <label className="gen-num">
                nicemleme{" "}
                <select
                  value={quantization}
                  onChange={(e) => setQuantization(e.target.value)}
                  disabled={vram === "low"}
                >
                  <option value="none">yok (bf16)</option>
                  <option value="fp8">fp8 (torchao, ~%35 daha az VRAM)</option>
                </select>
              </label>
            </div>
          </>
        )}

        {mode === "edit" && (
          <>
            <div className="dl-row">
              <select value={editOp} onChange={(e) => setEditOp(e.target.value as EditOp)}>
                <option value="trim">kes</option>
                <option value="concat">birleştir</option>
                <option value="speed">hız</option>
                <option value="add_audio">ses ekle</option>
                <option value="convert">format dönüştür</option>
                <option value="upscale">büyüt</option>
                <option value="interpolate">kare hızını artır (RIFE)</option>
              </select>
            </div>

            {editOp === "concat" ? (
              <div className="dl-row">
                <label className="file-pick">
                  video ekle…
                  <input
                    type="file"
                    accept="video/*"
                    onChange={(e) => e.target.files?.[0] && pickConcatVideo(e.target.files[0])}
                  />
                </label>
                {concatVideos.map((v, i) => (
                  <span className="muted" key={i}>
                    {i + 1}. {v.name}
                  </span>
                ))}
                {concatVideos.length > 0 && (
                  <button type="button" onClick={() => setConcatVideos([])}>
                    temizle
                  </button>
                )}
              </div>
            ) : (
              <div className="dl-row">
                <label className="file-pick">
                  kaynak video…
                  <input
                    type="file"
                    accept="video/*"
                    onChange={(e) => e.target.files?.[0] && pickSrcVideo(e.target.files[0])}
                  />
                </label>
                {srcVideo && <span className="muted">{srcVideo.name}</span>}
                {srcVideo && (
                  <button
                    type="button"
                    onClick={() => {
                      URL.revokeObjectURL(srcVideo.previewUrl);
                      setSrcVideo(null);
                      setSrcDuration(0);
                      setTrimStart(0);
                      setTrimEnd("");
                    }}
                  >
                    kaldır
                  </button>
                )}
              </div>
            )}

            {editOp === "trim" && (
              <>
                {srcVideo && (
                  <div className="mask-wrap">
                    <video
                      ref={scrubRef}
                      src={srcVideo.previewUrl}
                      className="video-scrub"
                      controls
                      onLoadedMetadata={(e) => {
                        const dur = e.currentTarget.duration || 0;
                        setSrcDuration(dur);
                        if (trimEnd === "") setTrimEnd(Number(dur.toFixed(1)));
                      }}
                    />
                    {srcDuration > 0 && (
                      <div className="dl-row">
                        <label className="gen-num" style={{ flex: 1 }}>
                          başlangıç: {trimStart.toFixed(1)}sn
                          <input
                            type="range"
                            min={0}
                            max={srcDuration}
                            step={0.1}
                            value={trimStart}
                            onChange={(e) => {
                              const v = Math.min(Number(e.target.value), typeof trimEnd === "number" ? trimEnd : srcDuration);
                              setTrimStart(v);
                              if (scrubRef.current) scrubRef.current.currentTime = v;
                            }}
                          />
                        </label>
                        <label className="gen-num" style={{ flex: 1 }}>
                          bitiş: {trimEnd === "" ? srcDuration.toFixed(1) : trimEnd.toFixed(1)}sn
                          <input
                            type="range"
                            min={0}
                            max={srcDuration}
                            step={0.1}
                            value={trimEnd === "" ? srcDuration : trimEnd}
                            onChange={(e) => {
                              const v = Math.max(Number(e.target.value), trimStart);
                              setTrimEnd(v);
                              if (scrubRef.current) scrubRef.current.currentTime = v;
                            }}
                          />
                        </label>
                      </div>
                    )}
                  </div>
                )}
                <div className="dl-row">
                  <label className="gen-num">
                    başlangıç (sn){" "}
                    <input
                      type="number"
                      min={0}
                      step={0.1}
                      value={trimStart}
                      onChange={(e) => setTrimStart(Number(e.target.value))}
                    />
                  </label>
                  <label className="gen-num">
                    bitiş (sn, boş = sona kadar){" "}
                    <input
                      type="number"
                      min={0}
                      step={0.1}
                      value={trimEnd}
                      onChange={(e) => setTrimEnd(e.target.value === "" ? "" : Number(e.target.value))}
                    />
                  </label>
                </div>
              </>
            )}

            {editOp === "speed" && (
              <div className="dl-row">
                <label className="gen-num">
                  hız çarpanı (0.5-2.0){" "}
                  <input
                    type="number"
                    min={0.5}
                    max={2}
                    step={0.1}
                    value={speedFactor}
                    onChange={(e) => setSpeedFactor(Number(e.target.value))}
                  />
                </label>
              </div>
            )}

            {editOp === "add_audio" && (
              <div className="dl-row">
                <label className="file-pick">
                  ses dosyası…
                  <input
                    type="file"
                    accept="audio/*"
                    onChange={(e) => e.target.files?.[0] && pickAudio(e.target.files[0])}
                  />
                </label>
                {audioFile && <span className="muted">{audioFile.name}</span>}
                <select value={audioMode} onChange={(e) => setAudioMode(e.target.value as "replace" | "mix")}>
                  <option value="replace">sesi değiştir</option>
                  <option value="mix">mevcut sesle karıştır</option>
                </select>
              </div>
            )}

            {editOp === "convert" && (
              <div className="dl-row">
                <select value={convertFormat} onChange={(e) => setConvertFormat(e.target.value)}>
                  <option value="mp4">MP4</option>
                  <option value="mov">MOV</option>
                  <option value="avi">AVI</option>
                  <option value="gif">GIF</option>
                </select>
              </div>
            )}

            {editOp === "upscale" && (
              <div className="dl-row">
                <select value={upscaleModelPath} onChange={(e) => setUpscaleModelPath(e.target.value)}>
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
                <span className="muted">her kare ayrı ayrı büyütülür — uzun videolarda yavaş olabilir</span>
              </div>
            )}

            {editOp === "interpolate" && (
              <div className="dl-row">
                <select value={rifeModelPath} onChange={(e) => setRifeModelPath(e.target.value)}>
                  <option value="" disabled>
                    {rifeModels.length === 0 ? "RIFE modeli bulunamadı (models/video/RIFE-4.25)" : "model seç"}
                  </option>
                  {rifeModels.map((m) => (
                    <option key={m.id} value={m.path}>
                      {m.name}
                    </option>
                  ))}
                </select>
                <select value={interpMulti} onChange={(e) => setInterpMulti(Number(e.target.value))}>
                  <option value={2}>2x kare hızı</option>
                  <option value={3}>3x</option>
                  <option value={4}>4x</option>
                  <option value={8}>8x</option>
                </select>
                <span className="muted">Practical-RIFE v4.25 (MIT) — ara kareler akış tahminiyle üretilir, ses korunur</span>
              </div>
            )}
          </>
        )}

        <div className="dl-row">
          <button type="submit" disabled={!connected || !!running}>
            {running ? "çalışıyor…" : mode === "generate" ? "üret" : "uygula"}
          </button>
        </div>
        {formError && <p className="err-text">{formError}</p>}
      </form>

      {running && (
        <div className="dl-item">
          <div className="dl-head">
            <span className="dl-name">
              {running.status === "queued" ? "kuyrukta" : runningPreview?.message ?? "başlıyor…"}
            </span>
          </div>
          <div className="dl-bar">
            <div className="dl-fill" style={{ width: `${((runningPreview?.value ?? 0) * 100).toFixed(0)}%` }} />
          </div>
          {runningPreview?.preview && (
            <img className="live-preview" src={runningPreview.preview} alt="ara kare önizlemesi" />
          )}
        </div>
      )}

      <div className="audio-results">
        {results.map((a) => (
          <div className="audio-result-card" key={a.id}>
            <video controls src={outUrl(a.path)} />
            <p className="muted">"{String(a.meta.params?.prompt ?? a.meta.params?.op ?? "").slice(0, 120)}"</p>
          </div>
        ))}
      </div>
      {results.length === 0 && !running && (
        <p className="muted">Henüz video yok. Bir prompt yazıp üretin ya da mevcut bir videoyu düzenleyin.</p>
      )}
    </div>
  );
}
