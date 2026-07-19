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
type EditOp = "trim" | "concat" | "speed" | "add_audio" | "convert";

function outUrl(path: string): string {
  return apiUrl(`/outputs/${path.split(/[\\/]/).pop()}`);
}

export default function VideoPanel() {
  const { events, connected } = useStudio();
  const [mode, setMode] = useState<Mode>("generate");

  // Üretim (LTX-Video)
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
  const [srcVideo, setSrcVideo] = useState<{ path: string; name: string } | null>(null);
  const [concatVideos, setConcatVideos] = useState<{ path: string; name: string }[]>([]);
  const [trimStart, setTrimStart] = useState(0);
  const [trimEnd, setTrimEnd] = useState<number | "">("");
  const [speedFactor, setSpeedFactor] = useState(1.5);
  const [audioFile, setAudioFile] = useState<{ path: string; name: string } | null>(null);
  const [audioMode, setAudioMode] = useState<"replace" | "mix">("replace");
  const [convertFormat, setConvertFormat] = useState("mp4");

  const [jobs, setJobs] = useState<JobInfo[]>([]);
  const [results, setResults] = useState<AssetInfo[]>([]);
  const [progress, setProgress] = useState<Record<string, { value: number; message: string }>>(
    {},
  );
  const [formError, setFormError] = useState("");
  const lastEvent = useRef(0);

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
    try {
      const up = await uploadVideoFile(file);
      setSrcVideo({ path: up.path, name: file.name });
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
      if (!srcVideo || !audioFile) {
        setFormError("hem video hem ses dosyası yükleyin");
        return;
      }
      params.video_path = srcVideo.path;
      params.audio_path = audioFile.path;
      params.mode = audioMode;
    } else {
      if (!srcVideo) {
        setFormError("bir video dosyası yükleyin");
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
            üret (LTX-Video)
          </button>
          <button type="button" className={mode === "edit" ? "active" : ""} onClick={() => setMode("edit")}>
            düzenle (ffmpeg)
          </button>
        </div>

        {mode === "generate" && (
          <>
            <div className="mode-tabs">
              <button type="button" className={genMode === "t2v" ? "active" : ""} onClick={() => setGenMode("t2v")}>
                metinden video
              </button>
              <button type="button" className={genMode === "i2v" ? "active" : ""} onClick={() => setGenMode("i2v")}>
                görselden video
              </button>
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
                <option value="">Lightricks/LTX-Video (varsayılan, ilk kullanımda indirir)</option>
                {videoModels.map((m) => (
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
                  <button type="button" onClick={() => setSrcVideo(null)}>
                    kaldır
                  </button>
                )}
              </div>
            )}

            {editOp === "trim" && (
              <div className="dl-row">
                <label className="gen-num">
                  başlangıç (sn){" "}
                  <input type="number" min={0} step={0.1} value={trimStart} onChange={(e) => setTrimStart(Number(e.target.value))} />
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
