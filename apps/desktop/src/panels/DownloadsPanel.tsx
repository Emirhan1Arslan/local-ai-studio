import { useEffect, useRef, useState } from "react";
import {
  createDownload,
  createSnapshotDownload,
  downloadAction,
  fetchDownloads,
  type DownloadInfo,
} from "../api";
import { useStudio } from "../state";

export const SUBDIRS = [
  "llm",
  "checkpoints",
  "image",
  "lora",
  "controlnet",
  "vae",
  "video",
  "3d",
  "tts",
  "stt",
  "ocr",
  "embeddings",
  "background_removal",
  "upscale",
  "captioning",
  "segmentation",
  "face_restoration",
];

export function fmtBytes(n: number): string {
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} GB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024).toFixed(0)} KB`;
}

const ACTIVE = new Set(["queued", "downloading"]);

export default function DownloadsPanel() {
  const { events, connected } = useStudio();
  const [rows, setRows] = useState<DownloadInfo[]>([]);
  const [mode, setMode] = useState<"url" | "hf">("url");
  const [url, setUrl] = useState("");
  const [repoId, setRepoId] = useState("");
  const [filename, setFilename] = useState("");
  const [snapshot, setSnapshot] = useState(false);
  const [subdir, setSubdir] = useState("llm");
  const [sha, setSha] = useState("");
  const [formError, setFormError] = useState("");
  const [formStatus, setFormStatus] = useState("");
  const lastEvent = useRef(0);

  const refresh = () => fetchDownloads().then(setRows).catch(() => setRows([]));

  useEffect(() => {
    if (connected) refresh();
  }, [connected]);

  // download.* olayları: progress → satırı yerinde güncelle; diğerleri → yeniden yükle
  useEffect(() => {
    let needsRefresh = false;
    for (let i = lastEvent.current; i < events.length; i++) {
      const e = events[i];
      if (!e.topic.startsWith("download.")) continue;
      if (e.topic === "download.progress") {
        const { id, downloaded, total } = e.data as {
          id: string;
          downloaded: number;
          total: number;
        };
        setRows((prev) =>
          prev.map((r) =>
            r.id === id ? { ...r, downloaded_bytes: downloaded, total_bytes: total } : r,
          ),
        );
      } else {
        needsRefresh = true;
      }
    }
    lastEvent.current = events.length;
    if (needsRefresh) refresh();
  }, [events]);

  const submit = async (ev: React.FormEvent) => {
    ev.preventDefault();
    setFormError("");
    setFormStatus("");
    try {
      if (mode === "hf" && snapshot) {
        const rows = await createSnapshotDownload({ repo_id: repoId, subdir });
        setFormStatus(`${rows.length} dosya kuyruğa alındı`);
      } else {
        await createDownload(
          mode === "hf"
            ? { repo_id: repoId, filename, subdir, sha256: sha || undefined }
            : { url, filename: filename || undefined, subdir, sha256: sha || undefined },
        );
      }
      setUrl("");
      setSha("");
      refresh();
    } catch (e) {
      setFormError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div className="panel-body">
      <form className="dl-form" onSubmit={submit}>
        <div className="dl-row">
          <label>
            <input
              type="radio"
              checked={mode === "url"}
              onChange={() => setMode("url")}
            />{" "}
            URL
          </label>
          <label>
            <input type="radio" checked={mode === "hf"} onChange={() => setMode("hf")} /> Hugging
            Face
          </label>
          <select value={subdir} onChange={(e) => setSubdir(e.target.value)}>
            {SUBDIRS.map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
        </div>
        {mode === "url" ? (
          <input
            placeholder="https://… (doğrudan dosya URL'i)"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            required
          />
        ) : (
          <>
            <div className="dl-row">
              <input
                placeholder="repo_id (ör. TheBloke/Qwen-7B-GGUF)"
                value={repoId}
                onChange={(e) => setRepoId(e.target.value)}
                required
              />
              {!snapshot && (
                <input
                  placeholder="dosya adı (ör. model.q4_k_m.gguf)"
                  value={filename}
                  onChange={(e) => setFilename(e.target.value)}
                  required
                />
              )}
            </div>
            <div className="dl-row">
              <label>
                <input
                  type="checkbox"
                  checked={snapshot}
                  onChange={(e) => setSnapshot(e.target.checked)}
                />{" "}
                tüm repo (birden fazla dosyalı model — ör. BiRefNet, Florence-2, SAM)
              </label>
            </div>
          </>
        )}
        <div className="dl-row">
          {!(mode === "hf" && snapshot) && (
            <input
              placeholder="sha256 (opsiyonel doğrulama)"
              value={sha}
              onChange={(e) => setSha(e.target.value)}
            />
          )}
          <button type="submit" disabled={!connected}>
            İndir
          </button>
        </div>
        {formStatus && <p className="ok-text">{formStatus}</p>}
        {formError && <p className="err-text">{formError}</p>}
      </form>

      {rows.length === 0 ? (
        <p className="muted">İndirme yok.</p>
      ) : (
        <div className="dl-list">
          {rows.map((d) => {
            const pct = d.total_bytes > 0 ? (d.downloaded_bytes / d.total_bytes) * 100 : 0;
            const name = d.dest_path.split(/[\\/]/).pop();
            return (
              <div className="dl-item" key={d.id}>
                <div className="dl-head">
                  <span className="dl-name" title={d.url}>
                    {name}
                  </span>
                  <span className={`badge st-${d.status}`}>{d.status}</span>
                  <span className="muted">
                    {fmtBytes(d.downloaded_bytes)}
                    {d.total_bytes > 0 && ` / ${fmtBytes(d.total_bytes)}`}
                  </span>
                  <span className="dl-actions">
                    {ACTIVE.has(d.status) && (
                      <button onClick={() => downloadAction(d.id, "pause").then(refresh)}>⏸</button>
                    )}
                    {(d.status === "paused" || d.status === "failed") && (
                      <button onClick={() => downloadAction(d.id, "resume").then(refresh)}>▶</button>
                    )}
                    {d.status !== "done" && d.status !== "canceled" && (
                      <button onClick={() => downloadAction(d.id, "cancel").then(refresh)}>✕</button>
                    )}
                  </span>
                </div>
                {ACTIVE.has(d.status) && (
                  <div className="dl-bar">
                    <div className="dl-fill" style={{ width: `${pct.toFixed(1)}%` }} />
                  </div>
                )}
                {d.error && <p className="err-text">{d.error}</p>}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
