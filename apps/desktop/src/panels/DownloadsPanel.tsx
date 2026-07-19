import { useEffect, useRef, useState } from "react";
import {
  createDownload,
  createSnapshotDownload,
  downloadAction,
  fetchDownloads,
  fetchSettings,
  updateSettings,
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

interface DownloadGroup {
  key: string;
  label: string;
  items: DownloadInfo[];
}

/** Bir indirmenin models/<subdir>/<repo-adı>/<...> deseninde (snapshot indirme —
 * çok dosyalı bir model) olup olmadığını belirler; tekil dosya indirmeleri
 * (models/<subdir>/<dosya>) gruplanmaz. Grup anahtarı subdir/repo-adı'dır. */
function groupKeyFor(destPath: string, modelsDir: string): { group: string; label: string } | null {
  const norm = destPath.replace(/\\/g, "/");
  const modelsNorm = modelsDir.replace(/\\/g, "/").replace(/\/+$/, "");
  const rel = norm.startsWith(modelsNorm) ? norm.slice(modelsNorm.length).replace(/^\/+/, "") : norm;
  const parts = rel.split("/").filter(Boolean);
  if (parts.length < 3) return null; // subdir/dosya — tekil, grup yok
  return { group: parts.slice(0, 2).join("/"), label: parts[1] };
}

function groupStatus(items: DownloadInfo[]): string {
  if (items.some((d) => d.status === "downloading")) return "downloading";
  if (items.some((d) => d.status === "failed")) return "failed";
  if (items.some((d) => d.status === "queued")) return "queued";
  if (items.some((d) => d.status === "paused")) return "paused";
  if (items.every((d) => d.status === "done")) return "done";
  return "canceled";
}

export default function DownloadsPanel() {
  const { events, connected } = useStudio();
  const [rows, setRows] = useState<DownloadInfo[]>([]);
  const [modelsDir, setModelsDir] = useState("");
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
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

  // İndirme hız sınırı (Faz 8, kullanıcı isteğiyle): tüm eşzamanlı indirmeler
  // arasında paylaşılan toplam sınır — canlı etkili, restart gerekmez. Yazarken
  // her tuşta istek atmamak için 500ms debounce ile kaydediliyor.
  const [speedLimit, setSpeedLimit] = useState<number>(0);
  const [speedSaved, setSpeedSaved] = useState(true);
  const speedDebounce = useRef<ReturnType<typeof setTimeout> | null>(null);

  const refresh = () => fetchDownloads().then(setRows).catch(() => setRows([]));

  useEffect(() => {
    if (!connected) return;
    refresh();
    fetchSettings()
      .then((s) => {
        setSpeedLimit(s.download_speed_limit_mbps);
        setModelsDir(s.paths.models_dir);
      })
      .catch(() => undefined);
  }, [connected]);

  const toggleExpanded = (key: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  // Ham liste → tek dosyalık indirmeler + çok dosyalı model grupları.
  const groups: DownloadGroup[] = [];
  const standalone: DownloadInfo[] = [];
  {
    const byKey = new Map<string, DownloadGroup>();
    for (const d of rows) {
      const gk = modelsDir ? groupKeyFor(d.dest_path, modelsDir) : null;
      if (!gk) {
        standalone.push(d);
        continue;
      }
      let g = byKey.get(gk.group);
      if (!g) {
        g = { key: gk.group, label: gk.label, items: [] };
        byKey.set(gk.group, g);
        groups.push(g);
      }
      g.items.push(d);
    }
  }

  const onSpeedLimitChange = (value: number) => {
    setSpeedLimit(value);
    setSpeedSaved(false);
    if (speedDebounce.current) clearTimeout(speedDebounce.current);
    speedDebounce.current = setTimeout(async () => {
      try {
        await updateSettings({ download_speed_limit_mbps: value });
        setSpeedSaved(true);
      } catch {
        /* bir sonraki değişiklikte yeniden denenir */
      }
    }, 500);
  };

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
      <div className="dl-row" style={{ marginBottom: 10 }}>
        <label className="gen-num">
          hız sınırı (MB/sn, 0 = sınırsız){" "}
          <input
            type="number"
            min={0}
            max={1000}
            step={0.5}
            value={speedLimit}
            onChange={(e) => onSpeedLimitChange(Number(e.target.value))}
          />
        </label>
        <span className="muted">
          {speedSaved ? "tüm eşzamanlı indirmeler için toplam sınır" : "kaydediliyor…"}
        </span>
      </div>
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
          {groups.map((g) => {
            const isOpen = expanded.has(g.key);
            const downloaded = g.items.reduce((s, d) => s + d.downloaded_bytes, 0);
            const total = g.items.reduce((s, d) => s + d.total_bytes, 0);
            const pct = total > 0 ? (downloaded / total) * 100 : 0;
            const st = groupStatus(g.items);
            const doneCount = g.items.filter((d) => d.status === "done").length;
            const active = g.items.filter((d) => ACTIVE.has(d.status));
            const resumable = g.items.filter((d) => d.status === "paused" || d.status === "failed");
            const cancelable = g.items.filter((d) => d.status !== "done" && d.status !== "canceled");
            return (
              <div className="dl-item dl-group" key={g.key}>
                <div className="dl-head">
                  <button
                    type="button"
                    className="dl-expand"
                    onClick={() => toggleExpanded(g.key)}
                    aria-label={isOpen ? "daralt" : "genişlet"}
                  >
                    {isOpen ? "▾" : "▸"}
                  </button>
                  <span className="dl-name" title={g.key}>
                    {g.label}
                  </span>
                  <span className={`badge st-${st}`}>{st}</span>
                  <span className="muted">
                    {doneCount}/{g.items.length} dosya
                    {total > 0 && ` — ${fmtBytes(downloaded)} / ${fmtBytes(total)}`}
                  </span>
                  <span className="dl-actions">
                    {active.length > 0 && (
                      <button
                        onClick={() =>
                          Promise.all(active.map((d) => downloadAction(d.id, "pause"))).then(refresh)
                        }
                      >
                        ⏸ tümü
                      </button>
                    )}
                    {resumable.length > 0 && (
                      <button
                        onClick={() =>
                          Promise.all(resumable.map((d) => downloadAction(d.id, "resume"))).then(
                            refresh,
                          )
                        }
                      >
                        ▶ tümü
                      </button>
                    )}
                    {cancelable.length > 0 && (
                      <button
                        onClick={() =>
                          Promise.all(cancelable.map((d) => downloadAction(d.id, "cancel"))).then(
                            refresh,
                          )
                        }
                      >
                        ✕ tümü
                      </button>
                    )}
                  </span>
                </div>
                {st === "downloading" && (
                  <div className="dl-bar">
                    <div className="dl-fill" style={{ width: `${pct.toFixed(1)}%` }} />
                  </div>
                )}
                {isOpen && (
                  <div className="dl-group-files">
                    {g.items.map((d) => (
                      <DownloadRow key={d.id} d={d} nested onAction={refresh} />
                    ))}
                  </div>
                )}
              </div>
            );
          })}
          {standalone.map((d) => (
            <DownloadRow key={d.id} d={d} onAction={refresh} />
          ))}
        </div>
      )}
    </div>
  );
}

function DownloadRow({
  d,
  nested = false,
  onAction,
}: {
  d: DownloadInfo;
  nested?: boolean;
  onAction: () => void;
}) {
  const pct = d.total_bytes > 0 ? (d.downloaded_bytes / d.total_bytes) * 100 : 0;
  const name = d.dest_path.split(/[\\/]/).pop();
  return (
    <div className={nested ? "dl-item dl-item-nested" : "dl-item"}>
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
            <button onClick={() => downloadAction(d.id, "pause").then(onAction)}>⏸</button>
          )}
          {(d.status === "paused" || d.status === "failed") && (
            <button onClick={() => downloadAction(d.id, "resume").then(onAction)}>▶</button>
          )}
          {d.status !== "done" && d.status !== "canceled" && (
            <button onClick={() => downloadAction(d.id, "cancel").then(onAction)}>✕</button>
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
}
