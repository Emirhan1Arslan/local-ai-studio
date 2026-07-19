import { useEffect, useRef, useState } from "react";
import {
  createDownload,
  createSnapshotDownload,
  fetchHfRepoFiles,
  searchHfModels,
  type HfModelInfo,
} from "../api";
import { SUBDIRS } from "./DownloadsPanel";

// [subdir ipucu, HF pipeline_tag, görünen ad] — subdir ipucu boşsa kullanıcı elle seçer
const TASK_FILTERS: [string, string, string][] = [
  ["", "", "tüm görevler"],
  ["checkpoints", "text-to-image", "görüntü üretimi (checkpoint)"],
  ["lora", "text-to-image", "LoRA"],
  ["controlnet", "image-to-image", "ControlNet"],
  ["background_removal", "image-segmentation", "arka plan kaldırma"],
  ["upscale", "image-to-image", "upscale"],
  ["captioning", "image-to-text", "caption"],
  ["segmentation", "mask-generation", "segmentasyon"],
  ["face_restoration", "image-to-image", "yüz onarma"],
  ["llm", "text-generation", "LLM"],
  ["embeddings", "feature-extraction", "embedding"],
  ["tts", "text-to-speech", "TTS"],
  ["stt", "automatic-speech-recognition", "STT"],
];

function fmtCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

export default function ExplorePanel() {
  const [query, setQuery] = useState("");
  const [taskIdx, setTaskIdx] = useState(0);
  const [results, setResults] = useState<HfModelInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [expanded, setExpanded] = useState<string | null>(null);
  const [files, setFiles] = useState<string[]>([]);
  const [filesError, setFilesError] = useState("");
  const [subdir, setSubdir] = useState("checkpoints");
  const [status, setStatus] = useState("");
  const debounceRef = useRef<number | undefined>(undefined);

  useEffect(() => {
    window.clearTimeout(debounceRef.current);
    debounceRef.current = window.setTimeout(() => {
      setLoading(true);
      setError("");
      searchHfModels(query, TASK_FILTERS[taskIdx][1])
        .then(setResults)
        .catch((e) => setError(e instanceof Error ? e.message : String(e)))
        .finally(() => setLoading(false));
    }, 400);
    return () => window.clearTimeout(debounceRef.current);
  }, [query, taskIdx]);

  useEffect(() => {
    const hint = TASK_FILTERS[taskIdx][0];
    if (hint) setSubdir(hint);
  }, [taskIdx]);

  const toggleExpand = async (id: string) => {
    if (expanded === id) {
      setExpanded(null);
      return;
    }
    setExpanded(id);
    setFiles([]);
    setFilesError("");
    try {
      setFiles(await fetchHfRepoFiles(id));
    } catch (e) {
      setFilesError(e instanceof Error ? e.message : String(e));
    }
  };

  const downloadSnapshot = async (id: string) => {
    setStatus("");
    setError("");
    try {
      const rows = await createSnapshotDownload({ repo_id: id, subdir });
      setStatus(`${id}: ${rows.length} dosya kuyruğa alındı — İndirmeler panelinden izleyin`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const downloadFile = async (id: string, filename: string) => {
    setStatus("");
    setError("");
    try {
      await createDownload({ repo_id: id, filename, subdir });
      setStatus(`${filename} kuyruğa alındı — İndirmeler panelinden izleyin`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div className="panel-body">
      <div className="dl-row">
        <input
          type="text"
          placeholder="model ara… (ör. birefnet, sdxl, whisper)"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <select value={taskIdx} onChange={(e) => setTaskIdx(Number(e.target.value))}>
          {TASK_FILTERS.map(([, , label], i) => (
            <option key={label} value={i}>
              {label}
            </option>
          ))}
        </select>
      </div>
      <div className="dl-row">
        <label className="gen-num">
          hedef kategori{" "}
          <select value={subdir} onChange={(e) => setSubdir(e.target.value)}>
            {SUBDIRS.map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
        </label>
        <span className="muted">indirilenler bu klasöre iner (Model Browser'da orada görünür)</span>
      </div>
      {status && <p className="ok-text">{status}</p>}
      {error && <p className="err-text">{error}</p>}
      {loading && <p className="muted">aranıyor…</p>}

      <div className="hf-results">
        {results.map((m) => (
          <div className="hf-card" key={m.id}>
            <div className="hf-card-head">
              <span className="hf-card-name">{m.id}</span>
              {m.gated && <span className="badge st-failed">kısıtlı erişim</span>}
              {m.license && <span className="fmt">{m.license}</span>}
            </div>
            <div className="hf-card-meta muted">
              {m.pipeline_tag && <span>{m.pipeline_tag}</span>}
              <span>⬇ {fmtCount(m.downloads)}</span>
              <span>♥ {fmtCount(m.likes)}</span>
            </div>
            <div className="dl-row">
              <button type="button" onClick={() => downloadSnapshot(m.id)}>
                tüm repo'yu indir
              </button>
              <button type="button" onClick={() => toggleExpand(m.id)}>
                {expanded === m.id ? "dosyaları gizle" : "dosyaları göster"}
              </button>
            </div>
            {expanded === m.id && (
              <div className="hf-files">
                {filesError && <p className="err-text">{filesError}</p>}
                {files.map((f) => (
                  <div className="hf-file-row" key={f}>
                    <span>{f}</span>
                    <button type="button" onClick={() => downloadFile(m.id, f)}>
                      indir
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
      {!loading && results.length === 0 && (
        <p className="muted">Sonuç yok. Bir arama terimi girin (ör. "sdxl", "whisper").</p>
      )}
    </div>
  );
}
