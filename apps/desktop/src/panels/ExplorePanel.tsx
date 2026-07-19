import { useEffect, useRef, useState } from "react";
import {
  createDownload,
  createSnapshotDownload,
  fetchHfRepoFiles,
  searchHfModels,
  type HfModelInfo,
} from "../api";
import { SUBDIRS } from "./DownloadsPanel";

// Her models/ kategorisi için görünen ad + arama filtresinde kullanılacak HF pipeline_tag'i
// (bazılarının net bir karşılığı yok — "" bırakılırsa arama yalnızca metinle çalışır).
const CATEGORY_META: Record<string, { label: string; pipelineTag: string }> = {
  llm: { label: "LLM", pipelineTag: "text-generation" },
  checkpoints: { label: "checkpoint (genel)", pipelineTag: "text-to-image" },
  image: { label: "görüntü üretimi", pipelineTag: "text-to-image" },
  lora: { label: "LoRA", pipelineTag: "text-to-image" },
  controlnet: { label: "ControlNet", pipelineTag: "image-to-image" },
  vae: { label: "VAE", pipelineTag: "" },
  video: { label: "video üretimi", pipelineTag: "text-to-video" },
  "3d": { label: "3D üretim", pipelineTag: "image-to-3d" },
  tts: { label: "TTS (metin→ses)", pipelineTag: "text-to-speech" },
  stt: { label: "STT (ses→metin)", pipelineTag: "automatic-speech-recognition" },
  ocr: { label: "OCR", pipelineTag: "" },
  embeddings: { label: "embedding", pipelineTag: "feature-extraction" },
  background_removal: { label: "arka plan kaldırma", pipelineTag: "image-segmentation" },
  upscale: { label: "upscale", pipelineTag: "image-to-image" },
  captioning: { label: "caption", pipelineTag: "image-to-text" },
  segmentation: { label: "segmentasyon", pipelineTag: "mask-generation" },
  face_restoration: { label: "yüz onarma", pipelineTag: "image-to-image" },
};

function fmtCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

export default function ExplorePanel() {
  const [query, setQuery] = useState("");
  const [subdir, setSubdir] = useState("checkpoints");
  const [noTaskFilter, setNoTaskFilter] = useState(false);
  const [results, setResults] = useState<HfModelInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [expanded, setExpanded] = useState<string | null>(null);
  const [files, setFiles] = useState<string[]>([]);
  const [filesError, setFilesError] = useState("");
  const [status, setStatus] = useState("");
  const debounceRef = useRef<number | undefined>(undefined);

  const meta = CATEGORY_META[subdir];
  const pipelineTag = noTaskFilter ? "" : meta?.pipelineTag ?? "";

  // Kategori değişince arama SONUÇLARI da değişsin diye tek seçici hem filtreyi hem
  // indirme hedefini birlikte sürüyor (önceden iki ayrı seçici vardı, biri arama
  // sonucunu etkilemiyordu — kafa karıştırıcıydı).
  useEffect(() => {
    window.clearTimeout(debounceRef.current);
    debounceRef.current = window.setTimeout(() => {
      setLoading(true);
      setError("");
      searchHfModels(query, pipelineTag)
        .then(setResults)
        .catch((e) => setError(e instanceof Error ? e.message : String(e)))
        .finally(() => setLoading(false));
    }, 400);
    return () => window.clearTimeout(debounceRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query, pipelineTag]);

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
        <select value={subdir} onChange={(e) => setSubdir(e.target.value)}>
          {SUBDIRS.map((s) => (
            <option key={s} value={s}>
              {CATEGORY_META[s]?.label ?? s}
            </option>
          ))}
        </select>
      </div>
      <div className="dl-row">
        <label>
          <input
            type="checkbox"
            checked={noTaskFilter}
            onChange={(e) => setNoTaskFilter(e.target.checked)}
          />{" "}
          görev filtresi uygulama (yalnız metinle ara)
        </label>
        <span className="muted">
          indirilenler "{meta?.label ?? subdir}" klasörüne iner — Model Browser'da orada görünür
        </span>
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
