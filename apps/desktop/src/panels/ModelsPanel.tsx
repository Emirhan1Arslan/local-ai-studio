import { useEffect, useState } from "react";
import { fetchModels, rescanModels, type ModelInfo } from "../api";
import { useStudio } from "../state";

function fmtSize(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024).toFixed(0)} KB`;
}

export default function ModelsPanel() {
  const { modelsVersion, info, connected } = useStudio();
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!connected) return;
    fetchModels()
      .then(setModels)
      .catch(() => setModels([]));
  }, [modelsVersion, connected]);

  const rescan = async () => {
    setBusy(true);
    try {
      await rescanModels();
      setModels(await fetchModels());
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="panel-body">
      <div className="toolbar">
        <button onClick={rescan} disabled={busy || !connected}>
          {busy ? "Taranıyor…" : "Yeniden tara"}
        </button>
        <span className="muted">{models.length} model</span>
      </div>
      {models.length === 0 ? (
        <p className="muted">
          Model bulunamadı. Dosyaları{" "}
          <code>{info ? `${info.data_dir}\\models` : "~/LocalAIStudio/models"}</code> altına
          bırakın — otomatik algılanır.
        </p>
      ) : (
        <table className="models">
          <thead>
            <tr>
              <th>Ad</th>
              <th>Tür</th>
              <th>Format</th>
              <th>Aile</th>
              <th>Boyut</th>
            </tr>
          </thead>
          <tbody>
            {models.map((m) => (
              <tr key={m.id} title={m.path}>
                <td>{m.name}</td>
                <td>{m.kind}</td>
                <td>
                  <span className={`fmt fmt-${m.format}`}>{m.format}</span>
                </td>
                <td>{m.family || "—"}</td>
                <td>{fmtSize(m.size_bytes)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
