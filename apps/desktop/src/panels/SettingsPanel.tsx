import { useEffect, useState } from "react";
import {
  createDiagnosticsBundle,
  fetchSettings,
  updateSettings,
  type SettingsInfo,
} from "../api";
import { useStudio } from "../state";

const PATH_LABELS: [keyof SettingsInfo["paths"], string][] = [
  ["data_dir", "Veri dizini"],
  ["models_dir", "Modeller"],
  ["outputs_dir", "Üretim çıktıları"],
  ["cache_dir", "Önbellek"],
  ["logs_dir", "Loglar"],
  ["envs_dir", "Worker ortamları"],
  ["uploads_dir", "Yüklemeler"],
  ["db_path", "Veritabanı"],
];

export default function SettingsPanel() {
  const { connected } = useStudio();
  const [settings, setSettings] = useState<SettingsInfo | null>(null);
  const [hfToken, setHfToken] = useState("");
  const [telemetry, setTelemetry] = useState(2);
  const [idleTimeout, setIdleTimeout] = useState(300);
  const [concurrency, setConcurrency] = useState(1);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [diagPath, setDiagPath] = useState("");
  const [busy, setBusy] = useState(false);

  const load = () => {
    fetchSettings()
      .then((s) => {
        setSettings(s);
        setTelemetry(s.telemetry_interval);
        setIdleTimeout(s.worker_idle_timeout);
        setConcurrency(s.worker_concurrency);
      })
      .catch(() => setSettings(null));
  };

  useEffect(() => {
    if (connected) load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [connected]);

  const save = async (patch: Parameters<typeof updateSettings>[0], msg: string) => {
    setError("");
    setStatus("");
    setBusy(true);
    try {
      const s = await updateSettings(patch);
      setSettings(s);
      setStatus(msg);
      window.setTimeout(() => setStatus(""), 2500);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const makeDiagnostics = async () => {
    setError("");
    setDiagPath("");
    setBusy(true);
    try {
      const { path } = await createDiagnosticsBundle();
      setDiagPath(path);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  if (!connected || !settings) {
    return (
      <div className="panel-body">
        <p className="muted">Backend bekleniyor…</p>
      </div>
    );
  }

  return (
    <div className="panel-body">
      <section className="settings-section">
        <h3>Hugging Face token</h3>
        <div className="dl-row">
          <input
            type="password"
            placeholder={settings.hf_token_set ? "•••••••• (ayarlı — değiştirmek için yazın)" : "hf_xxx…"}
            value={hfToken}
            onChange={(e) => setHfToken(e.target.value)}
          />
          <button
            disabled={busy || !hfToken}
            onClick={() => save({ hf_token: hfToken }, "token kaydedildi").then(() => setHfToken(""))}
          >
            kaydet
          </button>
          {settings.hf_token_set && (
            <button
              disabled={busy}
              onClick={() => save({ hf_token: "" }, "token temizlendi")}
            >
              temizle
            </button>
          )}
        </div>
        <p className="muted">Kapalı/lisans onaylı HF repoları için (ör. FLUX-dev). Asla loglanmaz.</p>
      </section>

      <section className="settings-section">
        <h3>Çalışma zamanı</h3>
        <div className="dl-row">
          <label className="gen-num">
            telemetri aralığı (sn){" "}
            <input
              type="number"
              min={0.5}
              max={30}
              step={0.5}
              value={telemetry}
              onChange={(e) => setTelemetry(Number(e.target.value))}
            />
          </label>
          <label className="gen-num">
            worker boşta kalma (sn){" "}
            <input
              type="number"
              min={10}
              max={3600}
              step={10}
              value={idleTimeout}
              onChange={(e) => setIdleTimeout(Number(e.target.value))}
            />
          </label>
          <label className="gen-num">
            worker eşzamanlılığı{" "}
            <input
              type="number"
              min={1}
              max={4}
              value={concurrency}
              onChange={(e) => setConcurrency(Number(e.target.value))}
            />
          </label>
          <button
            disabled={busy}
            onClick={() =>
              save(
                {
                  telemetry_interval: telemetry,
                  worker_idle_timeout: idleTimeout,
                  worker_concurrency: concurrency,
                },
                "kaydedildi — anında etkili",
              )
            }
          >
            kaydet
          </button>
        </div>
        <p className="muted">
          Değişiklikler restart gerektirmez. Eşzamanlılık &gt;1 yalnız birden fazla GPU'nuz varsa
          önerilir; tek GPU'da aynı anda iki ağır iş VRAM taşmasına yol açabilir.
        </p>
      </section>

      <section className="settings-section">
        <h3>Yollar</h3>
        <dl className="kv">
          {PATH_LABELS.map(([key, label]) => (
            <div className="path-row" key={key}>
              <dt>{label}</dt>
              <dd title={settings.paths[key]}>{settings.paths[key]}</dd>
            </div>
          ))}
        </dl>
        <p className="muted">
          Salt okunur — değiştirmek veri taşımayı gerektirir (ileride ilk-çalıştırma sihirbazı).
        </p>
      </section>

      <section className="settings-section">
        <h3>Tanı paketi</h3>
        <div className="dl-row">
          <button disabled={busy} onClick={makeDiagnostics}>
            tanı paketi oluştur
          </button>
          {diagPath && <span className="muted">oluşturuldu: {diagPath}</span>}
        </div>
        <p className="muted">
          Loglar + sistem bilgisi + son başarısız işler tek zip'e paketlenir. Hiçbir yere
          gönderilmez — bir sorunu bildirirken elle paylaşabilirsiniz.
        </p>
      </section>

      {status && <p className="ok-text">{status}</p>}
      {error && <p className="err-text">{error}</p>}
    </div>
  );
}
