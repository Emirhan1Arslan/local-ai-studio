import { useEffect, useRef, useState } from "react";
import {
  apiUrl,
  createJob,
  fetchAssets,
  fetchJobs,
  uploadAudioFile,
  type AssetInfo,
  type JobInfo,
} from "../api";
import { useStudio } from "../state";

type Mode = "stt" | "tts";

const VOICES: [string, string][] = [
  ["af_heart", "Heart (kadın, A. İngilizcesi)"],
  ["af_bella", "Bella (kadın, A. İngilizcesi)"],
  ["af_nicole", "Nicole (kadın, A. İngilizcesi)"],
  ["am_adam", "Adam (erkek, A. İngilizcesi)"],
  ["am_michael", "Michael (erkek, A. İngilizcesi)"],
  ["bf_emma", "Emma (kadın, B. İngilizcesi)"],
  ["bm_george", "George (erkek, B. İngilizcesi)"],
];

function outUrl(path: string): string {
  return apiUrl(`/outputs/${path.split(/[\\/]/).pop()}`);
}

export default function AudioPanel() {
  const { events, connected } = useStudio();
  const [mode, setMode] = useState<Mode>("stt");

  // STT
  const [audioFile, setAudioFile] = useState<{ path: string; name: string } | null>(null);
  const [modelSize, setModelSize] = useState("base");
  const [language, setLanguage] = useState("");
  const [recording, setRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunks = useRef<Blob[]>([]);

  // TTS
  const [text, setText] = useState("");
  const [voice, setVoice] = useState("af_heart");
  const [speed, setSpeed] = useState(1.0);

  const [jobs, setJobs] = useState<JobInfo[]>([]);
  const [results, setResults] = useState<AssetInfo[]>([]);
  const [progress, setProgress] = useState<
    Record<string, { value: number; message: string; delta?: string }>
  >({});
  const [formError, setFormError] = useState("");
  const lastEvent = useRef(0);

  const refresh = () => {
    fetchJobs().then((all) => setJobs(all.filter((j) => j.kind === "audio"))).catch(() => undefined);
    fetchAssets("audio", 24).then(setResults).catch(() => undefined);
  };

  useEffect(() => {
    if (connected) refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [connected]);

  useEffect(() => {
    let needsRefresh = false;
    for (let i = lastEvent.current; i < events.length; i++) {
      const e = events[i];
      if (e.topic === "job.progress") {
        const { id, value, message, delta } = e.data as {
          id: string;
          value: number;
          message: string;
          delta?: string;
        };
        setProgress((p) => ({ ...p, [id]: { value, message, delta } }));
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
      const up = await uploadAudioFile(file);
      setAudioFile({ path: up.path, name: file.name });
    } catch (e) {
      setFormError(e instanceof Error ? e.message : String(e));
    }
  };

  const startRecording = async () => {
    setFormError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const rec = new MediaRecorder(stream);
      recordedChunks.current = [];
      rec.ondataavailable = (e) => {
        if (e.data.size > 0) recordedChunks.current.push(e.data);
      };
      rec.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(recordedChunks.current, { type: "audio/webm" });
        try {
          const up = await uploadAudioFile(blob, "kayit.webm");
          setAudioFile({ path: up.path, name: "mikrofon kaydı" });
        } catch (e) {
          setFormError(e instanceof Error ? e.message : String(e));
        }
      };
      rec.start();
      mediaRecorderRef.current = rec;
      setRecording(true);
    } catch (e) {
      setFormError(
        e instanceof Error ? `mikrofon erişimi reddedildi: ${e.message}` : String(e),
      );
    }
  };

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
    mediaRecorderRef.current = null;
    setRecording(false);
  };

  const run = async (ev: React.FormEvent) => {
    ev.preventDefault();
    setFormError("");
    const params: Record<string, unknown> = { op: mode };
    if (mode === "stt") {
      if (!audioFile) {
        setFormError("bir ses dosyası yükleyin ya da mikrofonla kaydedin");
        return;
      }
      params.audio_path = audioFile.path;
      params.model_size = modelSize;
      if (language.trim()) params.language = language.trim();
    } else {
      if (!text.trim()) {
        setFormError("seslendirilecek metni yazın");
        return;
      }
      params.text = text.trim();
      params.voice = voice;
      params.lang_code = voice.startsWith("b") ? "b" : "a";
      params.speed = speed;
    }
    try {
      await createJob("audio", params);
      refresh();
    } catch (e) {
      setFormError(e instanceof Error ? e.message : String(e));
    }
  };

  const runningPreview = running ? progress[running.id] : undefined;

  return (
    <div className="panel-body">
      <form className="dl-form" onSubmit={run}>
        <div className="mode-tabs">
          <button type="button" className={mode === "stt" ? "active" : ""} onClick={() => setMode("stt")}>
            transkripsiyon
          </button>
          <button type="button" className={mode === "tts" ? "active" : ""} onClick={() => setMode("tts")}>
            seslendirme
          </button>
        </div>

        {mode === "stt" && (
          <>
            <div className="dl-row">
              <label className="file-pick">
                ses dosyası yükle…
                <input
                  type="file"
                  accept="audio/*"
                  onChange={(e) => e.target.files?.[0] && pickFile(e.target.files[0])}
                />
              </label>
              {!recording ? (
                <button type="button" onClick={startRecording}>
                  🎙 mikrofonla kaydet
                </button>
              ) : (
                <button type="button" onClick={stopRecording} className="recording">
                  ⏹ kaydı durdur
                </button>
              )}
              {audioFile && <span className="muted">{audioFile.name}</span>}
              {audioFile && (
                <button type="button" onClick={() => setAudioFile(null)}>
                  kaldır
                </button>
              )}
            </div>
            <div className="dl-row">
              <select value={modelSize} onChange={(e) => setModelSize(e.target.value)}>
                <option value="tiny">tiny (en hızlı)</option>
                <option value="base">base</option>
                <option value="small">small</option>
                <option value="medium">medium</option>
                <option value="large-v3">large-v3 (en doğru)</option>
              </select>
              <input
                type="text"
                placeholder="dil kodu (opsiyonel — boşsa otomatik algıla, ör. tr, en)"
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
              />
              <span className="muted">çıktı: json + txt + srt + vtt</span>
            </div>
          </>
        )}

        {mode === "tts" && (
          <>
            <textarea
              placeholder="seslendirilecek metni yazın…"
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={3}
            />
            <div className="dl-row">
              <select value={voice} onChange={(e) => setVoice(e.target.value)}>
                {VOICES.map(([v, label]) => (
                  <option key={v} value={v}>
                    {label}
                  </option>
                ))}
              </select>
              <label className="gen-num">
                hız{" "}
                <input
                  type="number"
                  min={0.5}
                  max={2}
                  step={0.1}
                  value={speed}
                  onChange={(e) => setSpeed(Number(e.target.value))}
                />
              </label>
            </div>
          </>
        )}

        <div className="dl-row">
          <button type="submit" disabled={!connected || !!running}>
            {running ? "çalışıyor…" : mode === "stt" ? "transkribe et" : "seslendir"}
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
            <div
              className="dl-fill"
              style={{ width: `${((runningPreview?.value ?? 0) * 100).toFixed(0)}%` }}
            />
          </div>
        </div>
      )}

      <div className="audio-results">
        {results.map((a) => {
          const resultOp = String(a.meta.params?.op ?? "");
          const isTts = resultOp === "tts";
          const transcript = String(a.meta.result?.text ?? "");
          return (
            <div className="audio-result-card" key={a.id}>
              {isTts ? (
                <>
                  <audio controls src={outUrl(a.path)} />
                  <p className="muted">"{String(a.meta.params?.text ?? "").slice(0, 120)}"</p>
                </>
              ) : (
                <>
                  <span className="fmt">transkript</span>
                  <p title={transcript}>{transcript.slice(0, 300) || "(boş transkript)"}</p>
                </>
              )}
            </div>
          );
        })}
      </div>
      {results.length === 0 && !running && (
        <p className="muted">Henüz sonuç yok. Bir ses dosyası yükleyin ya da metin yazın.</p>
      )}
    </div>
  );
}
