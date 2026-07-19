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
type Engine = "kokoro" | "xtts";

const VOICES: [string, string][] = [
  ["af_heart", "Heart (kadın, A. İngilizcesi)"],
  ["af_bella", "Bella (kadın, A. İngilizcesi)"],
  ["af_nicole", "Nicole (kadın, A. İngilizcesi)"],
  ["am_adam", "Adam (erkek, A. İngilizcesi)"],
  ["am_michael", "Michael (erkek, A. İngilizcesi)"],
  ["bf_emma", "Emma (kadın, B. İngilizcesi)"],
  ["bm_george", "George (erkek, B. İngilizcesi)"],
];

// XTTS-v2 checkpoint'ine gömülü 58 hazır ses (klonlama gerekmez)
const XTTS_PRESET_SPEAKERS: string[] = [
  "Claribel Dervla", "Daisy Studious", "Gracie Wise", "Tammie Ema",
  "Alison Dietlinde", "Ana Florence", "Annmarie Nele", "Asya Anara",
  "Brenda Stern", "Gitta Nikolina", "Henriette Usha", "Sofia Hellen",
  "Tammy Grit", "Tanja Adelina", "Vjollca Johnnie", "Andrew Chipper",
  "Badr Odhiambo", "Dionisio Schuyler", "Royston Min", "Viktor Eka",
  "Abrahan Mack", "Adde Michal", "Baldur Sanjin", "Craig Gutsy",
  "Damien Black", "Gilberto Mathias", "Ilkin Urbano", "Kazuhiko Atallah",
  "Ludvig Milivoj", "Suad Qasim", "Torcull Diarmuid", "Viktor Menelaos",
  "Zacharie Aimilios", "Nova Hogarth", "Maja Ruoho", "Uta Obando",
  "Lidiya Szekeres", "Chandra MacFarland", "Szofi Granger", "Camilla Holmström",
  "Lilya Stainthorpe", "Zofija Kendrick", "Narelle Moon", "Barbora MacLean",
  "Alexandra Hisakawa", "Alma María", "Rosemary Okafor", "Ige Behringer",
  "Filip Traverse", "Damjan Chapman", "Wulf Carlevaro", "Aaron Dreschner",
  "Kumar Dahl", "Eugenio Mataracı", "Ferran Simen", "Xavier Hayasaka",
  "Luis Moray", "Marcos Rudaski",
];

// XTTS-v2'nin desteklediği diller (coqui-tts multilingual/multi-dataset/xtts_v2)
const XTTS_LANGUAGES: [string, string][] = [
  ["tr", "Türkçe"],
  ["en", "İngilizce"],
  ["de", "Almanca"],
  ["fr", "Fransızca"],
  ["es", "İspanyolca"],
  ["it", "İtalyanca"],
  ["pt", "Portekizce"],
  ["pl", "Lehçe"],
  ["ru", "Rusça"],
  ["nl", "Felemenkçe"],
  ["cs", "Çekçe"],
  ["ar", "Arapça"],
  ["zh-cn", "Çince (basitleştirilmiş)"],
  ["ja", "Japonca"],
  ["hu", "Macarca"],
  ["ko", "Korece"],
  ["hi", "Hintçe"],
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
  const [engine, setEngine] = useState<Engine>("kokoro");
  const [voice, setVoice] = useState("af_heart");
  const [speed, setSpeed] = useState(1.0);
  const [xttsVoiceMode, setXttsVoiceMode] = useState<"preset" | "clone">("preset");
  const [xttsPresetSpeaker, setXttsPresetSpeaker] = useState(XTTS_PRESET_SPEAKERS[0]);
  const [speakerFile, setSpeakerFile] = useState<{ path: string; name: string } | null>(null);
  const [xttsLanguage, setXttsLanguage] = useState("tr");
  const [consent, setConsent] = useState(false);

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

  const pickSpeakerFile = async (file: File) => {
    setFormError("");
    try {
      const up = await uploadAudioFile(file);
      setSpeakerFile({ path: up.path, name: file.name });
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
      params.engine = engine;
      if (engine === "kokoro") {
        params.voice = voice;
        params.lang_code = voice.startsWith("b") ? "b" : "a";
        params.speed = speed;
      } else {
        params.language = xttsLanguage;
        if (xttsVoiceMode === "preset") {
          params.speaker = xttsPresetSpeaker;
        } else {
          if (!speakerFile) {
            setFormError("klonlanacak bir referans ses dosyası yükleyin");
            return;
          }
          if (!consent) {
            setFormError("devam etmeden önce ses klonlama onayını işaretleyin");
            return;
          }
          params.speaker_wav = speakerFile.path;
          params.consent = true;
        }
      }
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
            <div className="mode-tabs">
              <button
                type="button"
                className={engine === "kokoro" ? "active" : ""}
                onClick={() => setEngine("kokoro")}
              >
                Kokoro (hazır sesler)
              </button>
              <button
                type="button"
                className={engine === "xtts" ? "active" : ""}
                onClick={() => setEngine("xtts")}
              >
                XTTS-v2 (ses klonlama, çok dilli)
              </button>
            </div>

            {engine === "kokoro" && (
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
            )}

            {engine === "xtts" && (
              <>
                <p className="muted">
                  XTTS-v2 model ağırlıkları <strong>Coqui Public Model License (CPML)</strong>{" "}
                  ile dağıtılır — yalnızca ticari OLMAYAN kullanım içindir.
                </p>
                <div className="mode-tabs">
                  <button
                    type="button"
                    className={xttsVoiceMode === "preset" ? "active" : ""}
                    onClick={() => setXttsVoiceMode("preset")}
                  >
                    hazır ses (klonlama yok)
                  </button>
                  <button
                    type="button"
                    className={xttsVoiceMode === "clone" ? "active" : ""}
                    onClick={() => setXttsVoiceMode("clone")}
                  >
                    referans sesle klonla
                  </button>
                </div>

                {xttsVoiceMode === "preset" ? (
                  <div className="dl-row">
                    <select
                      value={xttsPresetSpeaker}
                      onChange={(e) => setXttsPresetSpeaker(e.target.value)}
                    >
                      {XTTS_PRESET_SPEAKERS.map((name) => (
                        <option key={name} value={name}>
                          {name}
                        </option>
                      ))}
                    </select>
                  </div>
                ) : (
                  <>
                    <div className="dl-row">
                      <label className="file-pick">
                        referans ses yükle (klonlanacak ses)…
                        <input
                          type="file"
                          accept="audio/*"
                          onChange={(e) => e.target.files?.[0] && pickSpeakerFile(e.target.files[0])}
                        />
                      </label>
                      {speakerFile && <span className="muted">{speakerFile.name}</span>}
                      {speakerFile && (
                        <button type="button" onClick={() => setSpeakerFile(null)}>
                          kaldır
                        </button>
                      )}
                    </div>
                    <label className="dl-row consent-row">
                      <input
                        type="checkbox"
                        checked={consent}
                        onChange={(e) => setConsent(e.target.checked)}
                      />
                      <span>
                        Yüklediğim referans sesin sahibi olduğumu veya sesini klonlamak için
                        sahibinden açık izin aldığımı beyan ederim.
                      </span>
                    </label>
                  </>
                )}

                <div className="dl-row">
                  <select value={xttsLanguage} onChange={(e) => setXttsLanguage(e.target.value)}>
                    {XTTS_LANGUAGES.map(([code, label]) => (
                      <option key={code} value={code}>
                        {label}
                      </option>
                    ))}
                  </select>
                </div>
              </>
            )}
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
          const resultEngine = String(a.meta.params?.engine ?? "kokoro");
          const resultCloned = Boolean(a.meta.result?.cloned);
          const transcript = String(a.meta.result?.text ?? "");
          return (
            <div className="audio-result-card" key={a.id}>
              {isTts ? (
                <>
                  {resultEngine === "xtts" && (
                    <span className="fmt">{resultCloned ? "XTTS-v2 klon" : "XTTS-v2"}</span>
                  )}
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
