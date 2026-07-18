import { useEffect, useMemo, useRef, useState } from "react";
import { useStudio } from "../state";

export default function ConsolePanel() {
  const { events } = useStudio();
  const [showTelemetry, setShowTelemetry] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  const visible = useMemo(
    () => (showTelemetry ? events : events.filter((e) => e.topic !== "system.telemetry")),
    [events, showTelemetry],
  );

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [visible]);

  return (
    <div className="panel-body console">
      <label className="console-filter">
        <input
          type="checkbox"
          checked={showTelemetry}
          onChange={(e) => setShowTelemetry(e.target.checked)}
        />{" "}
        telemetriyi göster
      </label>
      {visible.length === 0 && <p>Olay bekleniyor…</p>}
      {visible.map((e, i) => (
        <div className="line" key={i}>
          <span className="ts">{new Date(e.ts).toLocaleTimeString()}</span>{" "}
          <span className="topic">[{e.topic}]</span>{" "}
          {(() => {
            // canlı önizleme base64'ü gibi dev alanlar konsolu boğmasın
            const s = JSON.stringify(e.data);
            return s.length > 300 ? s.slice(0, 300) + "…" : s;
          })()}
        </div>
      ))}
      <div ref={endRef} />
    </div>
  );
}
