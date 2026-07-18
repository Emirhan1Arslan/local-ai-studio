import { useEffect, useRef } from "react";
import { useStudio } from "../state";

export default function ConsolePanel() {
  const { events } = useStudio();
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [events]);
  return (
    <div className="panel-body console">
      {events.length === 0 && <p>Olay bekleniyor…</p>}
      {events.map((e, i) => (
        <div className="line" key={i}>
          <span className="ts">{new Date(e.ts).toLocaleTimeString()}</span>{" "}
          <span className="topic">[{e.topic}]</span> {JSON.stringify(e.data)}
        </div>
      ))}
      <div ref={endRef} />
    </div>
  );
}
