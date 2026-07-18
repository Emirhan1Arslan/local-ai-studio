import { useMemo } from "react";
import type { TelemetrySnapshot } from "../api";
import { useStudio } from "../state";

function Meter({ label, used, total, unit }: { label: string; used: number; total: number; unit: string }) {
  const pct = total > 0 ? (used / total) * 100 : 0;
  return (
    <div className="meter">
      <div className="meter-head">
        <span>{label}</span>
        <span className="muted">
          {used.toFixed(0)} / {total.toFixed(0)} {unit} ({pct.toFixed(0)}%)
        </span>
      </div>
      <div className="dl-bar">
        <div
          className="dl-fill"
          style={{ width: `${pct.toFixed(1)}%`, background: pct > 90 ? "var(--err)" : undefined }}
        />
      </div>
    </div>
  );
}

export default function VramPanel() {
  const { events } = useStudio();

  const snap = useMemo<TelemetrySnapshot | null>(() => {
    for (let i = events.length - 1; i >= 0; i--) {
      if (events[i].topic === "system.telemetry") {
        return events[i].data as unknown as TelemetrySnapshot;
      }
    }
    return null;
  }, [events]);

  if (!snap) return <div className="panel-body muted">Telemetri bekleniyor…</div>;

  return (
    <div className="panel-body">
      {snap.gpus.map((g) => (
        <div key={g.index}>
          <p className="gpu-name">{g.name}</p>
          <Meter label="VRAM" used={g.vram_used_mb / 1024} total={g.vram_total_mb / 1024} unit="GB" />
          <Meter label="GPU" used={g.util_percent} total={100} unit="%" />
        </div>
      ))}
      {snap.gpus.length === 0 && <p className="muted">GPU algılanmadı (CPU modu).</p>}
      <Meter label="RAM" used={snap.ram_used_mb / 1024} total={snap.ram_total_mb / 1024} unit="GB" />
      <Meter label="CPU" used={snap.cpu_percent} total={100} unit="%" />
    </div>
  );
}
