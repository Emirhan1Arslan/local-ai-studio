import { useEffect, useRef, useState } from "react";
import { createJob, fetchJobs, jobAction, type JobInfo } from "../api";
import { useStudio } from "../state";

const ACTIVE = new Set(["queued", "running", "paused"]);

export default function QueuePanel() {
  const { events, connected } = useStudio();
  const [jobs, setJobs] = useState<JobInfo[]>([]);
  const [progress, setProgress] = useState<Record<string, number>>({});
  const lastEvent = useRef(0);

  const refresh = () => fetchJobs().then(setJobs).catch(() => setJobs([]));

  useEffect(() => {
    if (connected) refresh();
  }, [connected]);

  useEffect(() => {
    let needsRefresh = false;
    for (let i = lastEvent.current; i < events.length; i++) {
      const e = events[i];
      if (!e.topic.startsWith("job.")) continue;
      if (e.topic === "job.progress") {
        const { id, value } = e.data as { id: string; value: number };
        setProgress((p) => ({ ...p, [id]: value }));
      } else {
        needsRefresh = true;
      }
    }
    lastEvent.current = events.length;
    if (needsRefresh) refresh();
  }, [events]);

  const addTestJob = () =>
    createJob("dummy", { steps: 20, step_ms: 150 }).then(refresh).catch(() => undefined);

  return (
    <div className="panel-body">
      <div className="toolbar">
        <button onClick={addTestJob} disabled={!connected}>
          Test işi ekle
        </button>
        <span className="muted">{jobs.filter((j) => ACTIVE.has(j.status)).length} aktif</span>
      </div>
      {jobs.length === 0 ? (
        <p className="muted">Kuyruk boş.</p>
      ) : (
        <div className="dl-list">
          {jobs.map((j) => (
            <div className="dl-item" key={j.id}>
              <div className="dl-head">
                <span className="dl-name">
                  {j.kind} <span className="muted">p{j.priority}</span>
                </span>
                <span className={`badge st-${j.status}`}>{j.status}</span>
                <span className="dl-actions">
                  {j.status === "queued" && (
                    <button onClick={() => jobAction(j.id, "pause").then(refresh)}>⏸</button>
                  )}
                  {j.status === "paused" && (
                    <button onClick={() => jobAction(j.id, "resume").then(refresh)}>▶</button>
                  )}
                  {ACTIVE.has(j.status) && (
                    <button onClick={() => jobAction(j.id, "cancel").then(refresh)}>✕</button>
                  )}
                </span>
              </div>
              {j.status === "running" && (
                <div className="dl-bar">
                  <div
                    className="dl-fill"
                    style={{ width: `${((progress[j.id] ?? 0) * 100).toFixed(0)}%` }}
                  />
                </div>
              )}
              {j.error && <p className="err-text">{j.error}</p>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
