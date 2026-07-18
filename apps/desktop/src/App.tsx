import { useEffect, useRef, useState } from "react";
import {
  fetchHealth,
  fetchSystemInfo,
  openEventSocket,
  type StudioEvent,
  type SystemInfo,
} from "./api";

function SystemPanel({ info }: { info: SystemInfo | null }) {
  return (
    <section className="panel">
      <header>Sistem</header>
      <div className="body">
        {!info ? (
          <p>Backend bekleniyor… (uv run studio-api)</p>
        ) : (
          <dl className="kv">
            <dt>OS</dt>
            <dd>
              {info.os} {info.arch}
            </dd>
            <dt>Python</dt>
            <dd>{info.python}</dd>
            <dt>CPU</dt>
            <dd>{info.cpu_count} çekirdek</dd>
            <dt>RAM</dt>
            <dd>
              {(info.ram_available_mb / 1024).toFixed(1)} GB boş /{" "}
              {(info.ram_total_mb / 1024).toFixed(1)} GB
            </dd>
            <dt>Disk (veri dizini)</dt>
            <dd>{info.disk_free_gb} GB boş</dd>
            <dt>GPU backend</dt>
            <dd>{info.gpu_backend.toUpperCase()}</dd>
            {info.gpus.map((g) => (
              <SystemGpuRow key={g.index} name={g.name} used={g.vram_used_mb} total={g.vram_total_mb} />
            ))}
            <dt>Veri dizini</dt>
            <dd>{info.data_dir}</dd>
          </dl>
        )}
      </div>
    </section>
  );
}

function SystemGpuRow({ name, used, total }: { name: string; used: number; total: number }) {
  return (
    <>
      <dt>GPU</dt>
      <dd>
        {name} — VRAM {(used / 1024).toFixed(1)} / {(total / 1024).toFixed(1)} GB
      </dd>
    </>
  );
}

function ConsolePanel({ events }: { events: StudioEvent[] }) {
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [events]);
  return (
    <section className="panel">
      <header>Console — WS olay veriyolu</header>
      <div className="body console">
        {events.length === 0 && <p>Olay bekleniyor…</p>}
        {events.map((e, i) => (
          <div className="line" key={i}>
            <span className="ts">{new Date(e.ts).toLocaleTimeString()}</span>{" "}
            <span className="topic">[{e.topic}]</span> {JSON.stringify(e.data)}
          </div>
        ))}
        <div ref={endRef} />
      </div>
    </section>
  );
}

export default function App() {
  const [connected, setConnected] = useState(false);
  const [version, setVersion] = useState("");
  const [info, setInfo] = useState<SystemInfo | null>(null);
  const [events, setEvents] = useState<StudioEvent[]>([]);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let retry: number | undefined;
    let alive = true;

    const connect = () => {
      fetchHealth()
        .then((h) => {
          if (!alive) return;
          setVersion(h.version);
          setConnected(true);
          fetchSystemInfo().then(setInfo).catch(() => setInfo(null));
          ws = openEventSocket((e) =>
            setEvents((prev) => [...prev.slice(-499), e]),
          );
          ws.onclose = () => {
            if (!alive) return;
            setConnected(false);
            retry = window.setTimeout(connect, 2000);
          };
        })
        .catch(() => {
          if (!alive) return;
          setConnected(false);
          retry = window.setTimeout(connect, 2000);
        });
    };
    connect();
    return () => {
      alive = false;
      window.clearTimeout(retry);
      ws?.close();
    };
  }, []);

  return (
    <div className="app">
      <div className="titlebar">
        <h1>Local AI Studio</h1>
        {version && <span className="badge">v{version}</span>}
        <span className={`badge ${connected ? "ok" : "err"}`}>
          {connected ? "backend bağlı" : "backend bekleniyor"}
        </span>
      </div>
      <div className="panels">
        <SystemPanel info={info} />
        <ConsolePanel events={events} />
      </div>
    </div>
  );
}
