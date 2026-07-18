// REST /v1 istemcisi + WS olay tipleri (backend studio_core.events ile senkron tutulur).

export interface GpuInfo {
  index: number;
  name: string;
  vram_total_mb: number;
  vram_used_mb: number;
  backend: string;
}

export interface SystemInfo {
  os: string;
  os_version: string;
  arch: string;
  python: string;
  cpu_count: number;
  ram_total_mb: number;
  ram_available_mb: number;
  disk_free_gb: number;
  gpus: GpuInfo[];
  gpu_backend: string;
  data_dir: string;
}

export interface StudioEvent {
  v: number;
  topic: string;
  ts: string;
  data: Record<string, unknown>;
}

export interface ModelInfo {
  id: string;
  name: string;
  family: string;
  kind: string;
  format: string;
  path: string;
  size_bytes: number;
  sha256: string;
  license: string;
  meta: Record<string, unknown>;
  created_at: string;
}

export async function fetchModels(kind?: string): Promise<ModelInfo[]> {
  const q = kind ? `?kind=${encodeURIComponent(kind)}` : "";
  const r = await fetch(`/v1/models${q}`);
  if (!r.ok) throw new Error(`models ${r.status}`);
  return r.json();
}

export async function rescanModels(): Promise<{ added: number; removed: number; updated: number; total: number }> {
  const r = await fetch("/v1/models/rescan", { method: "POST" });
  if (!r.ok) throw new Error(`rescan ${r.status}`);
  return r.json();
}

export async function fetchSystemInfo(): Promise<SystemInfo> {
  const r = await fetch("/v1/system/info");
  if (!r.ok) throw new Error(`system/info ${r.status}`);
  return r.json();
}

export async function fetchHealth(): Promise<{ status: string; version: string }> {
  const r = await fetch("/v1/health");
  if (!r.ok) throw new Error(`health ${r.status}`);
  return r.json();
}

export function openEventSocket(onEvent: (e: StudioEvent) => void): WebSocket {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/v1/ws`);
  ws.onmessage = (msg) => {
    try {
      onEvent(JSON.parse(msg.data));
    } catch {
      // bozuk çerçeveyi yut — veriyolu şeması sürümlü, sessizce ilerle
    }
  };
  return ws;
}
