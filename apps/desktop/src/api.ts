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

export interface JobInfo {
  id: string;
  kind: string;
  status: string;
  priority: number;
  owner: string;
  project_id: string | null;
  params: Record<string, unknown>;
  result: Record<string, unknown>;
  error: string;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export async function fetchJobs(limit = 50): Promise<JobInfo[]> {
  const r = await fetch(`/v1/jobs?limit=${limit}`);
  if (!r.ok) throw new Error(`jobs ${r.status}`);
  return r.json();
}

export async function createJob(
  kind: string,
  params: Record<string, unknown>,
  priority = 0,
): Promise<JobInfo> {
  const r = await fetch("/v1/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ kind, params, priority }),
  });
  if (!r.ok) {
    const body = await r.json().catch(() => null);
    throw new Error(body?.error?.message ?? `job ${r.status}`);
  }
  return r.json();
}

export async function jobAction(
  id: string,
  action: "cancel" | "pause" | "resume",
): Promise<boolean> {
  const r = await fetch(`/v1/jobs/${id}/${action}`, { method: "POST" });
  if (!r.ok) return false;
  return (await r.json()).ok;
}

export interface AssetInfo {
  id: string;
  project_id: string;
  parent_id: string | null;
  kind: string;
  role: string;
  path: string;
  meta: { params?: Record<string, unknown>; [k: string]: unknown };
  created_at: string;
}

export async function fetchAssets(kind?: string, limit = 24): Promise<AssetInfo[]> {
  const q = new URLSearchParams({ limit: String(limit) });
  if (kind) q.set("kind", kind);
  const r = await fetch(`/v1/assets?${q}`);
  if (!r.ok) throw new Error(`assets ${r.status}`);
  return r.json();
}

export async function regenerateAsset(id: string): Promise<{ job_id: string }> {
  const r = await fetch(`/v1/assets/${id}/regenerate`, { method: "POST" });
  if (!r.ok) throw new Error(`regenerate ${r.status}`);
  return r.json();
}

export interface GpuTelemetry {
  index: number;
  name: string;
  util_percent: number;
  vram_used_mb: number;
  vram_total_mb: number;
}

export interface TelemetrySnapshot {
  cpu_percent: number;
  ram_used_mb: number;
  ram_total_mb: number;
  gpus: GpuTelemetry[];
}

export interface DownloadInfo {
  id: string;
  url: string;
  source: string;
  dest_path: string;
  status: string;
  total_bytes: number;
  downloaded_bytes: number;
  sha256_expected: string;
  sha256_actual: string;
  error: string;
  created_at: string;
  finished_at: string | null;
}

export interface DownloadRequest {
  url?: string;
  repo_id?: string;
  filename?: string;
  revision?: string;
  subdir: string;
  sha256?: string;
}

export async function fetchDownloads(): Promise<DownloadInfo[]> {
  const r = await fetch("/v1/downloads");
  if (!r.ok) throw new Error(`downloads ${r.status}`);
  return r.json();
}

export async function createDownload(req: DownloadRequest): Promise<DownloadInfo> {
  const r = await fetch("/v1/downloads", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!r.ok) {
    const body = await r.json().catch(() => null);
    throw new Error(body?.error?.message ?? `download ${r.status}`);
  }
  return r.json();
}

export async function downloadAction(
  id: string,
  action: "pause" | "resume" | "cancel",
): Promise<boolean> {
  const r = await fetch(`/v1/downloads/${id}/${action}`, { method: "POST" });
  if (!r.ok) return false;
  return (await r.json()).ok;
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
