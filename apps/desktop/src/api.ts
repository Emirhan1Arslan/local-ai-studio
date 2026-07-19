// REST /v1 istemcisi + WS olay tipleri (backend studio_core.events ile senkron tutulur).

// Tauri kabuğunda sidecar adresi/token'ı configureApi ile ayarlanır (state.tsx bootstrap).
// Tarayıcı dev modunda boş kalır: istekler göreli gider, vite proxy'si 8710'a yönlendirir.
let apiBase = "";
let apiToken = "";

export function configureApi(base: string, token: string): void {
  apiBase = base.replace(/\/+$/, "");
  apiToken = token;
}

/** Statik dosyalar (ör. /outputs/…) ve mutlak adres gereken yerler için tam URL. */
export function apiUrl(path: string): string {
  return apiBase + path;
}

function req(path: string, init?: RequestInit): Promise<Response> {
  const headers = new Headers(init?.headers);
  if (apiToken) headers.set("Authorization", `Bearer ${apiToken}`);
  return fetch(apiBase + path, { ...init, headers });
}

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
  const r = await req(`/v1/models${q}`);
  if (!r.ok) throw new Error(`models ${r.status}`);
  return r.json();
}

export async function rescanModels(): Promise<{ added: number; removed: number; updated: number; total: number }> {
  const r = await req("/v1/models/rescan", { method: "POST" });
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
  const r = await req(`/v1/jobs?limit=${limit}`);
  if (!r.ok) throw new Error(`jobs ${r.status}`);
  return r.json();
}

export async function createJob(
  kind: string,
  params: Record<string, unknown>,
  priority = 0,
): Promise<JobInfo> {
  const r = await req("/v1/jobs", {
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

export async function createJobBatch(
  kind: string,
  params: Record<string, unknown>,
  count: number,
  priority = 0,
): Promise<JobInfo[]> {
  const r = await req("/v1/jobs/batch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ kind, params, count, priority }),
  });
  if (!r.ok) {
    const body = await r.json().catch(() => null);
    throw new Error(body?.error?.message ?? `batch ${r.status}`);
  }
  return r.json();
}

export async function jobAction(
  id: string,
  action: "cancel" | "pause" | "resume",
): Promise<boolean> {
  const r = await req(`/v1/jobs/${id}/${action}`, { method: "POST" });
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
  meta: {
    params?: Record<string, unknown>;
    result?: Record<string, unknown>;
    [k: string]: unknown;
  };
  created_at: string;
}

export async function fetchAssets(kind?: string, limit = 24): Promise<AssetInfo[]> {
  const q = new URLSearchParams({ limit: String(limit) });
  if (kind) q.set("kind", kind);
  const r = await req(`/v1/assets?${q}`);
  if (!r.ok) throw new Error(`assets ${r.status}`);
  return r.json();
}

export interface UploadOut {
  path: string;
  size_bytes: number;
}

async function postUpload(filename: string, dataBase64: string): Promise<UploadOut> {
  const r = await req("/v1/uploads", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename, data_base64: dataBase64 }),
  });
  if (!r.ok) {
    const body = await r.json().catch(() => null);
    throw new Error(body?.error?.message ?? `upload ${r.status}`);
  }
  return r.json();
}

export async function uploadImageFile(file: File): Promise<UploadOut> {
  const dataUrl = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
  return postUpload(file.name, dataUrl.split(",", 2)[1] ?? "");
}

export async function uploadDataUrl(filename: string, dataUrl: string): Promise<UploadOut> {
  return postUpload(filename, dataUrl.split(",", 2)[1] ?? "");
}

export async function regenerateAsset(id: string): Promise<{ job_id: string }> {
  const r = await req(`/v1/assets/${id}/regenerate`, { method: "POST" });
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
  const r = await req("/v1/downloads");
  if (!r.ok) throw new Error(`downloads ${r.status}`);
  return r.json();
}

export async function createDownload(request: DownloadRequest): Promise<DownloadInfo> {
  const r = await req("/v1/downloads", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!r.ok) {
    const body = await r.json().catch(() => null);
    throw new Error(body?.error?.message ?? `download ${r.status}`);
  }
  return r.json();
}

export interface SnapshotRequest {
  repo_id: string;
  subdir: string;
  revision?: string;
}

export interface HfModelInfo {
  id: string;
  pipeline_tag: string;
  library_name: string;
  downloads: number;
  likes: number;
  license: string;
  gated: boolean;
}

export async function searchHfModels(
  q: string,
  pipelineTag = "",
  limit = 30,
): Promise<HfModelInfo[]> {
  const params = new URLSearchParams({ q, limit: String(limit) });
  if (pipelineTag) params.set("pipeline_tag", pipelineTag);
  const r = await req(`/v1/downloads/search?${params}`);
  if (!r.ok) {
    const body = await r.json().catch(() => null);
    throw new Error(body?.error?.message ?? `search ${r.status}`);
  }
  return r.json();
}

export async function fetchHfRepoFiles(repoId: string): Promise<string[]> {
  const r = await req(`/v1/downloads/files?${new URLSearchParams({ repo_id: repoId })}`);
  if (!r.ok) {
    const body = await r.json().catch(() => null);
    throw new Error(body?.error?.message ?? `files ${r.status}`);
  }
  return r.json();
}

export async function createSnapshotDownload(request: SnapshotRequest): Promise<DownloadInfo[]> {
  const r = await req("/v1/downloads/snapshot", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!r.ok) {
    const body = await r.json().catch(() => null);
    throw new Error(body?.error?.message ?? `snapshot ${r.status}`);
  }
  return r.json();
}

export async function downloadAction(
  id: string,
  action: "pause" | "resume" | "cancel",
): Promise<boolean> {
  const r = await req(`/v1/downloads/${id}/${action}`, { method: "POST" });
  if (!r.ok) return false;
  return (await r.json()).ok;
}

export async function fetchSystemInfo(): Promise<SystemInfo> {
  const r = await req("/v1/system/info");
  if (!r.ok) throw new Error(`system/info ${r.status}`);
  return r.json();
}

export async function fetchHealth(): Promise<{ status: string; version: string }> {
  const r = await req("/v1/health");
  if (!r.ok) throw new Error(`health ${r.status}`);
  return r.json();
}

export interface SettingsPaths {
  data_dir: string;
  models_dir: string;
  outputs_dir: string;
  cache_dir: string;
  logs_dir: string;
  envs_dir: string;
  uploads_dir: string;
  db_path: string;
}

export interface SettingsInfo {
  hf_token_set: boolean;
  telemetry_interval: number;
  worker_idle_timeout: number;
  worker_concurrency: number;
  paths: SettingsPaths;
}

export interface SettingsUpdate {
  hf_token?: string;
  telemetry_interval?: number;
  worker_idle_timeout?: number;
  worker_concurrency?: number;
}

export async function fetchSettings(): Promise<SettingsInfo> {
  const r = await req("/v1/settings");
  if (!r.ok) throw new Error(`settings ${r.status}`);
  return r.json();
}

export async function updateSettings(patch: SettingsUpdate): Promise<SettingsInfo> {
  const r = await req("/v1/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  if (!r.ok) {
    const body = await r.json().catch(() => null);
    throw new Error(body?.error?.message ?? `settings ${r.status}`);
  }
  return r.json();
}

export async function createDiagnosticsBundle(): Promise<{ path: string }> {
  const r = await req("/v1/settings/diagnostics", { method: "POST" });
  if (!r.ok) throw new Error(`diagnostics ${r.status}`);
  return r.json();
}

export function openEventSocket(onEvent: (e: StudioEvent) => void): WebSocket {
  // Tauri kabuğu: sidecar adresi ws şemasına çevrilir + token sorgusu; dev: göreli.
  const wsBase = apiBase
    ? apiBase.replace(/^http/, "ws")
    : `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}`;
  const q = apiToken ? `?token=${encodeURIComponent(apiToken)}` : "";
  const ws = new WebSocket(`${wsBase}/v1/ws${q}`);
  ws.onmessage = (msg) => {
    try {
      onEvent(JSON.parse(msg.data));
    } catch {
      // bozuk çerçeveyi yut — veriyolu şeması sürümlü, sessizce ilerle
    }
  };
  return ws;
}
