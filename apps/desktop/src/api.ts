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

/** Ses dosyası/mikrofon kaydı yükleme — postUpload aynı, yalnızca isimlendirme netliği için. */
export async function uploadAudioFile(file: File | Blob, filename = "kayit.webm"): Promise<UploadOut> {
  const dataUrl = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
  const name = file instanceof File ? file.name : filename;
  return postUpload(name, dataUrl.split(",", 2)[1] ?? "");
}

/** Dışarıdan getirilen video dosyası yükleme (videotool düzenleme için) — uygulamanın
 * kendi ürettiği videolar zaten yerel yol olduğu için bu uçtan geçmez. */
export async function uploadVideoFile(file: File): Promise<UploadOut> {
  const dataUrl = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
  return postUpload(file.name, dataUrl.split(",", 2)[1] ?? "");
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
  only_patterns?: string[];
}

export interface HfModelInfo {
  id: string;
  pipeline_tag: string;
  library_name: string;
  downloads: number;
  likes: number;
  license: string;
  gated: boolean;
  size_bytes: number;
}

export interface HfFileInfo {
  filename: string;
  size: number;
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

export async function fetchHfRepoFiles(repoId: string): Promise<HfFileInfo[]> {
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
  download_speed_limit_mbps: number;
  paths: SettingsPaths;
}

export interface SettingsUpdate {
  hf_token?: string;
  telemetry_interval?: number;
  worker_idle_timeout?: number;
  worker_concurrency?: number;
  download_speed_limit_mbps?: number;
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

export interface ChatSessionInfo {
  id: string;
  title: string;
  model_path: string;
  system_prompt: string;
  created_at: string;
  updated_at: string;
}

export interface ChatMessageInfo {
  id: string;
  role: string;
  content: string;
  created_at: string;
}

export async function fetchChatSessions(): Promise<ChatSessionInfo[]> {
  const r = await req("/v1/chat");
  if (!r.ok) throw new Error(`chat sessions ${r.status}`);
  return r.json();
}

export async function createChatSession(
  modelPath: string,
  title = "Yeni sohbet",
  systemPrompt = "",
): Promise<ChatSessionInfo> {
  const r = await req("/v1/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_path: modelPath, title, system_prompt: systemPrompt }),
  });
  if (!r.ok) {
    const body = await r.json().catch(() => null);
    throw new Error(body?.error?.message ?? `chat session ${r.status}`);
  }
  return r.json();
}

export async function deleteChatSession(id: string): Promise<boolean> {
  const r = await req(`/v1/chat/${id}`, { method: "DELETE" });
  if (!r.ok) return false;
  return (await r.json()).ok;
}

export async function fetchChatMessages(sessionId: string): Promise<ChatMessageInfo[]> {
  const r = await req(`/v1/chat/${sessionId}/messages`);
  if (!r.ok) throw new Error(`chat messages ${r.status}`);
  return r.json();
}

export async function sendChatMessage(
  sessionId: string,
  content: string,
  temperature = 0.7,
  maxTokens = 1024,
): Promise<{ job_id: string; user_message_id: string }> {
  const r = await req(`/v1/chat/${sessionId}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content, temperature, max_tokens: maxTokens }),
  });
  if (!r.ok) {
    const body = await r.json().catch(() => null);
    throw new Error(body?.error?.message ?? `send message ${r.status}`);
  }
  return r.json();
}

export interface PortSpecInfo {
  name: string;
  type: string;
  required: boolean;
}

export interface ParamSpecInfo {
  name: string;
  type: string;
  default: unknown;
  choices: string[] | null;
}

export interface NodeTypeInfo {
  type_id: string;
  category: string;
  label: string;
  inputs: PortSpecInfo[];
  outputs: PortSpecInfo[];
  params: ParamSpecInfo[];
}

export async function fetchNodeTypes(): Promise<NodeTypeInfo[]> {
  const r = await req("/v1/workflows/node-types");
  if (!r.ok) throw new Error(`node-types ${r.status}`);
  return r.json();
}

export interface WorkflowNodeData {
  id: string;
  type: string;
  params: Record<string, unknown>;
  ui?: Record<string, unknown>;
}

export interface WorkflowEdgeData {
  id: string;
  from_node: string;
  from_port: string;
  to_node: string;
  to_port: string;
}

export interface WorkflowGraphData {
  nodes: WorkflowNodeData[];
  edges: WorkflowEdgeData[];
}

export interface WorkflowInfo {
  id: string;
  name: string;
  project_id: string | null;
  graph: WorkflowGraphData;
  created_at: string;
  updated_at: string;
}

export async function fetchWorkflows(): Promise<WorkflowInfo[]> {
  const r = await req("/v1/workflows");
  if (!r.ok) throw new Error(`workflows ${r.status}`);
  return r.json();
}

export async function fetchWorkflow(id: string): Promise<WorkflowInfo> {
  const r = await req(`/v1/workflows/${id}`);
  if (!r.ok) throw new Error(`workflow ${r.status}`);
  return r.json();
}

async function postOrPutWorkflow(
  method: "POST" | "PUT",
  path: string,
  name: string,
  graph: WorkflowGraphData,
): Promise<WorkflowInfo> {
  const r = await req(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, graph }),
  });
  if (!r.ok) {
    const body = await r.json().catch(() => null);
    throw new Error(body?.error?.message ?? `workflow ${r.status}`);
  }
  return r.json();
}

export async function createWorkflow(name: string, graph: WorkflowGraphData): Promise<WorkflowInfo> {
  return postOrPutWorkflow("POST", "/v1/workflows", name, graph);
}

export async function updateWorkflow(
  id: string,
  name: string,
  graph: WorkflowGraphData,
): Promise<WorkflowInfo> {
  return postOrPutWorkflow("PUT", `/v1/workflows/${id}`, name, graph);
}

export async function deleteWorkflow(id: string): Promise<boolean> {
  const r = await req(`/v1/workflows/${id}`, { method: "DELETE" });
  if (!r.ok) return false;
  return (await r.json()).ok;
}

export interface WorkflowNodeResult {
  status: "done" | "failed" | "skipped";
  outputs: Record<string, unknown>;
  error: string;
  from_cache: boolean;
}

export interface WorkflowRunInfo {
  id: string;
  workflow_id: string;
  status: "running" | "done" | "failed";
  seed: number;
  node_results: Record<string, WorkflowNodeResult>;
  error: string;
  created_at: string;
  finished_at: string | null;
}

export async function runWorkflow(id: string, seed = 0): Promise<WorkflowRunInfo> {
  const r = await req(`/v1/workflows/${id}/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ seed }),
  });
  if (!r.ok) {
    const body = await r.json().catch(() => null);
    throw new Error(body?.error?.message ?? `run ${r.status}`);
  }
  return r.json();
}

export async function fetchWorkflowRun(runId: string): Promise<WorkflowRunInfo> {
  const r = await req(`/v1/workflows/runs/${runId}`);
  if (!r.ok) throw new Error(`workflow run ${r.status}`);
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
