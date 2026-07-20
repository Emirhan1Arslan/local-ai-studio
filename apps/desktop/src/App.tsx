import {
  DockviewReact,
  themeDark,
  type DockviewApi,
  type DockviewReadyEvent,
  type IDockviewPanelProps,
} from "dockview-react";
import "dockview-react/dist/styles/dockview.css";
import AudioPanel from "./panels/AudioPanel";
import ChatPanel from "./panels/ChatPanel";
import ConsolePanel from "./panels/ConsolePanel";
import DownloadsPanel from "./panels/DownloadsPanel";
import ExplorePanel from "./panels/ExplorePanel";
import GeneratePanel from "./panels/GeneratePanel";
import MeshPanel from "./panels/MeshPanel";
import ModelsPanel from "./panels/ModelsPanel";
import QueuePanel from "./panels/QueuePanel";
import SettingsPanel from "./panels/SettingsPanel";
import SystemPanel from "./panels/SystemPanel";
import ToolsPanel from "./panels/ToolsPanel";
import VideoPanel from "./panels/VideoPanel";
import VramPanel from "./panels/VramPanel";
import { useStudio } from "./state";

const LAYOUT_KEY = "lais.layout.v1";

const components: Record<string, React.FC<IDockviewPanelProps>> = {
  system: SystemPanel,
  console: ConsolePanel,
  models: ModelsPanel,
  downloads: DownloadsPanel,
  explore: ExplorePanel,
  queue: QueuePanel,
  vram: VramPanel,
  generate: GeneratePanel,
  settings: SettingsPanel,
  tools: ToolsPanel,
  chat: ChatPanel,
  audio: AudioPanel,
  video: VideoPanel,
  mesh: MeshPanel,
};

const EXTRA_PANELS: { id: string; title: string; near: string }[] = [
  { id: "generate", title: "Üretim", near: "models" },
  { id: "tools", title: "Araçlar", near: "generate" },
  { id: "chat", title: "Sohbet", near: "generate" },
  { id: "audio", title: "Ses", near: "chat" },
  { id: "video", title: "Video", near: "audio" },
  { id: "mesh", title: "3D", near: "video" },
  { id: "downloads", title: "İndirmeler", near: "models" },
  { id: "explore", title: "Keşfet", near: "downloads" },
  { id: "queue", title: "Kuyruk", near: "models" },
  { id: "vram", title: "VRAM", near: "system" },
  { id: "settings", title: "Ayarlar", near: "system" },
];

function addDefaultPanels(api: DockviewApi) {
  api.addPanel({ id: "system", component: "system", title: "Sistem" });
  api.addPanel({
    id: "models",
    component: "models",
    title: "Modeller",
    position: { referencePanel: "system", direction: "right" },
  });
  api.addPanel({
    id: "console",
    component: "console",
    title: "Console",
    position: { referencePanel: "system", direction: "below" },
  });
  ensurePanels(api);
}

/** Kayıtlı yerleşim eski sürümden kalmışsa sonradan eklenen panelleri tamamla. */
function ensurePanels(api: DockviewApi) {
  for (const p of EXTRA_PANELS) {
    if (api.getPanel(p.id)) continue;
    api.addPanel({
      id: p.id,
      component: p.id,
      title: p.title,
      position: api.getPanel(p.near) ? { referencePanel: p.near, direction: "within" } : undefined,
    });
  }
}

function onReady(event: DockviewReadyEvent) {
  const api = event.api;
  const saved = localStorage.getItem(LAYOUT_KEY);
  let restored = false;
  if (saved) {
    try {
      api.fromJSON(JSON.parse(saved));
      restored = true;
    } catch {
      localStorage.removeItem(LAYOUT_KEY);
    }
  }
  if (!restored || api.panels.length === 0) addDefaultPanels(api);
  else ensurePanels(api);
  api.onDidLayoutChange(() => {
    localStorage.setItem(LAYOUT_KEY, JSON.stringify(api.toJSON()));
  });
}

export default function App() {
  const { connected, version } = useStudio();
  return (
    <div className="app">
      <div className="titlebar">
        <h1>Local AI Studio</h1>
        {version && <span className="badge">v{version}</span>}
        <span className={`badge ${connected ? "ok" : "err"}`}>
          {connected ? "backend bağlı" : "backend bekleniyor"}
        </span>
      </div>
      <div className="dock-wrap">
        <DockviewReact components={components} onReady={onReady} theme={themeDark} />
      </div>
    </div>
  );
}
