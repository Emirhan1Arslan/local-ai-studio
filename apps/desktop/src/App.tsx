import {
  DockviewReact,
  themeDark,
  type DockviewApi,
  type DockviewReadyEvent,
  type IDockviewPanelProps,
} from "dockview-react";
import "dockview-react/dist/styles/dockview.css";
import ConsolePanel from "./panels/ConsolePanel";
import ModelsPanel from "./panels/ModelsPanel";
import SystemPanel from "./panels/SystemPanel";
import { useStudio } from "./state";

const LAYOUT_KEY = "lais.layout.v1";

const components: Record<string, React.FC<IDockviewPanelProps>> = {
  system: SystemPanel,
  console: ConsolePanel,
  models: ModelsPanel,
};

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
