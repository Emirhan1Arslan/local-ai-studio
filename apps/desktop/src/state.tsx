// Backend bağlantı durumu + olay akışı: tüm dock panellerinin ortak veri kaynağı.
import {
  createContext,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import {
  fetchHealth,
  fetchSystemInfo,
  openEventSocket,
  type StudioEvent,
  type SystemInfo,
} from "./api";

interface StudioState {
  connected: boolean;
  version: string;
  info: SystemInfo | null;
  events: StudioEvent[];
  /** model.* olayı geldikçe artar; Model Browser bunu izleyip yeniden yükler. */
  modelsVersion: number;
}

const Ctx = createContext<StudioState>({
  connected: false,
  version: "",
  info: null,
  events: [],
  modelsVersion: 0,
});

export function useStudio(): StudioState {
  return useContext(Ctx);
}

export function StudioProvider({ children }: { children: ReactNode }) {
  const [connected, setConnected] = useState(false);
  const [version, setVersion] = useState("");
  const [info, setInfo] = useState<SystemInfo | null>(null);
  const [events, setEvents] = useState<StudioEvent[]>([]);
  const [modelsVersion, setModelsVersion] = useState(0);

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
          setModelsVersion((v) => v + 1); // yeniden bağlanınca listeyi tazele
          ws = openEventSocket((e) => {
            setEvents((prev) => [...prev.slice(-499), e]);
            if (e.topic.startsWith("model.")) setModelsVersion((v) => v + 1);
          });
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
    <Ctx.Provider value={{ connected, version, info, events, modelsVersion }}>
      {children}
    </Ctx.Provider>
  );
}
