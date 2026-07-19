import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import {
  createChatSession,
  deleteChatSession,
  fetchChatMessages,
  fetchChatSessions,
  fetchModels,
  sendChatMessage,
  type ChatMessageInfo,
  type ChatSessionInfo,
  type ModelInfo,
} from "../api";
import { useStudio } from "../state";

export default function ChatPanel() {
  const { events, connected } = useStudio();
  const [sessions, setSessions] = useState<ChatSessionInfo[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessageInfo[]>([]);
  const [ggufModels, setGgufModels] = useState<ModelInfo[]>([]);
  const [newSessionModel, setNewSessionModel] = useState("");
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const [streamingJobId, setStreamingJobId] = useState<string | null>(null);
  const [streamingText, setStreamingText] = useState("");
  const [error, setError] = useState("");
  const lastEvent = useRef(0);
  const scrollRef = useRef<HTMLDivElement>(null);

  const refreshSessions = () => fetchChatSessions().then(setSessions).catch(() => undefined);

  useEffect(() => {
    if (!connected) return;
    fetchModels("llm")
      .then((all) => {
        const gguf = all.filter((m) => m.format === "gguf");
        setGgufModels(gguf);
        if (gguf.length > 0 && !newSessionModel) setNewSessionModel(gguf[0].path);
      })
      .catch(() => setGgufModels([]));
    refreshSessions();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [connected]);

  useEffect(() => {
    if (!activeId) {
      setMessages([]);
      return;
    }
    fetchChatMessages(activeId).then(setMessages).catch(() => setMessages([]));
  }, [activeId]);

  useEffect(() => {
    let sessionsNeedRefresh = false;
    for (let i = lastEvent.current; i < events.length; i++) {
      const e = events[i];
      if (e.topic === "job.progress") {
        const data = e.data as { id: string; delta?: string };
        if (data.id === streamingJobId && data.delta) {
          setStreamingText((prev) => prev + data.delta);
        }
      } else if (e.topic === "job.done" || e.topic === "job.failed") {
        const data = e.data as { id: string; error?: string };
        if (data.id === streamingJobId) {
          setStreamingJobId(null);
          setStreamingText("");
          if (e.topic === "job.failed") setError(data.error ?? "üretim başarısız oldu");
          if (activeId) fetchChatMessages(activeId).then(setMessages).catch(() => undefined);
          sessionsNeedRefresh = true;
        }
      }
    }
    lastEvent.current = events.length;
    if (sessionsNeedRefresh) refreshSessions();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [events]);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingText]);

  const createSession = async () => {
    setError("");
    if (!newSessionModel) {
      setError("bir GGUF model seçin (Model Browser'dan / Keşfet'ten indirin)");
      return;
    }
    try {
      const s = await createChatSession(newSessionModel);
      setSessions((prev) => [s, ...prev]);
      setActiveId(s.id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const removeSession = async (id: string) => {
    await deleteChatSession(id);
    setSessions((prev) => prev.filter((s) => s.id !== id));
    if (activeId === id) setActiveId(null);
  };

  const send = async () => {
    if (!activeId || !draft.trim() || streamingJobId) return;
    setError("");
    const text = draft;
    setDraft("");
    setSending(true);
    try {
      const { job_id } = await sendChatMessage(activeId, text);
      setStreamingJobId(job_id);
      setStreamingText("");
      const fresh = await fetchChatMessages(activeId);
      setMessages(fresh);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setDraft(text);
    } finally {
      setSending(false);
    }
  };

  const activeSession = sessions.find((s) => s.id === activeId);

  return (
    <div className="chat-layout">
      <div className="chat-sidebar">
        <div className="dl-row">
          <select value={newSessionModel} onChange={(e) => setNewSessionModel(e.target.value)}>
            <option value="" disabled>
              {ggufModels.length === 0 ? "GGUF model bulunamadı" : "model seç"}
            </option>
            {ggufModels.map((m) => (
              <option key={m.id} value={m.path}>
                {m.name}
              </option>
            ))}
          </select>
        </div>
        <button type="button" onClick={createSession} disabled={!connected}>
          + yeni sohbet
        </button>
        <div className="chat-session-list">
          {sessions.map((s) => (
            <div
              key={s.id}
              className={`chat-session-item ${s.id === activeId ? "active" : ""}`}
              onClick={() => setActiveId(s.id)}
            >
              <span className="chat-session-title">{s.title}</span>
              <button
                type="button"
                className="chat-session-del"
                onClick={(e) => {
                  e.stopPropagation();
                  removeSession(s.id);
                }}
              >
                ✕
              </button>
            </div>
          ))}
          {sessions.length === 0 && <p className="muted">Henüz sohbet yok.</p>}
        </div>
      </div>

      <div className="chat-main">
        {!activeSession ? (
          <p className="muted">Sol taraftan bir sohbet seçin ya da yeni bir tane başlatın.</p>
        ) : (
          <>
            <div className="chat-thread">
              {messages.map((m) => (
                <div key={m.id} className={`chat-bubble chat-${m.role}`}>
                  <ReactMarkdown>{m.content}</ReactMarkdown>
                </div>
              ))}
              {streamingJobId && (
                <div className="chat-bubble chat-assistant chat-streaming">
                  <ReactMarkdown>{streamingText || "…"}</ReactMarkdown>
                </div>
              )}
              <div ref={scrollRef} />
            </div>
            <form
              className="chat-input-row"
              onSubmit={(e) => {
                e.preventDefault();
                send();
              }}
            >
              <textarea
                placeholder="mesaj yaz… (Enter: gönder, Shift+Enter: yeni satır)"
                value={draft}
                onChange={(ev) => setDraft(ev.target.value)}
                onKeyDown={(ev) => {
                  if (ev.key === "Enter" && !ev.shiftKey) {
                    ev.preventDefault();
                    send();
                  }
                }}
                rows={2}
              />
              <button type="submit" disabled={!draft.trim() || sending || !!streamingJobId}>
                {streamingJobId ? "yanıtlanıyor…" : "gönder"}
              </button>
            </form>
          </>
        )}
        {error && <p className="err-text">{error}</p>}
      </div>
    </div>
  );
}
