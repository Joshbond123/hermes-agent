import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router";
import "./chat.css";
import { btApi } from "./api";
import { Composer } from "./components/Composer";
import { Header } from "./components/Header";
import { MessageView, Welcome } from "./components/Messages";
import { Sidebar } from "./components/Sidebar";
import { SystemPromptDialog } from "./components/SystemPromptDialog";
import { Icon, Spinner, ToastProvider, type MenuItem } from "./components/ui";
import { useToast } from "./components/toastContext";
import { useAutoScroll } from "./hooks/useAutoScroll";
import { useChat } from "./hooks/useChat";
import { useGpu } from "./hooks/useGpu";
import { useMedia } from "./hooks/useMedia";
import { useSessions } from "./hooks/useSessions";
import { exportMarkdown } from "./store";
import type { VersionInfo } from "./types";

type Theme = "system" | "light" | "dark";
const THEME_KEY = "bt-theme";
const SIDEBAR_KEY = "bt-sidebar";

function readTheme(): Theme {
  try {
    const v = localStorage.getItem(THEME_KEY);
    return v === "light" || v === "dark" ? v : "system";
  } catch {
    return "system";
  }
}

function Shell() {
  const toast = useToast();
  const navigate = useNavigate();
  const { sessionId } = useParams<{ sessionId?: string }>();
  const isMobile = useMedia("(max-width: 860px)");
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(() => {
    try {
      return window.innerWidth > 860 && localStorage.getItem(SIDEBAR_KEY) !== "0";
    } catch {
      return window.innerWidth > 860;
    }
  });
  const [theme, setThemeState] = useState<Theme>(readTheme);
  const [gpuOpen, setGpuOpen] = useState(false);
  const [promptOpen, setPromptOpen] = useState(false);
  const [version, setVersion] = useState<VersionInfo | null>(null);
  const [prefill, setPrefill] = useState<{ text: string; nonce: number } | null>(null);
  const [focusSignal, setFocusSignal] = useState(0);

  const onError = useCallback((m: string) => toast(m, "error"), [toast]);
  const sessions = useSessions(onError);
  const gpu = useGpu(onError);
  const chat = useChat({
    routeSessionId: sessionId ?? null,
    onSessionAssigned: useCallback((id: string) => navigate(`/chat/${id}`, { replace: true }), [navigate]),
    onHistoryChanged: sessions.refresh,
    onError,
  });
  const { state } = chat;

  // keep the layout in step with the breakpoint (drawer on phones, docked sidebar on desktop)
  const wasMobile = useRef(isMobile);
  useEffect(() => {
    if (wasMobile.current !== isMobile) {
      setSidebarOpen(!isMobile);
      wasMobile.current = isMobile;
    }
  }, [isMobile]);

  const toggleSidebar = useCallback(() => {
    setSidebarOpen((v) => {
      const next = !v;
      if (!isMobile) {
        try {
          localStorage.setItem(SIDEBAR_KEY, next ? "1" : "0");
        } catch {
          /* private mode */
        }
      }
      return next;
    });
  }, [isMobile]);
  const closeDrawer = useCallback(() => {
    if (isMobile) setSidebarOpen(false);
  }, [isMobile]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isMobile && sidebarOpen) setSidebarOpen(false);
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [isMobile, sidebarOpen]);

  const setTheme = useCallback((t: Theme) => {
    setThemeState(t);
    try {
      localStorage.setItem(THEME_KEY, t);
    } catch {
      /* private mode */
    }
  }, []);

  useEffect(() => {
    btApi.version().then(setVersion).catch(() => undefined);
  }, []);

  const listed = sessions.sessions.find((s) => s.id === state.activeId) ?? sessions.archived.find((s) => s.id === state.activeId);
  const title = listed?.title || state.title || "New chat";
  useEffect(() => {
    document.title = state.activeId || state.messages.length ? `${title} · Blackthorn` : "Blackthorn";
  }, [title, state.activeId, state.messages.length]);

  const scroller = useRef<HTMLDivElement>(null);
  const userCount = state.messages.filter((m) => m.role === "user").length;
  const { away, jump } = useAutoScroll(scroller, [state.messages, state.turn.status, state.turn.reasoning.active], `${state.activeId}:${userCount}:${state.loading}`);

  const newChat = useCallback(() => {
    navigate("/chat");
    setFocusSignal((n) => n + 1);
    closeDrawer();
  }, [navigate, closeDrawer]);

  const startGpu = useCallback(() => {
    setGpuOpen(true);
    void gpu.turnOn();
  }, [gpu]);
  const openGpu = useCallback(() => setGpuOpen(true), []);
  const regenerate = useCallback(() => {
    let thinking = false;
    try {
      thinking = localStorage.getItem("bt-thinking") === "1";
    } catch {
      /* ignore */
    }
    void chat.regenerate({ thinking });
  }, [chat]);

  const doExport = useCallback(() => {
    const md = exportMarkdown(title, state.messages);
    const url = URL.createObjectURL(new Blob([md], { type: "text/markdown;charset=utf-8" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `${title.replace(/[^A-Za-z0-9._-]+/g, "_").slice(0, 60) || "chat"}.md`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }, [title, state.messages]);

  const menu = useMemo<MenuItem[]>(() => {
    const built = version ? `${version.commit.slice(0, 7)}${version.web_build?.built_at ? ` · ${new Date(version.web_build.built_at).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}` : ""}` : "";
    return [
      { key: "export", label: "Export chat as Markdown", icon: "download", onSelect: doExport, disabled: state.messages.length === 0 },
      { key: "admin", label: "Admin dashboard", icon: "external", onSelect: () => navigate("/sessions") },
      { key: "theme-system", label: "Theme: match system", icon: "sliders", checked: theme === "system", separatorBefore: true, onSelect: () => setTheme("system") },
      { key: "theme-light", label: "Theme: light", icon: "sun", checked: theme === "light", onSelect: () => setTheme("light") },
      { key: "theme-dark", label: "Theme: dark", icon: "moon", checked: theme === "dark", onSelect: () => setTheme("dark") },
      ...(built ? [{ key: "build", label: `Build ${built}`, separatorBefore: true, disabled: true, onSelect: () => undefined }] : []),
    ];
  }, [doExport, navigate, setTheme, state.messages.length, theme, version]);

  const last = state.messages.length - 1;
  const gpuState = gpu.gpu?.state;
  const showBanner = !gpu.unreachable && gpuState && gpuState !== "ready";

  return (
    <div className="bt-root bt-chat-root" data-theme={theme} data-mobile={isMobile ? "1" : "0"} data-sidebar={sidebarOpen ? "open" : "closed"}>
      {isMobile && sidebarOpen ? <div className="bt-backdrop" onClick={closeDrawer} aria-hidden="true" /> : null}
      <aside id="bt-sidebar" className="bt-sidebar" aria-label="Conversations" inert={!sidebarOpen}>
        <Sidebar
          api={sessions}
          activeId={state.activeId}
          onNew={newChat}
          onChosen={closeDrawer}
          onActiveGone={() => navigate("/chat")}
          onClose={() => setSidebarOpen(false)}
          showClose={isMobile}
        />
      </aside>

      <main className="bt-main">
        <Header
          title={title}
          sidebarOpen={sidebarOpen}
          onToggleSidebar={toggleSidebar}
          onNewChat={newChat}
          onOpenSystemPrompt={() => setPromptOpen(true)}
          gpu={gpu}
          gpuOpen={gpuOpen}
          onGpuOpenChange={setGpuOpen}
          menu={menu}
        />

        <div className="bt-scroll" ref={scroller}>
          <div className="bt-thread" role="log" aria-live="polite" aria-relevant="additions" aria-label="Conversation">
            {state.loading ? (
              <div className="bt-loading" aria-busy="true" aria-label="Loading conversation">
                {[92, 70, 84, 40].map((w, i) => <div key={i} className="bt-skeleton bt-skeleton-block" style={{ width: `${w}%` }} />)}
              </div>
            ) : state.loadError ? (
              <div className="bt-load-error" role="alert">
                <Icon name="alert" />
                <p>{state.loadError}</p>
                <div className="bt-error-actions">
                  <button type="button" className="bt-btn" onClick={chat.reload}>Try again</button>
                  <button type="button" className="bt-btn" onClick={newChat}>New chat</button>
                </div>
              </div>
            ) : state.messages.length === 0 ? (
              <Welcome gpu={gpu.gpu} unreachable={gpu.unreachable} onPick={(text) => setPrefill({ text, nonce: Date.now() })} onStartGpu={startGpu} onOpenGpu={openGpu} />
            ) : (
              state.messages.map((m, i) => (
                <MessageView
                  key={m.key}
                  message={m}
                  isLast={i === last}
                  turn={m.status === "streaming" ? state.turn : null}
                  gpu={gpu.gpu}
                  onRegenerate={regenerate}
                  onStartGpu={startGpu}
                  onOpenGpu={openGpu}
                />
              ))
            )}
          </div>
        </div>

        {away ? (
          <button type="button" className="bt-jump" onClick={() => jump(true)} aria-label="Jump to the latest message">
            <Icon name="arrowDown" size={16} />
          </button>
        ) : null}

        <div className="bt-dock">
          {showBanner ? (
            <div className={`bt-banner bt-banner-${gpuState}`} role="status">
              {gpuState === "starting" || gpuState === "stopping" ? <Spinner size={12} /> : <span className={`bt-dot bt-dot-gpu-${gpuState}`} aria-hidden="true" />}
              <span className="bt-banner-text">
                {gpuState === "off" ? "The GPU is off — start it to get answers."
                  : gpuState === "starting" ? `${gpu.gpu?.stage_label || "Starting the GPU"}…`
                  : gpuState === "stopping" ? "The GPU is stopping…"
                  : gpuState === "error" ? gpu.gpu?.error?.message || "The GPU has a problem."
                  : "Checking the GPU…"}
              </span>
              {gpuState === "off" ? <button type="button" className="bt-btn bt-btn-small" onClick={startGpu} disabled={gpu.acting}>Start GPU</button> : null}
              {gpuState === "starting" || gpuState === "error" ? <button type="button" className="bt-btn bt-btn-small" onClick={openGpu}>Details</button> : null}
            </div>
          ) : null}
          <Composer
            busy={chat.busy}
            stopping={state.turn.status === "stopping"}
            onSend={(text, opts) => void chat.send(text, opts)}
            onStop={() => void chat.stop()}
            prefill={prefill}
            focusSignal={focusSignal}
          />
        </div>
      </main>

      {promptOpen ? <SystemPromptDialog open onClose={() => setPromptOpen(false)} /> : null}
    </div>
  );
}

export default function ChatApp() {
  return (
    <ToastProvider>
      <Shell />
    </ToastProvider>
  );
}
