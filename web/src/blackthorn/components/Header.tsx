import { useRef } from "react";
import type { GpuApi } from "../hooks/useGpu";
import { gpuDotClass, gpuLabel } from "../gpuFormat";
import { GpuPanel } from "./GpuPanel";
import { Icon, MenuButton, Popover, type MenuItem } from "./ui";

interface Props {
  title: string;
  sidebarOpen: boolean;
  onToggleSidebar: () => void;
  onNewChat: () => void;
  onOpenSystemPrompt: () => void;
  gpu: GpuApi;
  gpuOpen: boolean;
  onGpuOpenChange: (open: boolean) => void;
  menu: MenuItem[];
}

export function Header({ title, sidebarOpen, onToggleSidebar, onNewChat, onOpenSystemPrompt, gpu, gpuOpen, onGpuOpenChange, menu }: Props) {
  const gpuBtn = useRef<HTMLButtonElement>(null);
  return (
    <header className="bt-header">
      <button
        type="button"
        className="bt-icon-btn bt-sidebar-toggle"
        aria-label={sidebarOpen ? "Close sidebar" : "Open sidebar"}
        aria-expanded={sidebarOpen}
        aria-controls="bt-sidebar"
        onClick={onToggleSidebar}
      >
        <Icon name="panel" />
      </button>
      {!sidebarOpen ? (
        <button type="button" className="bt-icon-btn" aria-label="New chat" title="New chat" onClick={onNewChat}>
          <Icon name="plus" />
        </button>
      ) : null}
      <h1 className="bt-title" title={title}>{title}</h1>

      <div className="bt-header-actions">
        <button type="button" className="bt-hbtn" onClick={onOpenSystemPrompt} aria-label="System prompt" title="System prompt">
          <Icon name="sliders" size={16} />
          <span className="bt-hide-sm">System prompt</span>
        </button>
        <button
          ref={gpuBtn}
          type="button"
          className={`bt-hbtn bt-gpu-btn bt-gpu-${gpu.gpu?.state ?? "unknown"}`}
          aria-haspopup="dialog"
          aria-expanded={gpuOpen}
          aria-label={`GPU: ${gpuLabel(gpu.gpu, gpu.unreachable)}. Open GPU controls`}
          title="GPU controls"
          onClick={() => onGpuOpenChange(!gpuOpen)}
        >
          <span className={gpuDotClass(gpu.gpu, gpu.unreachable)} aria-hidden="true" />
          <span className="bt-gpu-label">{gpuLabel(gpu.gpu, gpu.unreachable)}</span>
        </button>
        <Popover open={gpuOpen} onClose={() => onGpuOpenChange(false)} anchorRef={gpuBtn} label="GPU controls" className="bt-gpu-popover" width={380}>
          <GpuPanel api={gpu} onClose={() => onGpuOpenChange(false)} />
        </Popover>
        <MenuButton items={menu} label="More options" className="bt-icon-btn" icon="dots" />
      </div>
    </header>
  );
}
