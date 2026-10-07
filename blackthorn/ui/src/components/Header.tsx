import type { GpuView } from '../gpu'
import { GpuButton, GpuPanel } from './GpuPanel'
import { MenuIcon, PlusIcon } from './Icons'

export function Header({ title, sidebarOpen, onToggleSidebar, onNew, gpu, gpuOpen, onToggleGpu, onCloseGpu, onPrompt }: {
  title: string
  sidebarOpen: boolean
  onToggleSidebar: () => void
  onNew: () => void
  gpu: GpuView
  gpuOpen: boolean
  onToggleGpu: () => void
  onCloseGpu: () => void
  onPrompt: () => void
}) {
  return (
    <header className="header" data-testid="header">
      <button type="button" className="icon-btn" onClick={onToggleSidebar} aria-label={sidebarOpen ? 'Close sidebar' : 'Open sidebar'} aria-expanded={sidebarOpen} aria-controls="sidebar" data-testid="sidebar-toggle"><MenuIcon /></button>
      <h1 className="title" data-testid="chat-title" title={title}>{title || 'New chat'}</h1>
      <div className="header-actions">
        <button type="button" className="btn ghost" onClick={onPrompt} data-testid="prompt-button" aria-label="System prompt" title="System prompt"><span className="long">System prompt</span><span className="short">Prompt</span></button>
        <div className="rel">
          <GpuButton gpu={gpu} open={gpuOpen} onToggle={onToggleGpu} />
          {gpuOpen && <GpuPanel gpu={gpu} onClose={onCloseGpu} />}
        </div>
        <button type="button" className="icon-btn only-narrow" onClick={onNew} aria-label="New chat" data-testid="new-chat-header"><PlusIcon /></button>
      </div>
    </header>
  )
}
