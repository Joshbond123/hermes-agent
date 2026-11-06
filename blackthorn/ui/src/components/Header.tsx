import type { CodespacesView } from '../codespaces'
import type { ModelsView } from '../models'
import { CodespacesButton, CodespacesPanel } from './CodespacesPanel'
import { MenuIcon, PlusIcon } from './Icons'
import { ModelPicker } from './ModelPicker'

export function Header({ title, sidebarOpen, onToggleSidebar, onNew, codespaces, csOpen, onToggleCodespaces, onCloseCodespaces,
  models, modelOpen, onToggleModels, onChooseModel, onCloseModels }: {
  title: string
  sidebarOpen: boolean
  onToggleSidebar: () => void
  onNew: () => void
  codespaces: CodespacesView
  csOpen: boolean
  onToggleCodespaces: () => void
  onCloseCodespaces: () => void
  models: ModelsView
  modelOpen: boolean
  onToggleModels: () => void
  onChooseModel: (id: string) => void
  onCloseModels: () => void
}) {
  return (
    <header className="header" data-testid="header">
      <button type="button" className="icon-btn" onClick={onToggleSidebar} aria-label={sidebarOpen ? 'Close sidebar' : 'Open sidebar'} aria-expanded={sidebarOpen} aria-controls="sidebar" data-testid="sidebar-toggle"><MenuIcon /></button>
      <h1 className="title" data-testid="chat-title" title={title}>{title || 'New chat'}</h1>
      <div className="header-actions">
        <ModelPicker models={models} open={modelOpen} onToggle={onToggleModels} onChoose={onChooseModel} onClose={onCloseModels} />
        <div className="rel">
          <CodespacesButton codespaces={codespaces} open={csOpen} onToggle={onToggleCodespaces} />
          {csOpen && <CodespacesPanel codespaces={codespaces} onClose={onCloseCodespaces} />}
        </div>
        <button type="button" className="icon-btn only-narrow" onClick={onNew} aria-label="New chat" data-testid="new-chat-header"><PlusIcon /></button>
      </div>
    </header>
  )
}
