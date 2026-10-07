import { useState } from 'react'
import { Dialog } from './Dialog'
import { SystemPromptDialog } from './SystemPrompt'

export function SettingsDialog({
  theme,
  onTheme,
  onClose,
  onPromptSaved,
}: {
  theme: 'system' | 'light' | 'dark'
  onTheme: (t: 'system' | 'light' | 'dark') => void
  onClose: () => void
  onPromptSaved: () => void
}) {
  const [promptOpen, setPromptOpen] = useState(false)

  return (
    <>
      <Dialog title="Settings" onClose={onClose} testId="settings-dialog">
        <div className="settings-body" data-testid="settings-body">
          <section className="settings-section">
            <h3 className="settings-h">Appearance</h3>
            <label className="field row">
              <span>Theme</span>
              <select
                value={theme}
                onChange={(e) => onTheme(e.target.value as 'system' | 'light' | 'dark')}
                data-testid="theme-select"
                aria-label="Theme"
              >
                <option value="system">System</option>
                <option value="dark">Dark</option>
                <option value="light">Light</option>
              </select>
            </label>
          </section>
          <section className="settings-section">
            <h3 className="settings-h">Agent</h3>
            <p className="muted small">System prompt shapes how Blackthorn answers. Changes apply to the next message.</p>
            <button
              type="button"
              className="btn"
              data-testid="open-system-prompt"
              onClick={() => setPromptOpen(true)}
            >
              Edit system prompt
            </button>
          </section>
        </div>
        <div className="dialog-actions">
          <span className="spacer" />
          <button type="button" className="btn primary" onClick={onClose} data-testid="settings-close">
            Done
          </button>
        </div>
      </Dialog>
      {promptOpen && (
        <SystemPromptDialog
          onClose={() => setPromptOpen(false)}
          onSaved={() => {
            onPromptSaved()
            setPromptOpen(false)
          }}
        />
      )}
    </>
  )
}
