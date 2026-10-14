import { useState } from 'react'
import { api, ApiError } from '../api'
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
  const [currentPassword, setCurrentPassword] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [passwordMessage, setPasswordMessage] = useState('')

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
          <section className="settings-section">
            <h3 className="settings-h">Account</h3>
            <form onSubmit={async (e) => {
              e.preventDefault(); setPasswordMessage('')
              try { await api.auth.reset(currentPassword, newPassword); setCurrentPassword(''); setNewPassword(''); setPasswordMessage('Password changed. Other sessions have been signed out.') }
              catch (err) { setPasswordMessage(err instanceof ApiError ? err.message : 'Could not reset password') }
            }}>
              <label className="field">Current password <input type="password" autoComplete="current-password" value={currentPassword} onChange={e => setCurrentPassword(e.target.value)} required /></label>
              <label className="field">New password (16+ characters) <input type="password" autoComplete="new-password" minLength={16} value={newPassword} onChange={e => setNewPassword(e.target.value)} required /></label>
              <button className="btn" type="submit">Change password</button>
              {passwordMessage && <p role="status" className="muted small">{passwordMessage}</p>}
            </form>
            <button type="button" className="btn" onClick={() => void api.auth.logout().then(() => window.location.reload())}>Sign out</button>
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
