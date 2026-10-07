import { useEffect, useState } from 'react'
import { api, ApiError } from '../api'
import { Dialog } from './Dialog'

const LIMIT = 8000

export function SystemPromptDialog({ onClose, onSaved }: { onClose: () => void; onSaved: () => void }) {
  const [value, setValue] = useState('')
  const [saved, setSaved] = useState('')
  const [phase, setPhase] = useState<'loading' | 'ready' | 'saving' | 'error'>('loading')
  const [error, setError] = useState<string | null>(null)
  const [justSaved, setJustSaved] = useState(false)

  useEffect(() => {
    let alive = true
    api.prompt.get().then((p) => { if (alive) { setValue(p); setSaved(p); setPhase('ready') } })
      .catch((e) => { if (alive) { setError(e instanceof ApiError ? e.message : 'Could not load the system prompt.'); setPhase('error') } })
    return () => { alive = false }
  }, [])

  const save = async () => {
    setPhase('saving'); setError(null)
    try {
      await api.prompt.put(value)
      setSaved(value); setPhase('ready'); setJustSaved(true); onSaved()
      setTimeout(() => setJustSaved(false), 2500)
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'Saving failed.'); setPhase('ready')
    }
  }
  const dirty = value !== saved
  return (
    <Dialog title="System prompt" onClose={onClose} testId="prompt-dialog" wide>
      <p className="dialog-text muted">Extra instructions added to every new response. Leave empty to use the built-in behaviour.</p>
      <textarea className="prompt-area" value={value} maxLength={LIMIT} disabled={phase === 'loading'} data-autofocus data-testid="prompt-input"
        placeholder={phase === 'loading' ? 'Loading…' : 'For example: Answer concisely and always show code in fenced blocks.'}
        onChange={(e) => { setValue(e.target.value); setJustSaved(false) }} />
      <div className="dialog-foot">
        <span className="muted small">{value.length.toLocaleString()} / {LIMIT.toLocaleString()}</span>
        {error && <span className="error-text" role="alert" data-testid="prompt-error">{error}</span>}
        {justSaved && <span className="ok-text" role="status" data-testid="prompt-saved">Saved. It applies to your next message.</span>}
      </div>
      <div className="dialog-actions">
        <button type="button" className="btn" onClick={() => setValue('')} disabled={!value || phase !== 'ready'}>Clear</button>
        <span className="spacer" />
        <button type="button" className="btn" onClick={onClose}>Close</button>
        <button type="button" className="btn primary" onClick={() => void save()} disabled={!dirty || phase !== 'ready'} data-testid="prompt-save">{phase === 'saving' ? 'Saving…' : 'Save'}</button>
      </div>
    </Dialog>
  )
}
