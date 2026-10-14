import { useEffect, useState, type ReactNode } from 'react'
import { api, ApiError } from './api'

export function LoginGate({ children }: { children: ReactNode }) {
  const [ready, setReady] = useState<boolean | null>(null)
  const [configured, setConfigured] = useState(true)
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  useEffect(() => { void api.auth.status().then(s => { setReady(s.authenticated); setConfigured(s.configured) }).catch(() => setError('Cannot verify login; retry later.')) }, [])
  if (ready) return <>{children}</>
  return <main className="login-page" data-testid="login-page">
    <form className="login-card" onSubmit={async e => {
      e.preventDefault(); setBusy(true); setError('')
      try { await api.auth.login(password); setPassword(''); setReady(true) }
      catch (err) { setError(err instanceof ApiError ? err.message : 'Login failed') }
      finally { setBusy(false) }
    }}>
      <h1>Blackthorn</h1>
      <p className="muted">{ready === null ? 'Checking access…' : configured ? 'Enter your owner password to continue.' : 'Login has not been configured on the server.'}</p>
      {ready === false && configured && <><label htmlFor="owner-password">Password</label>
        <input id="owner-password" autoComplete="current-password" type="password" required value={password} onChange={e => setPassword(e.target.value)} />
        <button className="btn primary" disabled={busy} type="submit">{busy ? 'Signing in…' : 'Sign in'}</button></>}
      {error && <p className="notice error" role="alert">{error}</p>}
    </form>
  </main>
}
