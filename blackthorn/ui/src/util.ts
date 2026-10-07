/** Small shared helpers. */

export function decodeEntities(text: string): string {
  return text.replace(/&(#x?[0-9a-f]+|amp|lt|gt|quot|apos|nbsp);/gi, (m, e: string) => {
    const lower = e.toLowerCase()
    if (lower === 'amp') return '&'
    if (lower === 'lt') return '<'
    if (lower === 'gt') return '>'
    if (lower === 'quot') return '"'
    if (lower === 'apos') return "'"
    if (lower === 'nbsp') return '\u00a0'
    const code = lower.startsWith('#x') ? parseInt(lower.slice(2), 16) : parseInt(lower.slice(1), 10)
    return Number.isFinite(code) && code > 0 && code < 0x110000 ? String.fromCodePoint(code) : m
  })
}

export function safeHref(href: string | null | undefined): string | null {
  const value = (href ?? '').trim()
  if (/^(https?:|mailto:)/i.test(value)) return value
  return null
}

export async function copyText(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text)
      return true
    }
  } catch {
    /* fall through to the legacy path */
  }
  try {
    const area = document.createElement('textarea')
    area.value = text
    area.setAttribute('readonly', '')
    area.style.cssText = 'position:fixed;top:0;left:0;opacity:0;pointer-events:none'
    document.body.appendChild(area)
    area.select()
    const ok = document.execCommand('copy')
    area.remove()
    return ok
  } catch {
    return false
  }
}

export function formatDuration(ms: number | undefined | null): string {
  if (ms == null) return ''
  if (ms < 1000) return `${Math.round(ms)} ms`
  const s = ms / 1000
  if (s < 60) return `${s.toFixed(s < 10 ? 1 : 0)} s`
  return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`
}

export function formatElapsed(totalSeconds: number): string {
  const s = Math.max(0, Math.round(totalSeconds))
  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, '0')}s`
}

export function formatBytes(n: number): string {
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(0)} KB`
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(0)} MB`
  return `${(n / 1024 ** 3).toFixed(2)} GB`
}

export function groupLabel(updatedAt: number | undefined, now = Date.now()): string {
  if (!updatedAt) return 'Older'
  const days = Math.floor((now / 1000 - updatedAt) / 86400)
  if (days < 1) return 'Today'
  if (days < 2) return 'Yesterday'
  if (days < 8) return 'Previous 7 days'
  if (days < 31) return 'Previous 30 days'
  return 'Older'
}

export const uid = (prefix: string) => `${prefix}${Math.random().toString(36).slice(2, 10)}${Date.now().toString(36).slice(-4)}`
