// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { SettingsDialog } from './Settings'
import { api } from '../api'

vi.mock('../api', async (importOriginal) => {
  const original = await importOriginal<typeof import('../api')>()
  return { ...original, api: { ...original.api, auth: { ...original.api.auth, reset: vi.fn() } } }
})
afterEach(() => { cleanup(); vi.clearAllMocks() })

const setup = () => render(<SettingsDialog theme="system" onTheme={vi.fn()} onClose={vi.fn()} onPromptSaved={vi.fn()} />)
const fill = (confirmation: string) => {
  fireEvent.change(screen.getByLabelText('Current password'), { target: { value: 'old-password-123456' } })
  fireEvent.change(screen.getByLabelText('New password (16+ characters)'), { target: { value: 'a-strong-new-password-123456' } })
  fireEvent.change(screen.getByLabelText('Confirm new password'), { target: { value: confirmation } })
  fireEvent.click(screen.getByRole('button', { name: 'Change password' }))
}

describe('password settings', () => {
  it('prevents mismatched confirmation without sending a credential', () => {
    setup(); fill('different-password-123456')
    expect(screen.getByRole('status').textContent).toContain('do not match')
    expect(api.auth.reset).not.toHaveBeenCalled()
  })
  it('changes the password and clears secret fields only after server success', async () => {
    vi.mocked(api.auth.reset).mockResolvedValue({ ok: true })
    setup(); fill('a-strong-new-password-123456')
    await waitFor(() => expect(api.auth.reset).toHaveBeenCalledWith('old-password-123456', 'a-strong-new-password-123456'))
    await waitFor(() => expect(screen.getByRole('status').textContent).toContain('Password changed'))
    expect((screen.getByLabelText('Current password') as HTMLInputElement).value).toBe('')
    expect((screen.getByLabelText('Confirm new password') as HTMLInputElement).value).toBe('')
  })
  it('keeps fields and reports errors when the server rejects the change', async () => {
    vi.mocked(api.auth.reset).mockRejectedValue(new Error('storage failed'))
    setup(); fill('a-strong-new-password-123456')
    await waitFor(() => expect(screen.getByRole('status').textContent).toContain('Could not change password'))
    expect((screen.getByLabelText('New password (16+ characters)') as HTMLInputElement).value).toBe('a-strong-new-password-123456')
  })
})
