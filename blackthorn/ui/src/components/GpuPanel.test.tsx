// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { GpuPanel } from './GpuPanel'
import type { GpuStatus, GpuUsage, GpuView } from '../gpu'

afterEach(() => { cleanup(); vi.clearAllMocks() })

const status = (over: Partial<GpuStatus> = {}): GpuStatus => ({
  provider: 'Kaggle GPU', state: 'OFF', running: false, serving: false, base_url: '', phase: '',
  loaded_seconds: null, weights_source: '', reconnected: false, session_seconds: 0, last_activity: null,
  last_activity_reason: '', stop_requested: false, error: '', dataset: '', kaggle_user: 'Joshbond123',
  busy: '', settings: { idle_minutes: 15, idle_choices: [5, 10, 15, 30, 45, 60], weekly_hours: 30 },
  ...over,
})

const usage: GpuUsage = {
  hours_total: 30, hours_used: 1.25, hours_remaining: 28.75, percent_used: 4.2,
  current_session_seconds: 754, sessions_this_week: 2, week_label: '2026-11-07',
  usage_source: 'measured from observed GPU sessions',
  hours_basis: ("Kaggle's published minimum is 30 GPU hours per week (resets Saturday 00:00 UTC). "
    + "Kaggle exposes no usage API, so this allowance is a labelled setting you can correct to the "
    + "figure in your account; hours used are measured here."),
}

const view = (over: Partial<GpuView> = {}): GpuView => ({
  status: status(), usage, loading: false, busy: false, error: '',
  refresh: vi.fn(), start: vi.fn(async () => {}), stop: vi.fn(async () => {}),
  setTimer: vi.fn(async () => {}), install: vi.fn(async () => {}),
  ...over,
})

const open = (gpu: GpuView) => render(<GpuPanel gpu={gpu} open onToggle={vi.fn()} onClose={vi.fn()} />)

describe('the GPU control', () => {
  it('shows the real state and measured hours, and says where the allowance came from', () => {
    open(view())
    expect(screen.getByTestId('gpu-state').textContent).toBe('GPU off')
    expect(screen.getByTestId('gpu-hours-total').textContent).toBe('30.0 h')
    expect(screen.getByTestId('gpu-hours-used').textContent).toBe('1.25 h')
    expect(screen.getByTestId('gpu-hours-left').textContent).toBe('28.75 h')
    expect(screen.getByTestId('gpu-percent').textContent).toBe('4.2%')
    expect(screen.getByTestId('gpu-session-usage').textContent).toBe('12:34')
    // the panel must never imply Kaggle told us the quota
    expect(screen.getByTestId('gpu-basis').textContent).toContain('no usage API')
  })

  it('switches the GPU on with one press', () => {
    const gpu = view()
    open(gpu)
    fireEvent.click(screen.getByTestId('gpu-on'))
    expect(gpu.start).toHaveBeenCalledTimes(1)
  })

  it('cannot be switched off while a task is running, and says why', () => {
    const gpu = view({ status: status({ state: 'READY', running: true, serving: true, busy: '1 task(s) running',
      weights_source: 'persistent cache (Q4_K_M)' }) })
    open(gpu)
    expect(screen.getByTestId('gpu-state').textContent).toBe('GPU on')
    expect((screen.getByTestId('gpu-off') as HTMLButtonElement).disabled).toBe(true)
    expect(screen.getByTestId('gpu-busy').textContent).toContain('will not be switched off mid-task')
    expect(screen.getByTestId('gpu-serving').textContent).toContain('persistent cache')
  })

  it('offers exactly the idle timers requested, plus a custom value', () => {
    const gpu = view()
    open(gpu)
    const select = screen.getByTestId('gpu-timer') as HTMLSelectElement
    expect([...select.options].map((o) => Number(o.value))).toEqual([5, 10, 15, 30, 45, 60])
    fireEvent.change(select, { target: { value: '45' } })
    expect(gpu.setTimer).toHaveBeenCalledWith(45)
  })

  it('shows the reason when the GPU could not start', () => {
    open(view({ status: status({ state: 'ERROR', error: 'Kaggle did not bring the GPU up in time.' }) }))
    expect(screen.getByTestId('gpu-error').textContent).toContain('did not bring the GPU up')
  })

  it('offers the one-time install when Kaggle has no model yet', () => {
    const gpu = view({ status: status({ kernels: { install: 'error' } }) })
    open(gpu)
    fireEvent.click(screen.getByTestId('gpu-install'))
    expect(gpu.install).toHaveBeenCalledTimes(1)
  })
})
