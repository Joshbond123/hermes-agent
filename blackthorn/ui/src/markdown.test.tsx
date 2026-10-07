import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { Markdown } from './Markdown'
import { decodeEntities, formatElapsed, groupLabel, safeHref } from './util'

afterEach(cleanup)

describe('Markdown', () => {
  it('renders structure: headings, lists, tables, inline code, emphasis', () => {
    const { container } = render(<Markdown text={'# Title\n\n- one\n- **two**\n\n1. a\n2. b\n\n| h1 | h2 |\n|---|---|\n| x | `y` |\n'} />)
    expect(container.querySelector('h1')?.textContent).toBe('Title')
    expect(container.querySelectorAll('ul li')).toHaveLength(2)
    expect(container.querySelector('strong')?.textContent).toBe('two')
    expect(container.querySelectorAll('ol li')).toHaveLength(2)
    expect(container.querySelector('.table-wrap table')).not.toBeNull()
    expect(container.querySelector('td .inline-code')?.textContent).toBe('y')
  })

  it('renders a fenced block with a language label and syntax highlighting, and scrolls horizontally', () => {
    const { container } = render(<Markdown text={'```python\ndef greet(name):\n    return "hello " + name  # ' + 'x'.repeat(300) + '\n```'} />)
    expect(screen.getByTestId('code-lang').textContent).toBe('python')
    expect(container.querySelector('pre code .hljs-keyword')?.textContent).toBe('def')
    expect(container.querySelector('pre code .hljs-string')).not.toBeNull()
    expect(container.querySelector('pre')?.className ?? '').toBe('')   // overflow is on .code pre via CSS (overflow-x: auto)
  })

  it('renders an UNTERMINATED fence progressively as a code block (streaming)', () => {
    const { container, rerender } = render(<Markdown text={'Here:\n\n```js\nconst a = 1'} streaming />)
    expect(container.querySelector('.code')).not.toBeNull()
    expect(container.querySelector('code')?.textContent).toContain('const a = 1')
    rerender(<Markdown text={'Here:\n\n```js\nconst a = 1\nconst b = 2\n```\n\nDone.'} />)
    expect(container.querySelector('code')?.textContent).toContain('const b = 2')
    expect(container.textContent).toContain('Done.')
  })

  it('unknown languages are shown plainly and safely (never as HTML)', () => {
    const { container } = render(<Markdown text={'```weirdlang\n<img src=x onerror=alert(1)>\n```'} />)
    expect(screen.getByTestId('code-lang').textContent).toBe('weirdlang')
    expect(container.querySelector('img')).toBeNull()
    expect(container.querySelector('code')?.textContent).toContain('<img src=x onerror=alert(1)>')
  })

  it('the copy button copies exactly the code and confirms', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true })
    Object.defineProperty(window, 'isSecureContext', { value: true, configurable: true })
    render(<Markdown text={'```bash\necho "hi"\nls -la\n```'} />)
    fireEvent.click(screen.getByTestId('copy-code'))
    await waitFor(() => expect(writeText).toHaveBeenCalledWith('echo "hi"\nls -la'))
    await waitFor(() => expect(screen.getByTestId('copy-code').textContent).toBe('Copied'))
  })

  it('never injects HTML or unsafe links', () => {
    const { container } = render(<Markdown text={'<script>alert(1)</script> <b>x</b>\n\n[bad](javascript:alert(1)) [ok](https://example.com) ![img](https://evil.test/p.png)'} />)
    expect(container.querySelector('script')).toBeNull()
    expect(container.querySelector('b')).toBeNull()
    expect(container.textContent).toContain('<script>')
    const anchors = Array.from(container.querySelectorAll('a'))
    expect(anchors.map((a) => a.getAttribute('href'))).toEqual(['https://example.com', 'https://evil.test/p.png'])
    expect(anchors.every((a) => a.getAttribute('rel') === 'noopener noreferrer' && a.getAttribute('target') === '_blank')).toBe(true)
    expect(container.querySelector('img')).toBeNull()
  })

  it('half-written markdown never throws while streaming', () => {
    for (const partial of ['**bol', '| a | b', '```', '- [', '> ', '1. ', '`', '[x](', '~~~']) {
      expect(() => render(<Markdown text={partial} streaming />)).not.toThrow()
      cleanup()
    }
  })
})

describe('util', () => {
  it('decodes entities', () => expect(decodeEntities('a &amp; b &lt;c&gt; &#65; &#x1F600; &nbsp;')).toBe('a & b <c> A 😀 \u00a0'))
  it('allows only http(s)/mailto links', () => {
    expect(safeHref('https://a.b')).toBe('https://a.b'); expect(safeHref('mailto:x@y.z')).toBe('mailto:x@y.z')
    for (const bad of ['javascript:alert(1)', 'data:text/html,x', '//evil', 'ftp://x', '']) expect(safeHref(bad)).toBeNull()
  })
  it('formats elapsed time and groups', () => {
    expect(formatElapsed(5)).toBe('5s'); expect(formatElapsed(125)).toBe('2m 05s')
    const now = Date.UTC(2026, 9, 7, 12) 
    expect(groupLabel(now / 1000 - 60, now)).toBe('Today'); expect(groupLabel(now / 1000 - 86400 * 1.5, now)).toBe('Yesterday')
    expect(groupLabel(now / 1000 - 86400 * 5, now)).toBe('Previous 7 days'); expect(groupLabel(undefined, now)).toBe('Older')
  })
})
