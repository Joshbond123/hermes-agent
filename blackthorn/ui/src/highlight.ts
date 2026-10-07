/** Syntax highlighting, bundled locally (no CDN): highlight.js core + the languages people actually paste. */
import hljs from 'highlight.js/lib/core'
import bash from 'highlight.js/lib/languages/bash'
import c from 'highlight.js/lib/languages/c'
import cpp from 'highlight.js/lib/languages/cpp'
import css from 'highlight.js/lib/languages/css'
import diff from 'highlight.js/lib/languages/diff'
import dockerfile from 'highlight.js/lib/languages/dockerfile'
import go from 'highlight.js/lib/languages/go'
import ini from 'highlight.js/lib/languages/ini'
import java from 'highlight.js/lib/languages/java'
import javascript from 'highlight.js/lib/languages/javascript'
import json from 'highlight.js/lib/languages/json'
import markdown from 'highlight.js/lib/languages/markdown'
import python from 'highlight.js/lib/languages/python'
import rust from 'highlight.js/lib/languages/rust'
import sql from 'highlight.js/lib/languages/sql'
import typescript from 'highlight.js/lib/languages/typescript'
import xml from 'highlight.js/lib/languages/xml'
import yaml from 'highlight.js/lib/languages/yaml'

const REGISTERED: Record<string, Parameters<typeof hljs.registerLanguage>[1]> = {
  bash, c, cpp, css, diff, dockerfile, go, ini, java, javascript, json, markdown, python, rust, sql, typescript, xml, yaml,
}
for (const [name, def] of Object.entries(REGISTERED)) hljs.registerLanguage(name, def)

const ALIASES: Record<string, string> = {
  js: 'javascript', jsx: 'javascript', mjs: 'javascript', node: 'javascript', ts: 'typescript', tsx: 'typescript', py: 'python', python3: 'python',
  sh: 'bash', shell: 'bash', zsh: 'bash', console: 'bash', terminal: 'bash', html: 'xml', svg: 'xml', yml: 'yaml', md: 'markdown',
  'c++': 'cpp', h: 'c', hpp: 'cpp', rs: 'rust', golang: 'go', toml: 'ini', jsonc: 'json', patch: 'diff', docker: 'dockerfile', postgres: 'sql', mysql: 'sql',
}

export function normalizeLang(lang: string | undefined | null): string {
  const raw = (lang ?? '').trim().toLowerCase().split(/\s+/)[0] ?? ''
  return ALIASES[raw] ?? raw
}

export function languageLabel(lang: string | undefined | null): string {
  const raw = (lang ?? '').trim().split(/\s+/)[0]
  return raw ? raw.toLowerCase() : 'text'
}

/** Highlighted HTML (already escaped by highlight.js), or null when the language is unknown / the block is huge. */
export function highlightCode(code: string, lang: string | undefined | null): string | null {
  const name = normalizeLang(lang)
  if (!name || !hljs.getLanguage(name) || code.length > 40_000) return null
  try {
    return hljs.highlight(code, { language: name, ignoreIllegals: true }).value
  } catch {
    return null
  }
}

export const supportedLanguages = () => hljs.listLanguages()
