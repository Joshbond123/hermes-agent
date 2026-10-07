import { createHash } from 'node:crypto'
import { readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs'
import { join, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import react from '@vitejs/plugin-react'
import { defineConfig, type Plugin } from 'vite'

const here = fileURLToPath(new URL('.', import.meta.url))
const outDir = resolve(here, '../static')

/** Hash of everything the build depends on. A Python test recomputes it, so source and committed build cannot drift apart. */
function sourceHash(): string {
  const entries: string[] = []
  const add = (full: string) => entries.push(`${relative(here, full).split('\\').join('/')}\n${createHash('sha256').update(readFileSync(full)).digest('hex')}\n`)
  const walk = (dir: string) => {
    for (const entry of readdirSync(dir).sort()) {
      const full = join(dir, entry)
      if (statSync(full).isDirectory()) walk(full)
      else add(full)
    }
  }
  walk(join(here, 'src'))
  for (const f of ['index.html', 'package.json', 'package-lock.json', 'vite.config.ts', 'tsconfig.json']) add(join(here, f))
  return createHash('sha256').update(entries.sort().join('')).digest('hex')
}

/** After the build: record what was shipped (name → short sha256) so /api/version can prove which UI is live. */
function buildInfo(): Plugin {
  return {
    name: 'blackthorn-build-info',
    apply: 'build',
    closeBundle() {
      const files: Record<string, string> = {}
      const walk = (dir: string) => {
        for (const entry of readdirSync(dir)) {
          const full = join(dir, entry)
          if (statSync(full).isDirectory()) walk(full)
          else if (entry !== 'build.json') {
            files[relative(outDir, full)] = createHash('sha256').update(readFileSync(full)).digest('hex').slice(0, 12)
          }
        }
      }
      walk(outDir)
      const hash = createHash('sha256').update(JSON.stringify(files)).digest('hex').slice(0, 12)
      writeFileSync(join(outDir, 'build.json'), JSON.stringify({ hash, source_hash: sourceHash(), files }, null, 1) + '\n')
    },
  }
}

export default defineConfig({
  plugins: [react(), buildInfo()],
  base: '/',
  build: { outDir, emptyOutDir: true, assetsDir: 'assets', sourcemap: false, target: 'es2022', chunkSizeWarningLimit: 900 },
  server: { port: 5173, proxy: { '/api': 'http://127.0.0.1:8765' } },
  test: { environment: 'jsdom', globals: true, include: ['src/**/*.test.{ts,tsx}'] },
})
