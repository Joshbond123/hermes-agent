#!/usr/bin/env bash
# Render *build command* for Blackthorn:   bash scripts/render_build.sh
#
# GitHub is the single source of truth. This script builds the web UI from web/ and prepares the
# Python runtime from the checkout — it never downloads code from anywhere else.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
export HERMES_HOME="${HERMES_HOME:-$ROOT/.hermes_home}"
COMMIT="${RENDER_GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}"
BRANCH="${RENDER_GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)}"
say() { printf '\n==> %s\n' "$*"; }

say "[1/5] trimming files the product does not use (same set the previous build removed)"
rm -rf plugins/memory/byterover plugins/memory/holographic plugins/memory/mem0 \
       plugins/memory/openviking plugins/memory/retaindb website evals apps/desktop 2>/dev/null || true

say "[2/5] Node.js (needed to build the UI)"
node_ok() { command -v node >/dev/null 2>&1 && node -e 'process.exit(Number(process.versions.node.split(".")[0]) >= 22 ? 0 : 1)'; }
if ! node_ok; then
  WANT="$(tr -d 'v \n' < .nvmrc)"          # e.g. "26"
  TOOLS="$ROOT/.render-tools"; mkdir -p "$TOOLS"
  VER="$(curl -fsSL https://nodejs.org/dist/index.json | python3 -c "import sys,json; w=sys.argv[1]; print(next(x['version'] for x in json.load(sys.stdin) if x['version'].lstrip('v').split('.')[0]==w))" "$WANT")"
  echo "installing Node $VER (.nvmrc says $WANT)"
  curl -fsSL "https://nodejs.org/dist/$VER/node-$VER-linux-x64.tar.gz" | tar -xz -C "$TOOLS"
  export PATH="$TOOLS/node-$VER-linux-x64/bin:$PATH"
fi
echo "node $(node --version), npm $(npm --version)"

say "[3/5] building the web UI from web/ (typecheck + Vite)"
npm ci --workspace web --include-workspace-root=false --no-audit --no-fund --ignore-scripts
node scripts/build/web.mjs

DIST="hermes_cli/web_dist"
test -s "$DIST/index.html" || { echo "FATAL: the UI build produced no index.html"; exit 1; }
# Guard against ever shipping a UI that lacks the Blackthorn chat (the failure mode of the old pipeline).
if ! grep -rlq "bt-chat-root" "$DIST/assets"; then
  echo "FATAL: the built UI does not contain the Blackthorn chat (marker bt-chat-root missing)."; exit 1
fi
SRC_HASH="$(find web/src -type f | sort | xargs sha256sum | sha256sum | cut -c1-16)"
printf '{"commit":"%s","branch":"%s","built_at":"%s","web_source_hash":"%s"}\n' \
  "$COMMIT" "$BRANCH" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SRC_HASH" > "$DIST/build-info.json"
echo "UI build stamp: $(cat "$DIST/build-info.json")"

say "[4/5] Python runtime (isolated Hermes environment)"
if [ "${BT_SKIP_PM_REPAIR:-0}" != "1" ]; then
  python3 -m hermes_cli.main pm repair
else
  echo "(skipped: BT_SKIP_PM_REPAIR=1)"
fi

say "[5/5] done — commit $COMMIT"
