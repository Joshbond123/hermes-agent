#!/usr/bin/env bash
# Blackthorn computer bootstrap — idempotent, safe to re-run after every sandbox replacement.
#
# Restores everything the agent needs on a fresh PandaStack microVM (or any Debian/Ubuntu host):
#   * the workspace directory layout
#   * the three cybersecurity repositories + the unified capability registry
#   * Camoufox and its browser build
#   * the base CLI security utilities
#
# The workspace itself (/workspace) is restored from the private GitHub Computer-drive by the
# runtime before this script runs; nothing here invents task state.
#
# Usage: bash bootstrap.sh [--fast] [--no-browser] [--no-tools]
set -euo pipefail

WS="${BLACKTHORN_WS:-/workspace/blackthorn_workspace}"
CATALOG="$WS/security_catalog"
ASSETS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAST=0; BROWSER=1; TOOLS=1
for arg in "$@"; do
  case "$arg" in
    --fast) FAST=1 ;;
    --no-browser) BROWSER=0 ;;
    --no-tools) TOOLS=0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

log() { printf '[bootstrap %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

mkdir -p "$WS"/{tools,downloads,browser_screenshots,malware_samples,security_catalog}

# ---------------------------------------------------------------- base utilities
if [ "$TOOLS" = "1" ] && [ "$FAST" = "0" ]; then
  if ! command -v nmap >/dev/null 2>&1; then
    log "installing base security utilities"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq || true
    apt-get install -y -qq --no-install-recommends \
      git unzip jq python3-venv python3-pip build-essential pkg-config \
      nmap dnsutils whois netcat-openbsd socat tcpdump binwalk foremost steghide \
      exiftool file binutils yara sqlmap nikto gobuster dirb sslscan smbclient \
      strace ltrace gdb radare2 || log "some packages were unavailable; continuing"
  fi
fi

# ---------------------------------------------------------------- registries
cp -f "$ASSETS/skillctl.py" "$WS/tools/skillctl.py"
export BLACKTHORN_SKILL_ROOT="$CATALOG"
log "syncing the three cybersecurity repositories"
python3 "$WS/tools/skillctl.py" ensure    >/dev/null || log "source sync reported a problem"
log "building the unified capability registry"
python3 "$WS/tools/skillctl.py" registry | tail -1

# ---------------------------------------------------------------- browser
if [ "$BROWSER" = "1" ] && [ "$FAST" = "0" ]; then
  VENV="$WS/tools/venv"
  if [ ! -x "$VENV/bin/python" ] || ! "$VENV/bin/python" -c 'import camoufox' 2>/dev/null; then
    log "installing Camoufox"
    python3 -m venv "$VENV" || true
    "$VENV/bin/pip" install -q --upgrade pip
    "$VENV/bin/pip" install -q 'camoufox[geoip]' playwright httpx
    "$VENV/bin/python" -m camoufox fetch || log "browser download failed; browse will retry at runtime"
  else
    log "Camoufox already present"
  fi
  cp -f "$ASSETS/browse.py" "$WS/tools/browse.py"
fi

log "bootstrap complete: $(python3 "$WS/tools/skillctl.py" preflight | tail -1 | cut -c1-220)"
