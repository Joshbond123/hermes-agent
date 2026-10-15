#!/usr/bin/env bash
# Decrypt the Com11 payload (payload.enc) into a directory. Used by every workflow;
# the passphrase comes from the COM11_REPO_KEY Actions secret and is never written here.
set -euo pipefail
: "${COM11_REPO_KEY:?COM11_REPO_KEY must be provided as an environment variable}"
OUT="${1:-payload}"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
TMP="$(mktemp /tmp/com11-payload.XXXXXX.tar)"
trap 'rm -f "$TMP"' EXIT
gpg --batch --yes --quiet --pinentry-mode loopback --passphrase "$COM11_REPO_KEY" \
    -d "$REPO_DIR/payload.enc" > "$TMP"
mkdir -p "$OUT"
tar -xf "$TMP" -C "$OUT"
echo "payload decrypted -> $OUT/ ($(find "$OUT" -type f | wc -l) files)"
