# Blackthorn **Com11** — a Kali Linux "computer" that lives on GitHub Actions.

Com11 replaces the Kaggle computer in the Blackthorn fleet. It boots inside a
privileged Kali container on a GitHub-hosted runner, comes up **fail-closed**
(anonymity first, readiness only after verification), exposes the same
`/computer/*` API through the existing Blackthorn relay, and persists its
workspace across workflow restarts.

## Public repository contents

Everything sensitive in this repository is **encrypted at rest**:

| file | state | why |
|---|---|---|
| `.github/workflows/*.yml` | plaintext | GitHub executes workflow YAML from the repo — it cannot be encrypted |
| `bootstrap.sh` | plaintext | tiny decryptor; contains no secrets |
| `README.md` | plaintext | this file |
| `payload.enc` | **GPG AES256 ciphertext** | every runner script, VPN material, docs and configs |
| `workspace` branch | **GPG AES256 ciphertext** | the agent workspace archive (`workspace.tar.gpg`) |

The passphrase is the `COM11_REPO_KEY` **Actions secret** (never committed here,
never in an artifact, never in logs). Rotate it by re-encrypting `payload.enc`
and having the next boot decrypt with the new secret.

## Lifecycle

* **start** — the GPU controller (Render) dispatches `com11-computer.yml`.
* **boot** — decrypt payload → restore workspace → start container →
  Tor first, then Proton VPN (OpenVPN over Tor; WireGuard fallback) →
  kill switch + leak verification → gateway + relay → `.ready` only if the
  verification battery passes.
* **run** — the fleet probes `GET /computer/activity` through the relay;
  agent commands stream back with heartbeats; the workspace is committed to the
  `workspace` branch every 60 s (encrypted) and on every `/computer/checkpoint`.
* **stop** — Render calls checkpoint then cancels the run; at ~5 h the run
  handoff-dispatches a successor (GitHub's hosted job limit is 6 h).

## Building images

`build-image.yml` produces `ghcr.io/<owner>/<repo>/com11-kali:base|tools`.
Without an image the boot builds an ephemeral Kali image on the runner.

## Security notes

* no PATs/keys/passwords in this repository, its history, logs or artifacts —
  only Actions secrets; the gateway process runs under `env -i` with an
  explicit whitelist (verified by `evidence/verify_report.json → env_leaks`).
* encryption key never stored next to the ciphertext it decrypts.
