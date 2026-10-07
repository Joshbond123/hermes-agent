# Blackthorn Agent Workspace Architecture

## Rule
Hermes Agent tools (terminal, files, browser, packages, builds) execute **only** on Kaggle Computer.
Render is the control/API plane only.

## Paths
- Workspace: `/kaggle/working/blackthorn_workspace`
- Computer API (on same Cloudflare tunnel as GPU):
  - GET  `/computer/info`
  - POST `/computer/exec`       {command, timeout_seconds, cwd}
  - POST `/computer/list_files` {path}
  - POST `/computer/read_file`  {path}
  - POST `/computer/write_file` {path, content}
  - POST `/computer/fetch_url`  {url}

## GPU
Dual T4 inference via `/v1/chat/completions` on the same host/tunnel.

## Kali Linux
Kaggle notebooks do **not** support nested Docker/VMs reliably (no privileged containers).
Closest supported lab:
- Isolated directory `/kaggle/working/blackthorn_workspace/security_lab`
- User-space tools via `pip` / downloadable static binaries
- Never target real external systems

## Flow
UI → Hermes agent loop (Render) → tool calls → HTTP to Kaggle Computer API → real results → stream to UI
