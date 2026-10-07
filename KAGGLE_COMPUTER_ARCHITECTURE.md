# Blackthorn Agent Workspace Architecture

## Rule
The agent's tools (terminal, files, web page fetch, packages, builds) execute **only** on Kaggle Computer.
Render is the control/API plane only. (`web_search` calls Tavily from Render; memory lives in Cloudflare D1.)

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
UI → `blackthorn.agent` loop (Render) → model chooses tools via native `tool_calls` → HTTP to the Kaggle Computer API →
real results go back as `role: tool` messages → text and tool events stream to the UI over SSE.
See `blackthorn/README.md`.
