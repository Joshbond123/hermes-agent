# Blackthorn

Blackthorn is the chat product served at <https://blackthorn.onrender.com>: a ChatGPT/Grok-style web UI, an agent that
chooses its own tools through the model's **native tool calling**, and a Kaggle GPU (Qwen3.8-27B, 2×T4) that runs the
model and the "remote computer" the tools act on.

```
browser ──SSE──▶ blackthorn.app (Render) ──OpenAI-compatible stream──▶ Kaggle gateway (Cloudflare tunnel) ──▶ Ollama/llama.cpp
                  │    ▲                                                   └─ /computer/{exec,read_file,write_file,list_files,fetch_url,info}
                  │    └── tools: web_search (Tavily) · run_command · read/write/list files · fetch_url · computer_info · remember
                  └── Cloudflare D1: sessions, messages (history), memories, GPU row, system prompt
```

## What runs in production (and only this)

`python3 blackthorn_start.py` → activates the sealed dependency environment built by `hermes_cli.main pm repair` →
`uvicorn blackthorn.app:create_app`. Cold start is a few seconds. **Git is the only source of the running code**: there
is no overlay, no download-and-extract step, and no credential in the repository.

| Path | Role |
| --- | --- |
| `blackthorn_start.py` | Render entrypoint (activates the environment, starts the app) |
| `blackthorn/app.py`, `api.py` | FastAPI app, routes, static serving (only content-hashed assets are `immutable`), CSP |
| `blackthorn/agent.py` | the agent loop (native tool calls, bounded, cancellable, never stores reasoning) |
| `blackthorn/tools/` | tool registry: JSON-schema validation, timeouts, output budgets, redaction, permissions |
| `blackthorn/llm.py`, `route.py` | streaming client for the Kaggle gateway; route (tunnel + key) read from D1 |
| `blackthorn/store.py`, `sql.py` | history on the real `sessions`/`messages` tables (D1 in prod, SQLite in tests) |
| `blackthorn/runs.py` | detached, resumable runs (refresh / reconnect never loses or duplicates events) |
| `blackthorn/gpu.py`, `kaggle_bundle.py` | GPU facade (honest status, no secrets) · notebook generated from source at push time |
| `cloudflare_d1_client.py` | GPU controller (Kaggle API, tunnel health, watchdog, quota) — credentials from env only |
| `kaggle_cyber_ornith/cyber_ornith_server.py` | the code that runs on Kaggle (gateway + model engine + `/computer/*`) |
| `blackthorn/ui/` | **UI source** (React + TypeScript + Vite) — `blackthorn/static/` is its committed, hashed build |

## Deployment contract (Render)

Render builds `main` from GitHub automatically. Required environment variables (Render → Environment):
`CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_D1_DATABASE_ID`, `KAGGLE_USERNAME`, `KAGGLE_API_TOKEN`,
`HERMES_HOME`; optional `TAVILY_API_KEYS` (comma separated; otherwise read from D1 `state_meta.tavily_api_keys`).

* **Start command:** `HERMES_HOME=/opt/render/project/src/.hermes_home HERMES_DISABLE_LAZY_INSTALLS=1 python3 blackthorn_start.py`
* **Build command (recommended):**
  `export HERMES_HOME=/opt/render/project/src/.hermes_home && rm -rf tests website apps/desktop evals && python3 -m hermes_cli.main pm repair`
  It must **not** fetch anything from D1 or extract an archive over the checkout. (Before this repo was fixed, the build command
  downloaded a tarball from D1 and extracted it over the Git checkout, so the served UI/backend came from D1, not from Git.)

### Proving which version is live

`GET /api/version` returns the commit Render built (`RENDER_GIT_COMMIT`), the UI build hash, and an **integrity check**:
every shipped file is re-hashed and compared with `blackthorn/MANIFEST.json`. Any out-of-band modification of the deployed
tree appears as `integrity.drift`, is logged at startup, and is shown in the sidebar footer. The sidebar footer and the GPU
panel also show `v<version> · <commit>`.

## Streaming protocol

`POST /api/chat/stream` → `text/event-stream`; each event is `id: <seq>` + `data: {json}` (heartbeat comments every 12 s):
`run.start`, `thinking` (state only — reasoning text never leaves the server), `text.delta`, `tool.start`, `tool.end`, `notice`,
`error`, `run.end`. A run is **detached** from the request: refresh/disconnect does not stop it; re-attach with
`GET /api/chat/runs/{id}/events?after=<last seq>`; `POST /api/chat/runs/{id}/cancel` stops it and closes the upstream request
so the model stops generating. The user message and an assistant placeholder are saved *before* the model is called and the
answer is saved progressively and finalised exactly once (`stop | length | cancelled | error | interrupted`).

## Model context budget

The model server runs with a **4096-token** window. The system prompt is ~150 tokens, tool schemas ~800, history is trimmed to fit,
tool output is clipped (~3 200 chars) and older tool output is stubbed when the window fills. Raise it by changing the Kaggle
engine's context size (`QWEN38_NUM_CTX`) *and* `BLACKTHORN_CONTEXT_TOKENS` on Render.

## Develop & test

```bash
# backend + browser tests (use the same sealed environment as production: HERMES_HOME=... python -m hermes_cli.main pm repair)
python -m pytest -c blackthorn/tests/pytest.ini blackthorn/tests        # needs: pytest pytest-asyncio playwright (+ chromium)
# front end
cd blackthorn/ui && npm ci && npm run typecheck && npx vitest run && npm run build
python -m blackthorn.version --write                                    # refresh blackthorn/MANIFEST.json, then commit
```
`test_integrity.py` fails when: a credential-shaped string is committed, the manifest does not match the tree, the entrypoint
grows a download/extract step, the UI build is not content-hashed or loads anything from a CDN, or `blackthorn/ui` changed
without rebuilding `blackthorn/static`.

## Security notes
* The app has **no login** (as before). Anyone who can reach the URL can chat, run the agent's tools on the Kaggle computer and
  turn the GPU on/off. Put it behind Cloudflare Access / a Render private service or add an access gate before sharing the URL.
* The public GPU status never contains the tunnel URL, gateway key or account identifiers.
* The gateway key is generated per Kaggle boot (`bt-…`) and published to D1 by the notebook. A previous constant key remains in
  old Git history: treat it as compromised (it stops working at the next GPU boot).
