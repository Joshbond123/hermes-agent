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

## Public remote sample analysis (Proton WireGuard)

The agent has `malware_analyzer_submit` (one selected file in the computer's `malware_samples/` directory) and `malware_analyzer_report` (SHA-256 lookup). Submissions go to MalwareAnalyzer's public unauthenticated REST API, **not** to an on-Kaggle malware VM. All submitted bytes and results may be public; only submit files approved for public release. A successful submit does not establish that the provider detonated the file; inspect the report's `dynamicDegraded` and behavior fields. Local `sample_triage` remains read-only static triage.

Configure `BLACKTHORN_MA_WG_CONFIG_B64` **only in Render's secret environment**, containing a base64-encoded Proton WireGuard profile (do not commit it or place it in D1). The userspace WireGuard client is a SHA-256-pinned `wireproxy` release, downloaded on first use to `/tmp` and bound to loopback. A dedicated HTTP CONNECT client uses `trust_env=False`, no redirects, an IP-literal VPN endpoint and DNS inside the tunnel. It compares direct and proxied public IP echoes before use and refuses uploads/lookup if unconfigured or the exit cannot be verified. There is **no direct connection fallback** to MalwareAnalyzer. The authenticated `GET /api/malware-analyzer/egress` checks the Render-side tunnel and provider without revealing either IP or sending a sample. `GET /api/malware-analyzer/report/{sha256}` returns only the public verdict and dynamic-analysis status, through that same verified exit. Provisioning or reconnecting can add seconds to the first remote tool call. Use a dedicated WireGuard profile for the Render process; starting the same profile on a second machine (including a local test client) can cause WireGuard endpoint roaming and interrupt the tunnel. A stalled proxy is restarted once and reverified before any remote call. A failed report GET can be retried once via the verified proxy; a public submission POST is never replayed automatically. The service provider, VPN operator, public sample content, and timing may still identify a submitter; this is not an untraceability guarantee. The browser and general-purpose tools are not forced through this VPN.

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
* The app requires an owner password on production D1. Configure `BLACKTHORN_AUTH_PASSWORD_HASH` (PBKDF2 format
  from `blackthorn.auth.hash_password`) and a random `BLACKTHORN_AUTH_SIGNING_KEY` (32+ characters) as Render
  environment variables **before** deploying this version; without them, the API fails closed with 503.
  The login sets an HttpOnly, Secure, SameSite=Strict cookie and refreshes it for active users. Settings can
  change the password; that invalidates older sessions across workers. Keep the site behind Cloudflare Access
  as additional protection. Neither the password nor the signing key belongs in Git or D1 in plaintext.
* The public GPU status never contains the tunnel URL, gateway key or account identifiers.
* The gateway key is generated per Kaggle boot (`bt-…`) and published to D1 by the notebook. A previous constant key remains in
  old Git history: treat it as compromised (it stops working at the next GPU boot).
