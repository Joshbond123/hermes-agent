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

---

## Reliability, lifecycle and capability registry (2026-10)

| Module | Responsibility |
| --- | --- |
| `recovery.py` | Classifies every tool outcome as `ok` / `invalid` / `duplicate` / `infra` / `logic` and supplies bounded retry policies (exponential backoff + jitter + hard deadline). Infrastructure failures are the *environment's* fault, so they are retried and never counted as the model repeating itself. |
| `lifecycle.py` | Sandbox-lifecycle manager. Five minutes before the provider TTL it publishes a durable checkpoint, creates an overlapping replacement sandbox, restores and verifies the workspace, promotes the replacement, then deletes the old VM. Failures roll back to the still-live sandbox and are retried with backoff. |
| `panda_keys.py` | PandaStack key pool. Verifies each key against `/v1/me` and `/v1/usage`, projects exhaustion from the provider's own month-to-date rollup + configured budget, and rotates ~5 minutes before the projected limit (or immediately on 401/402/403/429). Only 12-character fingerprints are ever logged or stored. |
| `computer_assets/skillctl.py` | One registry over Anthropic-Cybersecurity-Skills (818 playbooks), Claude-Red (79 offensive playbooks) and hackingtool-plugin (183 executable tools): discovery, loading, `preflight`, and install-once execution with real exit codes. |
| `computer_assets/browse.py` | Camoufox controller: open / click / type / scroll / screenshot / accessibility / tabs / download / close, with crash-relaunch, timeout retry, session persistence and bot-challenge *detection* (never a false claim of solving). |
| `computer_assets/bootstrap.sh` | Idempotent bootstrap for a replacement sandbox: workspace layout, all three repositories, the unified registry, Camoufox and the base utilities. |

### The two production failures these fix

1. `Provider VM expired while idle…` — the sandbox is recreated by `PandaControl.ensure_for_task`,
   the workspace is restored from the private GitHub drive, and the lifecycle manager now does this
   *proactively* before the TTL expires instead of only after a failure.
2. `Stopped using tools (repeated identical tool calls)…` — a provider failure no longer consumes the
   duplicate-call budget and no longer produces a "strike". Only a repeated call that genuinely
   *completed* against an unchanged workspace can strike, and even then the model is first told to
   change strategy. `BLACKTHORN_MAX_RECOVERY_ROUNDS` bounds the automatic recovery rounds.

### Live verification (PandaStack, this deployment)

Two scripts exist because "it works in a unit test" is not evidence:

| Script | What it proves |
| --- | --- |
| `blackthorn/scripts/live_acceptance.py` | Real provider: creates the managed sandbox, runs python/git on it, publishes a workspace checkpoint to the private GitHub drive, performs a full handover (replacement created → workspace restored → probe → promoted → old VM deleted), proves a file written *before* the handover is readable *after*, clones all three repositories in the guest, builds the 897-skill/183-tool registry, and releases every sandbox it created. |
| `blackthorn/scripts/live_browser_test.py` | Real Camoufox inside a real microVM: install (disk-backed scratch), browser fetch, then open / click / type+submit / scroll / screenshot / accessibility / tabs / download / challenge detection / session persistence / close, plus "no browser processes left behind". |
| `blackthorn/scripts/smoke_catalog.py` | The cybersecurity tools through the **real ToolRegistry** against a real catalogue: `cyber_preflight`, `skill_search`, `tool_search`, `skill_load`, `cyber_run` (including honest refusal of an entry with no command and rejection of a malformed id), and that no credential travels to the computer. |

Operational notes learned from the live run:

* The guest's `/tmp` is a ~1 GB tmpfs. The Camoufox bundle is ~1.5 GB, so `browse.py` moves
  `TMPDIR`/`XDG_CACHE_HOME` to disk (`/workspace/.cache` by default, `BLACKTHORN_BROWSER_CACHE`/`_TMP`
  to override) before fetching. Fetching into `/tmp` fails with `No space left on device`.
* Firefox needs its shared libraries; a bare microVM ships none of them and the browser dies with an
  opaque `XPCOMGlueLoad error`. `browse.py` checks the binary with `ldd` and installs the package set
  in `LINUX_LIBS` before launching, and raises a named error if a library still cannot be resolved.
* Rebuildable caches (`.cache`, `__pycache__`, `.npm`, `.tmp`) are excluded from the durable
  workspace checkpoint, and an archive over `BLACKTHORN_WORKSPACE_MAX_BYTES` is refused instead of
  pushed to GitHub.
* `export VAR=value command` in a guest shell exports a variable named `command` and runs nothing —
  environment assignments must prefix the command. This silently produced empty tool output until the
  harness was fixed.

### New API

* `GET /api/computer/session` — provider-reported sandbox timing (`seconds_left`, `handover_due`),
  per-key quota projection, and the last lifecycle events. No secrets.

### Environment

| Variable | Purpose |
| --- | --- |
| `PANDASTACK_API_KEY` (+ `_2`, `_3`, … or comma-separated `PANDASTACK_API_KEYS`) | Authorized keys available for rotation |
| `PANDASTACK_MONTHLY_BUDGET_USD` | Optional cap used only when the provider reports none (`monthly_budget_usd: 0`) |
| `PANDASTACK_ROTATE_BEFORE_SECONDS` | Rotation lead time (default 300) |
| `PANDASTACK_HANDOVER_WINDOW_SECONDS` | Handover lead time (default 300) |
| `PANDASTACK_LIFECYCLE_INTERVAL_SECONDS` | Lifecycle tick (default 60) |
| `BLACKTHORN_MAX_STRIKES` / `BLACKTHORN_MAX_RECOVERY_ROUNDS` / `BLACKTHORN_TOOL_ATTEMPTS` | Execution-continuity bounds |
