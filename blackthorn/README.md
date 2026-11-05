# Blackthorn

Blackthorn is the chat product served at <https://blackthorn.onrender.com>: a ChatGPT/Grok-style web UI, an agent that
chooses its own tools through the model's **native tool calling**, a hosted model (**NVIDIA Nemotron 3 Ultra** through
NVIDIA's OpenAI-compatible API), and a real machine to work on (**GitHub Codespaces**).

```
browser ──SSE──▶ blackthorn.app (Render) ──OpenAI-compatible stream──▶ integrate.api.nvidia.com ──▶ nvidia/nemotron-3-ultra-550b-a55b
                  │    ▲
                  │    └── tools: web_search (Tavily) · run_command · read/write/list files · fetch_url · computer_info · skills · lab
                  │
                  ├── Cloudflare D1: sessions, messages, memories, settings, Codespaces usage intervals
                  └── command channel (outbound long-poll) ──▶ agent inside the codespace ──▶ /workspace on its persistent disk
```

The app never dials into the codespace: the container's own agent opens the connection outwards
(`/api/codespaces/pull`), so there is no public port to expose and a restarted container simply reconnects.

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
| `blackthorn/llm.py`, `route.py` | streaming client; the route is NVIDIA's base URL + API key + model id (environment only) |
| `blackthorn/store.py`, `sql.py` | history on the real `sessions`/`messages` tables (D1 in prod, SQLite in tests) |
| `blackthorn/runs.py` | detached, resumable runs (refresh / reconnect never loses or duplicates events) |
| `blackthorn/codespaces.py` | GitHub Codespaces API (create/start/stop/find/duplicates) + the measured usage ledger |
| `blackthorn/codespace_channel.py` | the job channel: long-poll queue, presence, registration tokens, one re-queue on a restart |
| `blackthorn/codespace_computer.py` | the `/computer/*` surface the tools already speak, plus start-on-demand and stop-when-idle |
| `devcontainer/blackthorn_agent.py` (in the computer repo) | the in-codespace agent: authenticated exec/files/info over the outbound channel |
| `blackthorn/ui/` | **UI source** (React + TypeScript + Vite) — `blackthorn/static/` is its committed, hashed build |

## Deployment contract (Render)

Render builds `main` from GitHub automatically. Required environment variables (Render → Environment):
`CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_D1_DATABASE_ID`, `NVIDIA_API_KEY`,
`COMPUTER_DRIVE_GITHUB_PAT` (a Codespaces-capable fine-grained PAT for the computer repository), `HERMES_HOME`;
optional `TAVILY_API_KEYS`, `BLACKTHORN_MODEL`, `NVIDIA_BASE_URL`, `CODESPACE_REPO`, `CODESPACE_MACHINE`,
`CODESPACE_IDLE_STOP_SECONDS` (default 300), `CODESPACE_POWER_INTERVAL_SECONDS` (default 30),
`CODESPACES_MONTHLY_HOURS` (overrides the derived allowance), `CODESPACES_CORE_HOURS` (default 120).

* **Start command:** `HERMES_HOME=/opt/render/project/src/.hermes_home HERMES_DISABLE_LAZY_INSTALLS=1 python3 blackthorn_start.py`
* **Build command (recommended):**
  `export HERMES_HOME=/opt/render/project/src/.hermes_home && rm -rf tests website apps/desktop evals && python3 -m hermes_cli.main pm repair`
  It must **not** fetch anything from D1 or extract an archive over the checkout. (Before this repo was fixed, the build command
  downloaded a tarball from D1 and extracted it over the Git checkout, so the served UI/backend came from D1, not from Git.)

### Proving which version is live

`GET /api/version` returns the commit Render built (`RENDER_GIT_COMMIT`), the UI build hash, and an **integrity check**:
every shipped file is re-hashed and compared with `blackthorn/MANIFEST.json`. Any out-of-band modification of the deployed
tree appears as `integrity.drift`, is logged at startup, and is shown in the sidebar footer. The sidebar footer and the Codespaces
panel also show `v<version> · <commit>`.

## Public remote sample analysis (Proton WireGuard)

The agent has `malware_analyzer_submit` (one selected file in the computer's `malware_samples/` directory) and `malware_analyzer_report` (SHA-256 lookup). Submissions go to MalwareAnalyzer's public unauthenticated REST API, **not** to a lab VM. All submitted bytes and results may be public; only submit files approved for public release. A successful submit does not establish that the provider detonated the file; inspect the report's `dynamicDegraded` and behavior fields. Local `sample_triage` remains read-only static triage.

Configure `BLACKTHORN_MA_WG_CONFIG_B64` **only in Render's secret environment**, containing a base64-encoded Proton WireGuard profile (do not commit it or place it in D1). The userspace WireGuard client is a SHA-256-pinned `wireproxy` release, downloaded on first use to `/tmp` and bound to loopback. A dedicated HTTP CONNECT client uses `trust_env=False`, no redirects, an IP-literal VPN endpoint and DNS inside the tunnel. It compares direct and proxied public IP echoes before use and refuses uploads/lookup if unconfigured or the exit cannot be verified. There is **no direct connection fallback** to MalwareAnalyzer. The authenticated `GET /api/malware-analyzer/egress` checks the Render-side tunnel and provider without revealing either IP or sending a sample. `GET /api/malware-analyzer/report/{sha256}` returns only the public verdict and dynamic-analysis status, through that same verified exit. Provisioning or reconnecting can add seconds to the first remote tool call. Use a dedicated WireGuard profile for the Render process; starting the same profile on a second machine (including a local test client) can cause WireGuard endpoint roaming and interrupt the tunnel. A stalled proxy is restarted once and reverified before any remote call. A failed report GET can be retried once via the verified proxy; a public submission POST is never replayed automatically. The service provider, VPN operator, public sample content, and timing may still identify a submitter; this is not an untraceability guarantee. The browser and general-purpose tools are not forced through this VPN.

## Streaming protocol

`POST /api/chat/stream` → `text/event-stream`; each event is `id: <seq>` + `data: {json}` (heartbeat comments every 12 s):
`run.start`, `thinking` (state only — reasoning text never leaves the server), `text.delta`, `tool.start`, `tool.end`, `notice`,
`error`, `run.end`. A run is **detached** from the request: refresh/disconnect does not stop it; re-attach with
`GET /api/chat/runs/{id}/events?after=<last seq>`; `POST /api/chat/runs/{id}/cancel` stops it and closes the upstream request
so the model stops generating. The user message and an assistant placeholder are saved *before* the model is called and the
answer is saved progressively and finalised exactly once (`stop | length | cancelled | error | interrupted`).

## Model context budget

The agent plans against `BLACKTHORN_CONTEXT_TOKENS` (default 131 072, well inside Nemotron 3 Ultra's window). The
system prompt is small, tool schemas are counted separately, history is trimmed to fit and tool output is clipped;
older tool output is stubbed when the window fills. There is no engine to reconfigure — the number is a planning
budget, and the provider enforces its own hard limit.

## Codespaces: the agent's computer

* **Starts on demand.** Any computer tool call (`run_command`, `read_file`, `write_file`, …) calls `ensure_running`,
  which reuses the one codespace for the repository (never a second one) and waits for its agent to attach.
* **Stops when the task is finished.** `maybe_stop` refuses while a run is executing, while a tool call is in flight,
  or before `CODESPACE_IDLE_STOP_SECONDS` of real inactivity. Nothing is deleted: the container keeps its disk, so
  files, installed packages and task progress survive and a later task resumes them.
* **Reconnect, don't re-create.** Stopping is not deleting; `handover()` stops and starts the same codespace, and the
  agent re-registers on boot. A job that was in flight when the container died is re-queued once.
* **Honest usage.** `GET /api/codespaces/usage` reports the monthly allowance, hours measured from the intervals the
  app observed the machine running, the remainder and the percentage. GitHub's billing API needs a `Plan`-scoped
  token for personal accounts; without it the panel says "measured" instead of inventing a figure.

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
* No model credential is ever sent to the browser; `/api/health` reports only the provider, model id and whether a key is configured.
* The codespace agent authenticates with a session token it generates and keeps on the codespace disk. The app accepts
  that token only from a container that proves ownership with the `GITHUB_TOKEN` GitHub injects into it.

---

## Reliability, lifecycle and capability registry (2026-10)

| Module | Responsibility |
| --- | --- |
| `recovery.py` | Classifies every tool outcome as `ok` / `invalid` / `duplicate` / `infra` / `logic` and supplies bounded retry policies (exponential backoff + jitter + hard deadline). Infrastructure failures are the *environment's* fault, so they are retried and never counted as the model repeating itself. |
| `codespace_computer.py` | Power management for the codespace: start on demand, stop when idle, restart to resume, and a refusal to stop while work is in flight. |
| `computer_assets/skillctl.py` | One registry over Anthropic-Cybersecurity-Skills (811 loadable playbooks of 818 indexed), Claude-Red (68 offensive playbooks) and hackingtool-plugin (183 executable tools) — 879 skills / 183 tools in total, counted from the filesystem so nothing unloadable is advertised: discovery, loading, `preflight`, and install-once execution with real exit codes. |
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

### Live verification (real Codespaces, this deployment)

"Works in a unit test" is not evidence, so a live script exercises the real service:

| Script | What it proves |
| --- | --- |
| `blackthorn/scripts/live_codespaces.py` | Real GitHub Codespaces: start → run a command → write a file → stop → start again → read the same file back (the disk survived), plus the idle-refusal rule and a duplicate check (exactly one codespace for the repository). |
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
| `NVIDIA_API_KEY` | The model credential (NVIDIA API catalog) |
| `BLACKTHORN_MODEL` | Model id (default `nvidia/nemotron-3-ultra-550b-a55b`) |
| `COMPUTER_DRIVE_GITHUB_PAT` | Codespaces-capable PAT: creates/starts/stops the computer and reads its repository |
| `CODESPACE_REPO` / `CODESPACE_MACHINE` | Repository and machine type for the computer (default `jadeniran21/Computer-drive`, `basicLinux32gb`) |
| `CODESPACE_IDLE_STOP_SECONDS` / `CODESPACE_POWER_INTERVAL_SECONDS` | Idle window before an automatic stop (300) and how often the power loop checks (30) |
| `PANDASTACK_LAB_API_KEY` | Only for the separate malware lab; the computer no longer uses PandaStack |
| `BLACKTHORN_MAX_STRIKES` / `BLACKTHORN_MAX_RECOVERY_ROUNDS` / `BLACKTHORN_TOOL_ATTEMPTS` | Execution-continuity bounds |
