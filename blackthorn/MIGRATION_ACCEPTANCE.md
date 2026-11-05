# Migration acceptance — NVIDIA Nemotron 3 Ultra + GitHub Codespaces

Date: 2026-11-05 · app version 5.5.5 · commits `66730a4ab1` … `c71cb8f2f2` (branch `main`)
Live: https://blackthorn.onrender.com

Everything below was observed on the live deployment, not inferred from code. Commands, response
bodies and identifiers are quoted so each claim can be re-checked.

## 1. Kaggle GPU infrastructure — removed

| What was removed | Evidence |
| --- | --- |
| `cloudflare_d1_client.py`, `gpu.py`, `fleet.py`, `relay.py`, `kaggle_bundle.py`, `kaggle_cyber_ornith/`, `kaggle_cluster/`, `lifecycle.py`, `workspace_mirror.py`, the PandaStack adapter (`panda_computer/panda_control/panda_keys`) and their tests | deleted in `66730a4ab1` |
| D1 control-plane tables | `DROP TABLE kaggle_gpu_state` (69 rows), `DROP TABLE gateway_hygiene_state` — both executed; a listing of `sqlite_master` no longer contains them |
| Render environment | `KAGGLE_API_TOKEN`, `KAGGLE_USERNAME`, `KAGGLE_FLEET_MODEL_PRIMARY/_BACKUP`, `QWEN38_*`, `PANDASTACK_*`, `BLACKTHORN_RELAY_URL`, `BLACKTHORN_TUNNEL_HOSTNAME`, `OPENAI_API_KEY/_BASE_URL`, `INROK_API_KEY` deleted; 25 variables remain |
| Routes | `/api/kaggle-gpu/status` → 404, `/api/fleet` → 404, `/api/relay/poll` → 404 (authenticated) |
| Header GPU controller | `ui/src/gpu.ts`, `GpuPanel.tsx` deleted; rebuilt bundle `index-D8FJeySr.js` / `index-CEbKNgHR.css` ships the Codespaces controller |
| Dashboard auth allowlist | `/api/kaggle-gpu/*` entries removed from `hermes_cli/dashboard_auth/public_paths.py` |

A repository-wide search for `kaggle` now matches only third-party training documentation under
`optional-skills/mlops/` (unsloth/lightning course text, where "Kaggle" is a hosting platform, not
our infrastructure). Local llama.cpp GPU support in `hermes_cli/local_runtime/` is untouched: it is
the desktop CLI's own model runtime, unrelated to the removed cloud GPUs.

## 2. Nemotron 3 Ultra — connected and driving the agent

`/api/version` reports the deployed commit; the live run below reports `model:
nvidia/nemotron-3-ultra-550b-a55b` on NVIDIA's OpenAI-compatible endpoint
(`NVIDIA_BASE_URL=https://integrate.api.nvidia.com`, `NVIDIA_API_KEY` in the Render environment,
`BLACKTHORN_MODEL=nvidia/nemotron-3-ultra-550b-a55b`).

Live task (11:38 UTC, `live_task_proof2.json`), 4 tool calls, finish `stop`:

> open `/workspace/handoff.txt` created by an earlier task, report it, append `date -u` + cwd, read back

Answer returned:
```
blackthorn codespace ok 6.8.0-1064-azure
Thu Nov  5 11:38:08 UTC 2026 /workspaces/blackthorn
```
The kernel string is the codespace's own, and the first line was written by a **previous run** — so
the agent reasoned, planned, called tools and the disk survived a stop/start cycle.

A rejected provider key now fails fast and honestly (`model_auth`, retryable=False) instead of
re-reading a control-plane row that no longer exists.

## 3. Superpowers skills — searched, loaded, used

Catalog source `obra/superpowers` is part of the index (ids `superpowers:<name>`, category
`process`), cloned on first use inside the computer, and the skill-first rule is in every system
prompt.

Live verification inside the codespace (`skillctl.py search --q 'systematic debugging'`):
`{"ok": true, "matches": 12, "counts": {"skills": 912, "tools": 183, "by_source": {"anthropic":
818, "claude-red": 79, "hackingtool": 0, "superpowers": 15}}}`, top hit
`superpowers:systematic-debugging`.

Live task (11:56 UTC, `live_task_proof4.json`), 19 s, `skill_search` → `skill_load` → `run_command`:

> find the process skill covering systematic debugging, load it, quote its first three non-empty lines

Answer: `# Systematic Debugging / ## Overview` (the skill's own text) plus command output `45`.

This test also found and fixed a real defect: the tools write helpers to the logical root
`/workspace` while the container keeps its workspace under `/workspaces`, so `skill_search` failed
in-container for a whole run. The container now guarantees `/workspace` (start script on every
start, plus the agent at boot), and a catalog failure reports the tail of the real output.

## 4. GitHub Codespaces — the agent's computer

Repo `jadeniran21/Computer-drive`, codespace `blackthorn-gx694jr47v9x3vg57` (2-core
`basicLinux32gb`). Verified in-container: `uname -s` = Linux, kernel `6.8.0-1064-azure`, workspace
`/workspaces/blackthorn` (`/workspace` linked to it), agent v1.1 registered with the app
(`registered as blackthorn-gx694jr47v9x3vg57`).

Capabilities proven by real use: command execution, file write/read, and the earlier 11-phase
harness (install, git, package installs) through the same `/computer/*` surface. Nothing inside the
container is reachable from the internet: the agent dials out and long-polls
`/api/codespaces/pull` (25 s hold); the devcontainer declares **no** forwarded ports. Registration
is authenticated with the GitHub token GitHub injects into the codespace, verified against `/user`
and the codespace's owner.

## 5. Power management — start on demand, stop when idle, never mid-task

- Auto-start: the live runs above began with the machine `OFF` and status went
  `OFF → STARTING → ON` only once a tool needed it.
- On-demand wait: a first boot, or any boot after the devcontainer changed, rebuilds the image and
  sits in `Provisioning` for many minutes — waited for (`CODESPACE_START_TIMEOUT_SECONDS=1200`),
  reported as `STARTING`, never declared `ON` early.
- Liveness: a cached registration token is not proof of life; the machine counts as up only when
  the container answers (presence or a cheap authenticated call). This is what the first live run
  caught (`did not become available within 420s` while the panel said ON).
- Never mid-task: `stop` is refused while a run is executing (`_runs`), a tool call is in flight
  (`_inflight`) or a job is queued.
- Auto-stop: after the 11:56 run the idle clock climbed 40 → 85 → 131 → 176 s and the machine
  stopped at the 300 s mark (see the log below); GitHub reported `Shutdown` while the disk — and
  `handoff.txt` — stayed.
- Adoption: a redeploy forgets the codespace name while GitHub keeps the machine (and meter)
  running. The power loop now adopts it from the agent's own polls (falling back to GitHub,
  throttled), reports it `ON`, seeds its idle clock from GitHub's `last_used_at`, and stops it.
- One machine, no waste: creation checks for an existing codespace first, deletes duplicates, and a
  redundant probe codespace was deleted. Polling no longer resets the idle clock (it used to keep an
  idle machine awake for ever).

## 6. Header: Codespaces usage controller

Rendered from `/api/codespaces/usage` + `/api/codespaces/status` (refreshed on an interval and on
every action), ids `codespaces-button`, `codespaces-panel`, `codespaces-hours-left`,
`codespaces-remaining`, `codespaces-{start,stop,restart,banner}` in the served bundle.

Live figures at 11:58 UTC:
```json
{"hours_total": 60.0, "hours_used": 0.34, "hours_remaining": 59.66, "percent_used": 0.6,
 "usage_source": "measured", "month": "2026-11", "machine": "basicLinux32gb", "cores": 2,
 "state": "ON", "agent_using": false}
```
Remaining hours are the prominent figure. `hours_total` is GitHub's included 120 core-hours ÷ 2
cores (override `CODESPACES_CORE_HOURS` / `CODESPACES_MONTHLY_HOURS`); `hours_used` is **measured**
from observed running intervals recorded by the app — no placeholder. GitHub's billing API answers
for personal accounts only with a token carrying the `Plan` permission (403 without it), so the
source is labelled rather than invented.

## 7. Regression suite

`python3 -m pytest -c blackthorn/tests/pytest.ini blackthorn/tests` → exit 0, **325 passed,
5 skipped** (browser/E2E tests skip where Playwright is absent). UI: `tsc` clean, `vite build` OK,
`vitest` 32/32. Integrity: `/api/version` → `{"checked": 61, "drift": [], "extra": []}`.

## 8. Known limits (stated, not hidden)

- Playwright is not installed in this environment: the browser/E2E suite skips here, and the header
  was verified through its API contract, the built bundle contents and the unit tests rather than a
  screenshot.
- `hackingtool` indexing resolves 0 skills (its 183 tools index correctly); it was 0 before this
  migration too.
- Codespaces hours are measured by the app, not read from GitHub billing (permission), and are
  labelled as such in the API.
