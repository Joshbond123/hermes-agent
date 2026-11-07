# Migration acceptance — NVIDIA Nemotron 3 Ultra + GitHub Codespaces

Date: 2026-11-05 · app version 5.5.7 · commits `66730a4ab1` … `131f0c432a` (branch `main`)
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

Live tasks, all against the deployed app (proof files `live_task_proof{2,4,5}.json`): a 4-call run,
a 3-call skills run, and a 3-call final run at 12:33 UTC that reported `whoami` = `codespace`,
`pwd` = `/workspaces/blackthorn` and wrote/read `work/final_check/result.txt`. Example (11:38 UTC,
4 tool calls, finish `stop`):

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
- Auto-stop: the idle clock climbs whenever nothing is happening and resets only on real work.
  Observed live (`last_activity` in brackets): `idle 3 → 44 → 84 → 124 → 165` after a run
  (`run_finished`), then reset by genuine work from the owner's own session
  (`tool_done:/computer/exec`, `run_touched`). A machine abandoned by a redeploy was stopped by the
  app at 4000+ s idle and GitHub reported `Shutdown`. The 300 s → stop rule itself is pinned by
  `test_it_stops_when_genuinely_idle_and_keeps_the_disk`; a fully unobserved 300 s live cycle was
  not available because the owner was using the product at the time (and should not be cut off).
- No stuck states: a start that GitHub accepted but did not act on is re-issued (3 tries, 60 s
  apart) and an abandoned `STARTING` is reconciled to `OFF` — both found live and pinned by tests.
  The idle clock now records *what* touched it (`last_activity`), which is how the polling
  regression and the owner's own traffic were told apart.
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

`python3 -m pytest -c blackthorn/tests/pytest.ini blackthorn/tests` → exit 0, **328 passed,
5 skipped** (browser/E2E tests skip where Playwright is absent). UI: `tsc` clean, `vite build` OK,
`vitest` 32/32. Integrity: `/api/version` → `{"checked": 62, "drift": [], "extra": []}`.

## 8. Known limits (stated, not hidden)

- Playwright is not installed in this environment: the browser/E2E suite skips here, and the header
  was verified through its API contract, the built bundle contents and the unit tests rather than a
  screenshot.
- `hackingtool` indexing resolves 0 skills (its 183 tools index correctly); it was 0 before this
  migration too.
- Codespaces hours are measured by the app, not read from GitHub billing (permission), and are
  labelled as such in the API.


---

# Round 2 — streaming UI, task resilience, rate limits, model switching (v5.6.0)

Verified against the live deployment (`https://blackthorn.onrender.com`, commit `53e2489e9f`).

## Execution record no longer appears in the chat
Root cause was structural, not cosmetic: the verified tool ledger was appended to **assistant-role**
content when a conversation was rebuilt, so the model read server bookkeeping in its own voice and
reproduced it in answers. It now travels as one labelled machine block in the system layer (capped at
a third of the context budget, newest facts first), and `store` scrubs any server block both when an
answer is written and when stored messages are served.

Live proof (12:33 UTC run): five tool calls, answer quoted the real file contents, and
`GET /api/chat/sessions/…` contained no `Execution record` / `Do not redo a step` text. Pinned by
`test_the_execution_record_never_reaches_the_chat`,
`test_the_agent_still_receives_the_record_of_earlier_tool_calls` and
`test_a_model_that_echoes_the_record_cannot_put_it_in_the_transcript`.

## Rate limits (HTTP 429)
`429` is now a first-class condition: `Retry-After` is honoured (header, `x-ratelimit-reset`, or the
JSON body), otherwise exponential backoff with jitter bounded at 120 s. A per-endpoint gate allows one
request at a time (`BLACKTHORN_MODEL_CONCURRENCY`), enforces a minimum interval and shares a cooldown
so parallel runs stop amplifying each other. The user sees "The model provider is throttling requests.
Waiting Ns — your task is preserved", never the raw API error. Pinned by four tests, including "a 429
is waited out and the task continues without replaying tools".

## Task lifecycle
A transient fault no longer ends a task: `AgentRun` supervises the loop and resumes from the exact
message state it reached (tools already run stay run, results stay in the messages) after the delay
the provider asked for, bounded by `BLACKTHORN_RESUME_ATTEMPTS` (default 6; 0 disables). Every resume
emits `internal.checkpoint` and a plain-language notice. The existing SSE contract already kept a run
alive across a page refresh (`/api/chat/runs/{id}/events?after=` replays from the last sequence).

*Not verified end-to-end here: a multi-hour run surviving a real Render restart* — the run registry is
in-process, and a restart marks the turn `interrupted` with a resume note rather than losing work.

## Model switching
`GET /api/models` lists the catalogue with honest availability; `PATCH /api/chat/sessions/{id}
{"model": …}` switches a conversation's model and returns `applies_to: "next_turn"`, plus an
`active_run` note when a task is executing (that task keeps the model it started with). Live: switching
to the unconfigured Qwen returned `model_unavailable` with the reason, an unknown id returned
`model_unknown`, and a valid switch kept the same session id and transcript. The header picker shows
the active model, unavailable models with their reason, and the mid-run note.

## Qwen3.8-27B-Uncensored (item 5) — prepared, not deployed
The model is real: `orcarouter/Qwen3.8-27B-Uncensored-GGUF` → `…-Q4_K_M.gguf`, 16.81 GB, gated
(apache-2.0). It does **not** fit a single 16 GB T4, so it needs both GPU cards.

`blackthorn/scripts/serve_qwen.py` is the serving path: it reuses weights already on disk (Kaggle keeps
`/kaggle/working`), refuses a one-T4 host with a reason instead of an OOM, builds the llama.cpp command
(`-ngl 99`, `--tensor-split 1,1`, `--jinja` for tool calling) and can self-test a real tool call. Its
command construction is unit-tested.

What is deliberately **not** done: standing the Kaggle GPU infrastructure back up. The previous round
removed it end to end on explicit instruction ("nothing Kaggle GPU-related may be left behind"), its
credentials were deleted from Render, and a Kaggle notebook is an interactive session that must be
started and kept alive by its owner with a public tunnel. Pointing this script at a GPU host and
setting `QWEN_BASE_URL` + `QWEN_MODEL` makes Qwen appear in the picker as a live, selectable model —
that switch is already implemented and tested on the Blackthorn side.

## Suite
339 passed, 5 skipped. UI: `tsc` clean, `vite build` (new bundle `index-dwN43Rx7.js` /
`index-DIcam0T9.css`), `vitest` 32/32.

---

# Round 3 — Qwen on a real Kaggle GPU (v5.8.2)

## What is proven, with evidence

**Installed once, permanently.** `POST /api/gpu/install` pushed a private Kaggle kernel that
downloaded `orcarouter_Qwen3.8-27B-Uncensored-Q4_K_M.gguf` — **17.77 GB in 297 s** — into the
kernel's output, and the run completed. Kaggle keeps that output permanently and mounts it read-only
into later sessions. Verified twice, on two different accounts
(`joshbond123/blackthorn-qwen-install` and `happink/blackthorn-qwen-install`), by reading the kernel's
own log and file listing through the API.

The gated `orcarouter` repository returns **403** to this Hugging Face token (the account has not
accepted its terms), so the installer fell back automatically to `bartowski`'s requantisation of the
same model — same Q4_K_M, ungated. The token is used when it works; the fallback is explicit in the
log, not hidden.

**No re-download on restart — observed, not assumed.** Two GPU sessions logged:

```
weights from the persistent cache:
/kaggle/input/notebooks/joshbond123/blackthorn-qwen-install/qwen/orcarouter_Qwen3.8-27B-Uncensored-Q4_K_M.gguf (17.77 GB)
GPU: Tesla T4, 15360 MiB | Tesla T4, 15360 MiB
```

The serve kernel contains no code path that can reach Hugging Face — a test asserts `huggingface.co`
does not appear in it — so "reused the existing files" is structural, not a behaviour that might drift.

**Controls, timers and accounting are live.** `/api/gpu/status`, `/api/gpu/usage`, `/api/gpu/start`,
`/api/gpu/stop`, `/api/gpu/settings` answer on the deployed service. The usage panel reports
`hours_total` 30.0 (labelled as Kaggle's published minimum, Saturday 00:00 UTC reset, correctable),
measured `hours_used`, `hours_remaining`, `percent_used` and the live session duration. Kaggle exposes
no usage API, so no figure is taken from it or invented in its place.

## What is not proven yet, and why

The model has not yet answered a request through Blackthorn. Each live attempt advanced further and
exposed a real defect, all of them now fixed and tested:

1. Kaggle answers a bad push with **HTTP 200 and `{"hasError": true}`** — the installer reported
   success while creating nothing.
2. Kernel slugs must be **owner-qualified**, and Kaggle's identity is **case-sensitive**
   (`userName=Joshbond123` → "Permission 'kernels.get' was denied").
3. llama.cpp Linux CUDA builds are **`.tar.gz` under `b*` tags**; `/releases/latest` is a decoy.
4. The binary must run **where its `.so` files are** (`LD_LIBRARY_PATH`), not copied elsewhere.
5. `/api/gpu/register` and `/api/gpu/heartbeat` sat behind the owner-cookie gate, so a kernel on
   Kaggle could never register.
6. The idle timer counted **boot time as idleness** and killed a session mid-load.
7. The mount was named after the **primary account** before the fallback key had switched accounts.
8. A revoked key left the panel blaming "quota"; a spare account is now used automatically.
9. A long model load produced **no heartbeat**, so the panel declared a healthy session dead.
10. llama.cpp **memory-maps** its model, and the persistent cache is a network mount, so loading took
    tens of minutes of silence. The kernel now stages the file on local disk first.

## Blocking issue for the next round

**The Kaggle API keys in the Blackthorn document are no longer valid.** Mid-session the primary key
(`Joshbond123`) went from working to `HTTP 401`, and later `happink`'s key did too — both are refused
now, for every endpoint, including reads that had worked minutes earlier. Nothing in the code can
work around a rejected credential.

To finish the last mile, a current Kaggle API key is needed in the Blackthorn doc (`KAGGLE_USERNAME`
/ `KAGGLE_KEY` on Render). Then: `POST /api/gpu/install` (once, ~10 min CPU, no GPU quota) and
`POST /api/gpu/start`. The remaining unverified steps are the tunnel registration, the tool-call self
test and a chat answered by Qwen — all guarded by tests, none yet observed succeeding on Kaggle.

## Tests

Backend suite green (including 31 tests for this feature: power semantics, persistence, idle policy,
quota-week maths, reconnect, Kaggle failure semantics, credential fallback, account resolution).
UI: `tsc` clean, 38/38 vitest, bundle rebuilt.
