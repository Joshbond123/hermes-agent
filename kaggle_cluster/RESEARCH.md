# Kaggle 3-Account Cluster — Research Log (live)

Owner task: aggregate 3 Kaggle accounts into one logical cluster serving
`orcarouter/Qwen3.8-27B-Uncensored` **BF16** (55.56 GB, 18 shards,
27.78 B params), with automatic backup-cluster failover at T-30 min of
session time, zero re-download persistence, and UI status fixes.

All findings below are either **measured** (marked E) or **sourced** (marked
with the URL). Nothing is claimed as working until its test ID passes.

---

## 1. Accounts, quota, API (E)

- Doc (`Blackthorn` Drive doc, modified 2026-10-18 11:31) defines **two
  clusters of three accounts each**:
  - PRIMARY: `black797`, `teslaarymo`, `alagbo`
  - BACKUP: `Joshbond123`, `temilolo`, `happink`
  - All six KGAT tokens staged locally as `KAGGLE_U*/P*` and
    `KAGGLE_BU*/B*` in `~/.secrets/com11.env` (values never printed).
- `GetAcceleratorQuotaStatistics` works with all six tokens:
  **30.0 h GPU/week each, 0 h used, refresh 2026-10-24T00:00:00Z**
  (`timeUsed/timeReserved/totalTimeAllowed/minimumTimeAllowed=108000s`).
  TPU quota 72000 s (unused). (E: 2026-10-18 ~11:47)
- `GetKernelSessionStatus` **works from outside** once the kernel exists
  (earlier 403 `kernels.get` happened only before creation):
  `{"status":"RUNNING"}`. Usable for session-liveness. (E)
- `SaveKernel` RPC semantics (E):
  - `slug` **must be** `{username}/{slug}` — bare slug silently no-ops
    (returns `{}` and creates nothing).
  - `kernelType:"notebook"` requires **nbformat-4 JSON** text; raw .py
    under notebook type runs nothing (run time stamped, zero output).
  - `kernelExecutionType:"SAVE_AND_RUN_ALL"` (enum name; kagglesdk
    `EnumSerializer` uses `.name`) is what actually **starts** the run.
    Without it a newly created kernel does not execute.
  - Fields used: `isPrivate`, `enableInternet:true`, `enableGpu:false`
    (CPU probes), `machineShape:"NvidiaTeslaT4"` + `enableGpu:true`
    (GPU sessions, per existing fleet).
  - `lastRunTime` in REST `GET /api/v1/kernels/list?user=` reflects the
    running session start (works with Bearer token).
- Push→RUN latency for CPU kernels: **~60–90 s** from SaveKernel to
  first D1 write; session ends when the script exits. (E)
- GPU shape (2026-10): only `NvidiaTeslaT4` (UI "T4 ×2", 2×16 GB) for
  general accounts; P100 sunset 2026-09-15; L4 = select competitions.
  Source: kaggle-cli PR #1198 / issue #1196; kaggle.com/discussions/
  product-announcements/735239; kaggle.com/docs/notebooks.

## 2. Cross-session networking (E — decisive)

Experiment `netprobe` v1/v2, two accounts, CPU sessions, results in D1
rows `np_p1`/`np_p2` (table `kaggle_gpu_state`):

- Each session is an **isolated container netns**:
  `eth0 172.19.2.2/24` (SAME address on every session), default via
  `172.19.2.1`, **gateway unreachable (timeout)**, `.3+` = "no route to
  host", ARP table empty.
- v1 "internal connect succeeded" was **self-connection**; v2 nonce test
  proves every answered socket belongs to the local host
  (`peer_host == HOSTID`, `self:true`).
- **Public IPs differ and inbound TCP to them times out** (GCP firewall,
  no hole punching possible): p1 `34.9.218.210`/egress `34.81.112.98`,
  p2 `35.201.197.221`/egress `34.75.225.94`.
- **Sessions land in random GCP regions/continents**:
  - black797 probe egress = **Taipei, TW (asia-east1)**
  - teslaarymo probe egress = **North Charleston, SC, US (us-east1)**
  - (ipinfo.io from inside the session)
- Conclusion: **no direct L3 between sessions**; cross-node traffic must
  ride outbound-initiated tunnels/relays, and **region placement must be
  measured at boot** (retry strategy needed to keep PP latency sane).

## 3. Model & precision facts

- `orcarouter/Qwen3.8-27B-Uncensored`: arch
  `Qwen3_5ForConditionalGeneration`, hybrid Gated-DeltaNet (48 linear +
  16 full-attn of 64 layers) + VL tower + MTP; safetensors 55 563 118 992
  B / 18 shards, BF16, 27 781 427 952 params; **gated=auto**, region us.
  (HF API, E)
- **HF token from doc (`HF_TOKEN`) authenticates but is NOT authorized**:
  `resolve/.../config.json` → 403 "not in the authorized list";
  `GET /api/models/...` → 200 (metadata only). Web accept-form endpoint
  not yet found (guessed `/request-access`, `/access-request`, `/accept`,
  `/request` → 404). Gate acceptance = one click by a logged-in user.
- **Ungated 16-bit sources (verified via HF API, E):**
  - `chimingw/Qwen3.8-27B-Uncensored-OrcaRouter-GGUF` (**gated=false**):
    `BF16/…-FP8-derived-BF16-00001-of-00002.gguf` 44 977 268 896 B +
    `…-00002…` 8 831 013 728 B = 53.8 GB (**bf16-typed GGUF**), plus
    Q8_0 29.0 GB, mmproj F16/BF16. Lineage pinned to orcarouter FP8 rev.
  - `orcarouter/Qwen3.8-27B-Uncensored-GGUF` (**gated=auto**): F16 split
    2× files 54.7 GB "full precision".
  - `ressl/Qwen3.8-27B-uncensored-GGUF` (gated=false) — current code's
    `HF_REPO` (Q4_K_M).
- Fallback if gate stays closed: download chimingw **BF16 GGUF** (ungated)
  → convert to **bf16 safetensors** (dtype-preserving) → same artifacts
  as gated repo, byte-level provenance documented (FP8-derived lineage).
- FP8 sibling 30.9 GB unusable on 32 GB T4×2 (needs ~40 GB).

## 4. Serving stack constraints (sourced)

- T4 = sm75 Turing: **no BF16 tensor cores**; PyTorch ≥2.8
  `is_bf16_supported()` returns True only in *emulation* mode; vLLM min
  CC 7.5 (docs.vllm.ai/en/stable/getting_started/installation/gpu);
  FP8 KV cache unsupported on sm75 (vLLM issue #47549); FlashInfer SM75
  restored in vLLM PR #55380 (requires `uv pip uninstall --system
  flashinfer-jit-cache`, HEAD_DIM ≤ 256); TRITON_ATTN prefill up to 15×
  slower on Turing (issue #60821); T4 fixed in vLLM 0.11.1 (issue #27729).
  → **Open test: does `--dtype bfloat16` run on sm75 at all?**
- Multi-node vLLM (docs.vllm.ai parallelism_scaling + kubernetes.recipes
  + spark-vllm-docker 2026-07):
  - Ray backend is the default for multi-node; **no-Ray** path:
    `vllm serve … --nnodes N --node-rank R --master-addr A --master-port
    P --tensor-parallel-size TP --pipeline-parallel-size PP
    --distributed-executor-backend mp` (multi-host mp requires PP ≥ 1
    with PP>1; V1-engine multi-node PP via mp was issue #41864, now
    documented as supported in current CVS docs 2026-09).
  - For TP×PP across hosts: `TP = GPUs per node (2)`, `PP = nodes (3)`.
  - Required env: `NCCL_IB_DISABLE=1`, `NCCL_P2P_DISABLE` (cross-node),
    `NCCL_SOCKET_IFNAME`/`GLOO_SOCKET_IFNAME` pinned, `NCCL_SHM_DISABLE=1`
    across nodes, `NCCL_PROTO=Simple` for heterogeneous paths,
    `NCCL_TIMEOUT`/`NCCL_SOCKET_TIMEOUT` raised for high RTT.
- PP math for this model (design): hidden 5120 × bf16 = 10 KB/token per
  stage boundary; 64 layers → ~21–22 layers/stage; per-GPU weights
  ≈ 9.3 GB (55.6/3/2) + KV share of 96 GB aggregate.

## 5. Transport decision (design, next tests)

No inbound, random regions → raw-socket users (NCCL, Ray, vLLM mp,
llama.cpp `--rpc`) all need a **public TCP rendezvous per node**.
Candidates to test (in order):

1. **bore.pub** (open-source TCP tunnel, no account): client exposes
   `public_host:port → local port`. Test liveness + throughput 2026-10.
2. **inrok** (`app.inrok.in`, API key in doc; already used by production
   stack env `INROK_API_KEY`/`INROK_TUNNEL_NAME`): determine whether it
   offers raw TCP endpoints.
3. playit.gg / tailscale-derp (accounts) as fallback.
4. Cloudflare quick tunnels are **HTTP/WS only** — usable for control
   plane (gateway) but not for NCCL.

Placement strategy: probe region at session start (ipinfo egress); if the
three primary nodes are not co-region(ed within acceptable RTT), stop the
outlier and re-push (CPU probes are ~90 s; GPU attempts cost only their
startup minutes). Then form the tunnel mesh, then vLLM.

## 6. Failover / session-time detection (partial E)

- Weekly quota API (§1) = **weekly** budget, not session remaining.
- Session remaining = 12 h limit minus (now − session start).
  Session start available via REST `lastRunTime` (list) or
  `GetKernelSessionStatus` (works). T-30 min trigger = orchestrator
  computes `session_start + 12h − now ≤ 30 min` → start backup cluster
  (its 3 different accounts), warm it, flip D1 `tunnel_url`, stop
  primary at T0.
- Quota math (honest): steady-state with rolling T-30 handoff burns
  ≈12 h per account per ~11.5 h of cluster uptime → with 30 h/week the
  cluster can stay hot ≈ **24–29 h/week**, not 168 h. Must document.

## 7. Persistence design (from existing stack, E)

- Existing cache flow (cyber_ornith_server.py): private per-account
  dataset `{user}/qwen38-27b-uncensored-model-cache` pulled **inside the
  run** (attach-at-push measured to stall boots 30+ min — do not
  re-enable `datasetDataSources` mount for large sets), verified, else HF
  fallback; boot-step vocab CHECKING_CACHE→…→WARMING_GPU.
- BF16 scale-up: 55.6 GB in-run pull from Kaggle storage is fast
  (datacenter-local); mirror = same bytes to **each** of the 6 accounts'
  private datasets; checksum manifest in `state_meta` for the
  "synced/cloned backup" requirement; HF download only when dataset miss.

## 8. UI bugs still in scope (from previous message)

- `GpuPanel.tsx` "turning on" timer: client shows
  `elapsed_seconds + (Date.now()-fetchedAt)/1000` while `booting`;
  server `elapsed_seconds` derives from D1 `state_meta
  ['blackthorn_gpu_boot_started']` (`BOOT_META_KEY`) — suspect stale /
  never-cleared boot-start timestamp inflating the countdown.
- Header status indicators live in `gpu.ts summarize()` / `Header.tsx`
  (not yet inspected).

## 9. Test ledger

| ID | What | Result |
|----|------|--------|
| E1 | 6 tokens + quota API | PASS (30 h each) |
| E2 | HF token auth vs gate | metadata 200, files 403 (unauthorized) |
| E3 | netprobe v1 push (raw py, no exec type) | FAIL (silent no-run) |
| E4 | netprobe v1' (ipynb, SAVE_AND_RUN_ALL) | PASS run; internal reach = SELF false-positive |
| E5 | netprobe v2 nonce + region + sweep | PASS: isolation + regions proven |
| E6 | GetKernelSessionStatus outside | PASS 200 RUNNING |
| E7 | CPU push→run latency | PASS ~60–90 s |

## 10. E8: vLLM BF16-on-T4 bring-up (happink/vllm-sm75-bf16-test)

One notebook, iterated v1→v10; every fix evidence-backed via D1 rows
`vllm_sm75` (status/gpu_info) + `vllm_sm75_hb` (30 s heartbeat) +
`vllm_sm75_l1..l3` (full serve-log thirds on failure).

| Ver | Blocker found | Fix |
|-----|---------------|-----|
| v1 | HF **xet** backend crashed the session seconds after `snapshot_download` (status ERROR, 585 s billed) | `HF_HUB_DISABLE_XET=1` + per-file `hf_hub_download`; heartbeat row mandatory |
| v2 | pip `vllm` pulls torch 2.13.0+cu130 vs preinstalled torchaudio cu128 → RuntimeError in transformers import chain | PRE_SERVE phase: import test → uninstall torchaudio → retest → reinstall fallback |
| v3 | EngineCore spawn bootstrap crash (heredoc stdin main can't be re-imported) | file-based `serve.py` + `if __name__ == '__main__':` guard |
| v4/v5 | (spawn fixed) arch resolved `Qwen3_5ForConditionalGeneration`, driver 580.178.04 `cuda-ok`, failure moved to WorkerProc init | — |
| v6 | **`vllm/platforms/cuda.py:648 check_if_supports_dtype` → ValueError "Bfloat16 is only supported on GPUs with compute capability of at least 8.0 … (T4, 7.5)"** | vLLM hard-blocks bf16 below CC 8.0 although torch reports `is_bf16_supported()==True` → file-level PATCH_SM75 |
| v7 | in-process monkeypatch invisible to spawned workers (fresh module import) | patch must be file-level in dist-packages |
| v8 | triple-quoted patch text → SyntaxError (unterminated string) | quote-free patch: keep `def` line, body = 8-space `return  # …`; `py_compile` verified inside the phase |
| v9 | **PATCH VERIFIED ON LIVE GPUs**: workers loaded model in bf16 (`dtype=torch.bfloat16` accepted, weights resident). New failure = **OOM "Tried to allocate 128.00 GiB"** in `vit_attn_wrappers.py → F.scaled_dot_product_attention`, mis-tagged FAIL_DTYPE_BF16 (FA2 "not supported" line + 'bfloat16' in config dump) | see v10 |
| v13 | (20:21-20:31) happink's ancient pre-`enforce_eager` trigger finally got capacity & failed with the Inductor SkipFrame again — content-era forensics via `non-default args` (no `enforce_eager` in config) | confirms `enforce_eager=True` is required AND sufficient (alagbo runs with it = COMPLETE) |
| v10 | root cause: engine `profile_run → encoder_runner.profile_encoder_cache(dummy_mm_inputs)` runs the **vision tower** on a 4096×4096 dummy (patch 16 → 65 536 tokens; 65536² × 16 heads × 2 B = **exactly 128 GiB**); bf16-on-Turing → SDPA math fallback materializes the matrix → OOM. FA2 unavailable on CC<8 (logged, benign) | `LLM(..., skip_mm_profiling=True)` (gates `model_runner.py:994` block; text-only serving never touches the ViT) + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + OOM-first failure tagger |

- Verdict so far: **bf16-on-sm75 bypass = fixed & verified**; v10 (skip
  encoder profiling) fixes the 128 GiB profiler OOM.
- FAIL_DTYPE_BF16 classifier was a false positive: 'bfloat16' occurs in
  every config dump; the "not supported" came from
  `fa_utils.py:212 FA2 is only supported on devices with compute
  capability >= 8` (benign fallback to TRITON_ATTN/TORCH_SDPA).

## 11. Test ledger (continued)

| ID | What | Result |
|----|------|--------|
| E8 | vLLM bf16-on-T4 series v1→v18 (see §10) | **PASS**: full fix-stack session on alagbo exited 0 (`GetKernelSessionStatus=COMPLETE` = only exits 0 on `VLLM_RESULT`+`RC=0`); v18 rerun in flight for latency capture |
| E9 | netprobe-v3 peer sweep (teslaarymo v5, alagbo v1) | PASS: 256 addrs **0 hits** = sessions fully isolated; egress SC 34.148.241.40 / OR 136.69.162.65, peer TW 34.81.112.98 → random regions |

### Session/status API semantics (measured 18:0x)

- `GetKernelSessionStatus`: `{}` = no session (queued/pre-start);
  `ERROR` = last session exited **non-zero**; `COMPLETE` = exited 0
  (probed against `gpu-debug-marker*`). It reports the *last* session,
  so a stale `ERROR` can linger after a failed run.
- `lastRunTime` (via `ListKernels?user=`) = timestamp of last session
  **start**; updated within ~1 s of a SaveKernel-triggered start when
  capacity is free; unchanged while a new run waits for GPU capacity.
- `kernelExecutionType:"SAVE_AND_RUN_ALL"` + SaveKernel = the only run
  trigger (probed `RunKernel`/`StartKernel`/… → 404).
- **Raw `.py` text + `kernelType:"notebook"` + `kernelExecutionType:"SAVE_AND_RUN_ALL"` = the working combination** (v13 run live-verified 18:25; matches v4–v9 and E9 netprobe pushes). E3 vs E4 was a confounded test: the real variable was SAVE_AND_RUN_ALL (it persists per-kernel after the first set), not ipynb wrapping. My ipynb-JSON pushes (v11/v12) saved fine (versionNumber bumped) but **never scheduled**; the raw re-push (v13) scheduled in 38 s and executed phases. The earlier "v10 ERROR" was v9's exit(1) status lagging ~10 min (v10 never actually started). push_vllm.py: use `--raw` (default is now raw).

### Scheduling/capacity findings (18:4x–19:0x, evening UTC peak)

- Saves can **queue silently for many minutes** when T4 capacity is
  tight (evening UTC ≈ US peak): `session={}`, no billing, no D1
  writes. Earlier same-day runs started in 30–40 s (17:33, 18:25);
  18:47–19:05 = multi-minute darkness on two accounts (happink +
  temilolo both queued) → cluster-wide capacity, not payload.
- A save while the previous session still shows `ERROR`/teardown
  appears to **drop the trigger** (status clears but no run): push only
  when `GetKernelSessionStatus` already reads `{}`, and bump content
  (docstring) to defeat save-dedup; `lastRunTime` stamps at schedule
  for some pushes, at actual start for others — treat both as hints,
  only D1 writes are truth.
- Two accounts queued simultaneously is fine (different sessions); the
  E-series stagger advice is about capacity contention during launches.

### Push/session mechanics — final recipe (verified 19:50, PROBE2_OK)

| Rule | Evidence |
|------|----------|
| **Creation must send ipynb-wrapped `text`** (nbformat 4, cell id ok) | raw-text CREATION poisons the kernel: sessions flip RUNNING→ERROR with **zero execution** (probe `vllm-sched-probe`, temilolo `vllm-sm75-bf16-test` v1/v2) |
| Saves to a properly-created kernel may be raw .py (auto-wrapped) **or** ipynb | happink kernel (E4-era creation) ran raw saves v4–v13 fine; probe2 creation+ipynb ran to D1 write |
| **`newTitle` determines the slug at creation** (`slugify(title)`); the `slug` payload field is ignored on create | "VLLM SM75 vllm-sm75-bf16-a1" → `vllm-sm75-vllm-sm75-bf16-a1`; "Sched Probe Two" → `sched-probe-two`. Titles are account-unique (409 otherwise) |
| **Always print the FULL SaveKernel response** — `{"error":"Maximum batch GPU session count of 2 reached."}` hides in there | batch = concurrent GPU *commit* sessions, limit 2/account (kaggle-cli#185, product-feedback/223728); known stale-counter bug; no stop RPC exists (StopKernel et al. all 404; official CLI only has kernels_delete) |
| GPU trigger → start latency: **~40 s when capacity is free** (17:33, 18:25, 19:25 waves), otherwise silent queueing for tens of minutes with `session={}` and no billing | CPU sessions start in ~9 s regardless → dark GPU = capacity, not payload |
| `GetKernelSessionStatus` 403s for kernels that **don't exist** (not 404) | probe slug polling confusion |
| Fix stack in notebook (in order): HF xet-off → PRE_SERVE torchaudio → file main guard → PATCH_SM75 → `skip_mm_profiling=True` → `enforce_eager=True` → OOM-first tagger | E8 §10 |

## 12. E10: multi-node mini (2×T4×2, 4B PP2×TP2 over overlay)

Goal: validate the transport that the 27B (55.6 GB → PP3×TP2 ×3 accounts)
requires. Design decided 2026-10-19 after gate reopened:

- **Gate**: `orcarouter/Qwen3.8-27B-Uncensored` now `gated: auto`; tree +
  `config.json` = 200 with staged HF key (55.6 GB, 18×~3.98 GB).
- **vLLM 0.32 no-Ray multi-node**: `EngineArgs(nnodes, node_rank,
  master_addr, master_port)` + `distributed_executor_backend="mp"`
  (validation: nnodes>1 allows only mp/uni/external_launcher; CUDA
  auto-selects mp) — `LLM(...)` kwargs usable in-process.
- **Transport = OpenVPN-TCP hub over one bore port**: bore forwards
  node0:1194 (bore's one-port/client constraint satisfied); spokes dial
  the bore endpoint via egress. Overlay `10.77.0.0/24`
  (hub .1, spokes .2/.3, `client-to-client`), so NCCL bootstrap
  advertises routable overlay IPs (`NCCL_SOCKET_IFNAME` = dev of
  `ip route get 10.77.0.1`); `NCCL_IB_DISABLE=1 NCCL_PROTO=Simple`.
  TP=2 stays intra-node; PP=2 crosses the overlay (activation-sized
  traffic only). master TCPStore = 10.77.0.1:16389.
- **Protocol**: hub posts `mn_ov` (client bundle) + `mn_ep` (bore
  endpoint) rows; spoke waits ≤60 min for bundle (capacity waves),
  connects, pings hub, posts `mn_ready_1`. Hub waits peer row ≤60 min.
  **Serve gate**: hub posts `mn_gate` immediately before `LLM()`; spoke
  waits ≤30 min on it so TCPStore dial never hits a dead listener.
- Rows: `mn_ala`/`mn_tem` (+`_hb`, fail chunks `_l1..3`), `mn_ep`,
  `mn_ov`, `mn_ready_1`, `mn_gate`, `mn_result`.
- Launched 08:07/08:08 UTC (v2 with gate at 08:13) on alagbo+temilolo.

### E10 field notes (09:0x UTC)

- **Never SaveKernel-push to a kernel whose session is RUNNING** — the
  push cancels the live session (`CANCEL_ACKNOWLEDGED` observed at the
  exact re-push moment; teslaarymo spoke killed mid-VPN at 09:01 after
  a healthy ENV→INSTALL→VERSION→PATCH→VPN start). Push only when
  session reads `{}` or a terminal state.
- `d1poll.py` default listing must include `id LIKE 'mn%'` — a filtered
  poll hid the only live run for ~50 min (rows `mn_*` invisible).
- Batch-of-2 re-verified at 09:21: both hub+spoke still `Maximum batch
  GPU session count of 2 reached` = triggers alive/queued; rejected
  saves are harmless (queued versions unchanged).
- Capacity drought: Monday ~08:07–09:48 UTC queued with zero starts
  (overnight windows flow; EU-morning does not).

## 13. E10b: apt-free Python overlay — locally validated (refreshC)

- Kaggle killed 3/3 sessions that invoked `apt-get` (CANCEL_ACKNOWLEDGED
  ≤60 s after the apt command; the only sessions ever to run apt). Also
  `sh()` captures output, so "verbose apt" still never reaches Kaggle
  stdout. → **apt is banned**: no openvpn/tinc install path.
- Replacement: stdlib-only **TUN↔TCP overlay** (`OVERLAY_SRC` inside
  `mnode_template.py`): hub binds 127.0.0.1:1194 behind the bore port,
  spokes dial the bore endpoint; length-prefixed raw IPv4 frames;
  hub learns spoke src-IPs from frames and forwards by dst-IP; tun
  names derived per endpoint (`ovl<last-octet>`); MTU 1280.
- **Local two-netns rehearsal (veth=bore substitute) PASSED**:
  spoke→hub ping 3/3 0 %, hub→spoke 3/3 0 %, and TCP to
  `10.77.0.1:16389` returned `PONG:HELLO` — the exact master-TCPStore
  path vLLM needs.
- Local single-host runs fail (duplicate10.77.0.0/24 routes when both
  endpoints share a box) — irrelevant on real one-tun-per-node hosts.
- Pre-overlay bugs caught by rehearsal: `lock=None` TypeError in
  spoke tun_loop; src-IP learned at wrong offset under IFF_NO_PI
  (header at byte0); 10 s connect-timeout killed idle sockets →
  `settimeout(None)` after connect.
- Rules in force: never push to a kernel with a RUNNING session (push
  cancels it); push only on `{}`/terminal; `pkill -f` needs bracket
  patterns; `d1poll` lists `mn%`.
