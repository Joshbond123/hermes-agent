# vLLM 0.32 multi-node (`mp`) rendezvous bug — root cause, patch, and verification

**Symptom:** `torch.distributed.DistStoreError: Timed out after 601 seconds waiting
for clients. 2/6 clients joined` when serving with
`--distributed-executor-backend mp --nnodes 3 --node-rank {0,1,2}` across three
machines. Only the 2 workers of the local node ever joined, on every node.

**This is not a networking problem.** The rendezvous store vLLM builds for
multi-node `mp` is a *per-node file*, so it can never be shared.

## Root cause

`vllm/v1/executor/multiproc_executor.py` (v0.32.0, `MultiprocessingDistributedExecutor.__init__`):

```python
if aiter_requires_tcp_store():                       # ROCm + AITER only -> False on CUDA
    distributed_init_method = get_distributed_init_method(
        get_loopback_ip(), get_open_port()
    )
else:
    distributed_init_method = get_file_store_init_method()      # <-- always taken
```

`vllm/utils/network_utils.py`:

```python
def get_distributed_init_method(ip: str, port: int) -> str:
    return get_tcp_uri(ip, port)

def get_file_store_init_method() -> str:
    return f"file://{tempfile.gettempdir()}/vllm_dist_{uuid4().hex}"

def aiter_requires_tcp_store() -> bool:
    from vllm._aiter_ops import rocm_aiter_ops
    return rocm_aiter_ops.is_custom_all_reduce_enabled()
```

Consequences:

1. The value is `file://<tmp>/vllm_dist_<uuid4>` — a fresh random path **per call**,
   inside each node's own `/tmp`.
2. `FileStore` is node-local by definition: 2 local workers share it, workers on
   other nodes cannot. Hence exactly `2/6 clients joined`, deterministically.
3. `--master-addr/--master-port` are honoured but never used, because no TCP store
   is created — so **no TCP port is ever bound**. Verified on all three nodes with
   `ss -ltnp`: only the relay (`:9998`) and the tunnel WS server (`:10001`) listen;
   `:29501` never appears.
4. Even the TCP branch is node-local (`get_loopback_ip()` + `get_open_port()`), so
   flipping that condition would not fix multi-node either.
5. **There is no environment variable to override it** — `vllm/envs.py` has no
   store switch.

## Patch

Patch the symbol *after* import and *before* the executor is constructed. The
executor does `from vllm.utils.network_utils import get_file_store_init_method`,
so the name is a module global resolved at call time — patching
`vllm.v1.executor.multiproc_executor.get_file_store_init_method` is sufficient.
Workers receive the resolved string as an argument, so the patch does not need to
propagate into spawned processes.

```python
import vllm.utils.network_utils as _nu
import vllm.v1.executor.multiproc_executor as _mp

def _tcp_init_method():
    return "tcp://%s:%d" % (os.environ["NP_STORE_ADDR"], int(os.environ["NP_STORE_PORT"]))

_nu.get_file_store_init_method = _tcp_init_method
_mp.get_file_store_init_method = _tcp_init_method

from vllm.entrypoints.cli.main import main
sys.argv = ["vllm", "serve"] + sys.argv[1:]
sys.exit(main())
```

Launch with this as the entrypoint instead of `vllm serve`.

`NP_STORE_ADDR` is the master address as seen by each rank:

| rank | NP_STORE_ADDR | why |
|---|---|---|
| 0 | `127.0.0.1` | store binds loopback; local workers connect directly |
| 1..N | master's routed shared IP | routed by the LD_PRELOAD hook to the owner's relay, which dials loopback |

Note: pointing rank0 at the shared IP instead would route rank0's *own* workers
through the relay (client -> local relay -> tunnel -> own WS server -> loopback),
turning a local connect into a 4-hop round trip.

## Verification (GPU-free, `test_store_patch.py`)

```
[A] PASS  per-node paths differ -> no shared rendezvous possible
     node1: vllm_dist_d81ed912e03941eaaa...
     node2: vllm_dist_5cf19310d32343e9b5...
[B] PASS  executor now returns tcp://127.0.0.1:29501 (was file://...)
[C] keys seen by master: 6 / 6
    per-rank: ['joined', 'joined', 'joined', 'joined', 'joined', 'joined']
[C] PASS  6/6 clients joined via TCPStore (the '2/6' symptom is gone)

RESULT: ALL PASS -> fix premise verified without a GPU
```

- **[A]** reproduces the defect: two calls mint different per-node `file://` paths.
- **[B]** mirrors vLLM's import style exactly and proves the patched symbol is what
  the executor resolves (`file://...` -> `tcp://127.0.0.1:29501`).
- **[C]** runs a real `torch.distributed.TCPStore` with 6 client processes
  (2 per virtual node) and confirms **6/6 join** — the `2/6` symptom is gone.

## Remaining unknown

The patch is verified in isolation; the full 3-account cluster run (store join ->
NCCL full-mesh -> token generation) is still pending because Kaggle GPU batch
sessions were refusing to start (phantom "Maximum batch GPU session count of 2
reached" on three accounts, silent non-start on the others; CPU sessions on the
same accounts still start).

Also required on Kaggle specifically: force IPv4 on tunnel connections
(`websockets.connect(..., family=socket.AF_INET)`) — there is no IPv6 route and
`getaddrinfo` returns AAAA records (`OSError(101, 'Network is unreachable')`).
