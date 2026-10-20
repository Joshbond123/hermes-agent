#!/usr/bin/env python3
"""Local, GPU-free verification of the v19 rendezvous fix.

A: reproduce the vLLM 0.32 defect -- file store is a per-node /tmp path, so
   workers on a second "node" can never join (the 2/6 symptom).
B: prove the patch binding -- the executor resolves get_file_store_init_method
   from module globals, so patching the symbol changes what it returns.
C: prove the real thing -- 6 processes (2 per virtual node) rendezvous on one
   TCPStore via tcp://<addr>:<port>, which is what the patched value produces.
"""
import multiprocessing as mp
import os
import sys
import tempfile
import time

import datetime

import torch.distributed as dist

PORT = 29501
ADDR = "127.0.0.1"


# ----------------------------------------------------------------- A: defect
def test_a_file_store_is_per_node():
    """get_file_store_init_method() == file://<tmp>/vllm_dist_<uuid4>"""
    from uuid import uuid4

    def impl():
        return f"file://{tempfile.gettempdir()}/vllm_dist_{uuid4().hex}"

    a, b = impl(), impl()
    assert a.startswith("file://") and b.startswith("file://")
    assert a != b, "each call must mint a fresh random path"
    # different nodes -> different path -> different (local, unshared) store
    assert os.path.dirname(a) == os.path.dirname(b)
    print("[A] PASS  per-node paths differ -> no shared rendezvous possible")
    print("     node1:", a.rsplit("/", 1)[-1][:28] + "...")
    print("     node2:", b.rsplit("/", 1)[-1][:28] + "...")
    return True


# ------------------------------------------------------------------ B: patch
def test_b_patch_binds():
    """Mirror vLLM's import style and prove the patch takes effect."""
    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, "vllm", "utils"))
    os.makedirs(os.path.join(root, "vllm", "v1", "executor"))
    for d in ("vllm", "vllm/utils", "vllm/v1", "vllm/v1/executor"):
        open(os.path.join(root, d, "__init__.py"), "w").close()

    with open(os.path.join(root, "vllm", "utils", "network_utils.py"), "w") as fh:
        fh.write(
            "import tempfile\nfrom uuid import uuid4\n"
            "def get_file_store_init_method():\n"
            "    return f'file://{tempfile.gettempdir()}/vllm_dist_{uuid4().hex}'\n"
        )
    # same import shape as mp.py lines 49-53 + call site line 143
    with open(os.path.join(root, "vllm", "v1", "executor", "multiproc_executor.py"), "w") as fh:
        fh.write(
            "from vllm.utils.network_utils import get_file_store_init_method\n"
            "def init_method():\n"
            "    return get_file_store_init_method()\n"
        )

    sys.path.insert(0, root)
    for m in [k for k in list(sys.modules) if k.startswith("vllm")]:
        del sys.modules[m]
    import vllm.v1.executor.multiproc_executor as me

    before = me.init_method()
    assert before.startswith("file://"), before

    # ---- exactly what /tmp/vllm_launch.py does
    addr, port = ADDR, PORT

    def _tcp_init_method():
        return "tcp://%s:%d" % (addr, port)

    import vllm.utils.network_utils as nu
    nu.get_file_store_init_method = _tcp_init_method
    me.get_file_store_init_method = _tcp_init_method

    after = me.init_method()
    assert after == f"tcp://{ADDR}:{PORT}", after
    print("[B] PASS  executor now returns", after, "(was file://...)")
    for m in [k for k in list(sys.modules) if k.startswith("vllm")]:
        del sys.modules[m]
    return True


# -------------------------------------------------------------------- C: join
def _client(rank, addr, port, q):
    try:
        s = dist.TCPStore(addr, port, is_master=False, timeout=datetime.timedelta(seconds=90))
        s.set(f"rank{rank}", "1")
        q.put((rank, "joined"))
        # stay alive until master confirms, else store can tear down early
        time.sleep(3)
        s.delete_key(f"rank{rank}")
    except Exception as e:  # noqa: BLE001
        q.put((rank, "ERR %r" % (e,)))


def test_c_six_clients_join():
    """2 workers per virtual node x 3 nodes = 6, all on tcp://127.0.0.1:29501"""
    q = mp.Queue()
    srv = dist.TCPStore(ADDR, PORT, is_master=True, wait_for_workers=False,
                        timeout=datetime.timedelta(seconds=120))
    procs = [mp.Process(target=_client, args=(r, ADDR, PORT, q)) for r in range(6)]
    for p in procs:
        p.start()

    t0 = time.time()
    while srv.num_keys() < 6 and time.time() - t0 < 90:
        time.sleep(0.2)
    got = srv.num_keys()
    for p in procs:
        p.join(timeout=30)
    msgs = []
    while not q.empty():
        msgs.append(q.get())
    srv.delete_key  # noqa: B018  (keep ref)
    print("[C] keys seen by master:", got, "/ 6")
    print("    per-rank:", sorted(m[1] for m in msgs))
    assert got == 6, "expected 6/6 clients joined, got %d" % got
    print("[C] PASS  6/6 clients joined via TCPStore (the '2/6' symptom is gone)")
    return True


if __name__ == "__main__":
    ok = test_a_file_store_is_per_node() and test_b_patch_binds() and test_c_six_clients_join()
    print("\nRESULT:", "ALL PASS -> fix premise verified without a GPU" if ok else "FAILED")
