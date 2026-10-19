"""E10 multi-node mini test v2 — +serve gate (hub posts mn_gate before LLM init).

Injected at push time: @@CF_ACCOUNT@@ @@CF_DB@@ @@CF_KEY@@ @@HF_TOKEN@@
                       @@TAG@@ @@NODE_RANK@@ @@NNODES@@ @@MASTER@@ @@MODEL@@ @@ROWPEER@@ @@PP@@ @@DL_MODE@@ @@HOLDSECS@@ @@PEERS@@
Rows: @@TAG@@ (status), @@TAG@@_hb, mn_ep, mn_ov (client bundle), mn_ready_*,
      mn_result, fail chunks @@TAG@@_l1..l3.
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error

CF_ACCT = "@@CF_ACCOUNT@@"
CF_DB = "@@CF_DB@@"
CF_KEY = "@@CF_KEY@@"
HF_TOKEN = "@@HF_TOKEN@@"
TAG = "@@TAG@@"
RANK = int("@@NODE_RANK@@")
NNODES = int("@@NNODES@@")
MASTER = "@@MASTER@@"
MODEL = "@@MODEL@@"
ROWPEER = "@@ROWPEER@@"          # readiness row written by this node
MASTER_PORT = 16389
VPN_NET = "10.77.0."
T0 = time.time()
ROW = "mn_" + TAG
HB = ROW + "_hb"
NOTES = []
PHASE = ["INIT"]
HBN = [0]


def d1(sql, params=()):
    req = urllib.request.Request(
        "https://api.cloudflare.com/client/v4/accounts/%s/d1/database/%s/query" % (CF_ACCT, CF_DB),
        data=json.dumps({"sql": sql, "params": list(params)}).encode(),
        method="POST",
        headers={"Authorization": "Bearer " + CF_KEY, "Content-Type": "application/json",
                 "User-Agent": "mnode"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def put(row, status, info):
    try:
        d1("INSERT INTO kaggle_gpu_state (id,status,tunnel_url,api_key,model,gpu_info,updated_at) "
           "VALUES (?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET status=excluded.status, "
           "gpu_info=excluded.gpu_info, updated_at=excluded.updated_at;",
           [row, str(status)[:80], "", "", MODEL, json.dumps(info)[:16000], time.time()])
    except Exception as e:
        print("D1FAIL", row, repr(e), flush=True)


def get_row(row, tries=80, delay=15):
    """Poll a row until it exists (returns gpu_info str) or give up.
    Prints every poll: Kaggle cancelled two GPU sessions during silent waits."""
    for _i in range(tries):
        if _i % 4 == 0:
            print("[wait %s] poll %d/%d" % (row, _i + 1, tries), flush=True)
        try:
            res = d1("SELECT gpu_info FROM kaggle_gpu_state WHERE id=?", [row])
            rr = res["result"][0]["results"]
            if rr:
                r = rr[0]
                v = r.get("gpu_info") if isinstance(r, dict) else r[1]
                if v:
                    return v
        except Exception as e:
            print("GETROW err", repr(e)[:80], flush=True)
        time.sleep(delay)
    return None


def log(msg):
    print("[%7.1fs] %s" % (time.time() - T0, msg), flush=True)


def save(status, detail="", extra=None):
    info = {"status": status, "detail": str(detail)[:16000], "notes": NOTES[-24:],
            "elapsed": round(time.time() - T0, 1), "pid": os.getpid(), "rank": RANK}
    if extra:
        info.update(extra)
    put(ROW, status, info)


def phase(name, detail=""):
    PHASE[0] = name
    NOTES.append({"phase": name, "t": round(time.time() - T0, 1), "d": str(detail)[:400]})
    log("PHASE %s" % name)
    save("RUNNING_" + name, str(detail))


def heartbeat():
    while True:
        time.sleep(30)
        HBN[0] += 1
        put(HB, "HB", {"phase": PHASE[0], "n": HBN[0], "t": round(time.time() - T0, 1),
                       "pid": os.getpid(), "rank": RANK})


threading.Thread(target=heartbeat, daemon=True).start()


def sh(cmd, timeout=3600):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or ""), time.time() - t0
    except subprocess.TimeoutExpired as e:
        return 124, ((e.stdout or "") + (e.stderr or ""))[-3000:], time.time() - t0
    except Exception as e:
        return 1, str(e), time.time() - t0


# ------------------------------------------------------------------ phases
phase("ENV")
rc, out, dt = sh("python3 -V; nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader; "
                 "free -g|head -2; df -h /kaggle/working /tmp /dev/shm 2>&1 | head -8; "
                 "echo -n 'tun:'; ls /dev/net/tun 2>&1; "
                 "echo -n 'sudo:'; sudo -n true 2>&1 && echo OK")
save("RUNNING_GPU_INFO", out)

rc, out, dt = sh("python3 - <<'PY'\n"
                 "import torch\n"
                 "print('torch', torch.__version__, 'cuda', torch.version.cuda)\n"
                 "print('cap', torch.cuda.get_device_capability(0), 'bf16', torch.cuda.is_bf16_supported(),\n"
                 "      'name', torch.cuda.get_device_name(0))\n"
                 "PY")
phase("CAPABILITY", out)

rc, out, dt = sh("pip show vllm 2>/dev/null|head -2")
if "Name: vllm" not in out:
    phase("INSTALL")
    rc, out, dt = sh("pip install --quiet vllm 2>&1|tail -4", timeout=1500)
    if rc != 0:
        save("FAIL_INSTALL", out[-4000:])
        sys.exit(1)

rc, out, dt = sh("python3 -c \"import vllm;print('vllm',vllm.__version__)\"")
m = re.search(r"vllm\s+([\d.]+)", out)
VER = m.group(1) if m else "?"
phase("VERSION", out)

rc, out, _ = sh("pip show flashinfer-jit-cache >/dev/null 2>&1 && echo PRESENT || echo ABSENT")
if "PRESENT" in out:
    sh("pip uninstall -y flashinfer-jit-cache 2>&1|tail -1", timeout=300)

phase("PATCH_SM75")
rc, out, _ = sh("python3 - <<'PY2'\n"
                "import re, pathlib\n"
                "p = pathlib.Path('/usr/local/lib/python3.13/dist-packages/vllm/platforms/cuda.py')\n"
                "s = p.read_text()\n"
                "m = re.search(r'def check_if_supports_dtype', s)\n"
                "assert m, 'method not found'\n"
                "start = m.start()\n"
                "m2 = re.search(r'\\n    def ', s[start+10:])\n"
                "end = start + 10 + m2.start() if m2 else len(s)\n"
                "head = s[start:end].splitlines()[0]\n"
                "new = head + '\\n        return  # bf16 CC check bypassed for sm75 Turing emulation\\n'\n"
                "p.write_text(s[:start] + new + s[end:])\n"
                "import py_compile\n"
                "py_compile.compile(str(p), doraise=True)\n"
                "print('PATCHED_OK', head)\n"
                "PY2", timeout=120)
if rc != 0 or "PATCHED_OK" not in out:
    save("FAIL_PATCH", out[-1500:])
    sys.exit(1)
save("RUNNING_PATCH_OK", out[:400])

# ------------------------------------------------------------------ VPN overlay (python tun<->tcp over bore)
phase("VPN")
print("VPN_PHASE_START sudo/tun check:", flush=True)
rc, out, _ = sh("mkdir -p /dev/net 2>/dev/null; sudo -n mknod /dev/net/tun c 10 200 2>/dev/null; "
                "sudo -n chmod 666 /dev/net/tun 2>/dev/null; "
                "ls -la /dev/net/tun 2>&1; sudo -n true 2>&1 && echo SUDO_OK || echo SUDO_FAIL; "
                "python3 -V; ip -V | head -1")
print("VPN_ENV_RC", rc, out[:600], flush=True)
put("mn_env", "ENV", {"rank": RANK, "out": out[:1500], "t": time.time()})
if "SUDO_FAIL" in out:
    save("FAIL_SUDO", out[-1200:])
    sys.exit(1)

OVERLAY_SRC = """
import fcntl, os, select, socket, struct, sys, threading, time

TUNNAME = "ovl0"
MYIP = None

def log(m):
    print("[ovl %s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)

def open_tun(name, ip, prefix="24"):
    import subprocess
    name = "ovl" + ip.split(".")[-1]   # unique per endpoint (hub=ovl1, spokeN=ovlN)
    # Kaggle containers lack the /dev/net/tun node (module is present): create it
    if not os.path.exists("/dev/net/tun"):
        subprocess.run(["mkdir", "-p", "/dev/net"], check=False)
        subprocess.run(["mknod", "/dev/net/tun", "c", "10", "200"], check=False)
        subprocess.run(["chmod", "666", "/dev/net/tun"], check=False)
    tun = os.open("/dev/net/tun", os.O_RDWR)
    ifr = struct.pack("16sH22s", name.encode(), 0x0001 | 0x1000, b"")  # IFF_TUN|IFF_NO_PI
    fcntl.ioctl(tun, 0x400454ca, ifr)  # TUNSETIFF
    subprocess.run(["ip", "addr", "add", "%s/%s" % (ip, prefix), "dev", name], check=False)
    subprocess.run(["ip", "link", "set", "dev", name, "up", "mtu", "1280"], check=False)
    subprocess.run(["ip", "route", "flush", "cache"], check=False)
    return tun

def ip_of(pkt):
    # returns dst ip str of an IPv4 packet (skip 4-byte PI if present)
    if len(pkt) >= 20 and (pkt[0] >> 4) == 4:
        off = 0
    elif len(pkt) >= 24 and (pkt[4] >> 4) == 4:
        off = 4
    else:
        return None
    return socket.inet_ntoa(pkt[off + 16:off + 20])

def frame_send(sock, pkt):
    try:
        sock.sendall(struct.pack("!I", len(pkt)) + pkt)
        return True
    except Exception:
        return False

def frame_recv_loop(sock, tun, tag):
    buf = b""
    while True:
        try:
            data = sock.recv(65536)
        except Exception:
            break
        if not data:
            break
        buf += data
        while len(buf) >= 4:
            (n,) = struct.unpack("!I", buf[:4])
            if len(buf) < 4 + n:
                break
            pkt = buf[4:4 + n]
            buf = buf[4 + n:]
            try:
                os.write(tun, pkt)
            except Exception:
                pass
    log("conn %s closed" % tag)
    try:
        sock.close()
    except Exception:
        pass

def tun_loop(tun, conns, lock, fallback=None):
    # kernel -> network: read tun, pick conn by dst ip (fallback=hub relay)
    while True:
        r, _, _ = select.select([tun], [], [], 30.0)
        if not r:
            continue
        try:
            pkt = os.read(tun, 65536)
        except Exception:
            continue
        dst = ip_of(pkt)
        if dst is None:
            continue
        sock = None
        if lock is not None:
            with lock:
                sock = conns.get(dst)
        if sock is None and fallback is not None:
            sock = fallback
        if sock is not None:
            frame_send(sock, pkt)

def hub(port):
    global MYIP
    MYIP = "10.77.0.1"
    tun = open_tun(TUNNAME, MYIP)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(16)
    conns = {}
    lock = threading.Lock()
    threading.Thread(target=tun_loop, args=(tun, conns, lock, None), daemon=True).start()
    log("hub listening 127.0.0.1:%d tun=%s" % (port, TUNNAME))
    while True:
        c, addr = srv.accept()
        c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # learn speaker ip from first frame (spoke announces via its src ip traffic)
        def handle(sock, addr):
            buf = b""
            while True:
                try:
                    data = sock.recv(65536)
                except Exception:
                    break
                if not data:
                    break
                buf += data
                while len(buf) >= 4:
                    (n,) = struct.unpack("!I", buf[:4])
                    if len(buf) < 4 + n:
                        break
                    pkt = buf[4:4 + n]
                    buf = buf[4 + n:]
                    src = None
                    if len(pkt) >= 20 and (pkt[0] >> 4) == 4:
                        src = socket.inet_ntoa(pkt[12:16])
                    elif len(pkt) >= 24 and (pkt[4] >> 4) == 4:
                        src = socket.inet_ntoa(pkt[4 + 12:4 + 16])
                    if src is not None:
                        with lock:
                            conns.setdefault(src, sock)
                    dst = ip_of(pkt)
                    if dst is None:
                        continue
                    if dst == MYIP:
                        try:
                            os.write(tun, pkt)
                        except Exception:
                            pass
                    else:
                        with lock:
                            fwd = conns.get(dst)
                        if fwd is not None and fwd is not sock:
                            frame_send(fwd, pkt)
            with lock:
                dead = [k for k, v in conns.items() if v is sock]
                for k in dead:
                    del conns[k]
            try:
                sock.close()
            except Exception:
                pass
            log("peer %s gone" % (addr,))
        threading.Thread(target=handle, args=(c, addr), daemon=True).start()

def spoke(host, port, ip):
    global MYIP
    MYIP = ip
    tun = open_tun(TUNNAME, ip)
    while True:
        try:
            s = socket.create_connection((host, int(port)), timeout=10)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.settimeout(None)   # blocking after connect: no 10s idle kill
            break
        except Exception as e:
            log("dial %s:%s failed: %s" % (host, port, e))
            time.sleep(5)
    log("connected to hub %s:%s as %s" % (host, port, ip))
    threading.Thread(target=frame_recv_loop, args=(s, tun, "hub"), daemon=True).start()
    while True:
        try:
            tun_loop(tun, {}, None, fallback=s)
        except Exception as e:
            log("tun_loop err: %r" % (e,))
        time.sleep(3)
        try:
            s.close()
        except Exception:
            pass
        while True:
            try:
                s = socket.create_connection((host, int(port)), timeout=10)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.settimeout(None)
                log("redialed hub")
                threading.Thread(target=frame_recv_loop, args=(s, tun, "hub2"), daemon=True).start()
                break
            except Exception as e:
                log("redial fail: %r" % (e,))
                time.sleep(5)

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "hub":
        hub(int(sys.argv[2]))
    else:
        spoke(sys.argv[2], int(sys.argv[3]), sys.argv[4])
"""

# write overlay module once
with open("/kaggle/working/ovl.py", "w") as _f:
    _f.write(OVERLAY_SRC)


def fetch(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "mnode"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

if RANK == 0:
    phase("BORE")
    try:
        import io
        import tarfile
        data = fetch("https://github.com/ekzhang/bore/releases/download/v0.6.0/"
                     "bore-v0.6.0-x86_64-unknown-linux-musl.tar.gz")
        tf = tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")
        member = [m for m in tf.getmembers() if m.name.endswith("/bore") or m.name == "bore"]
        tf.extractall("/tmp/borex", members=member, filter="data")
        bore_bin = None
        for root, _, files in os.walk("/tmp/borex"):
            for f in files:
                if f == "bore":
                    bore_bin = os.path.join(root, f)
        os.chmod(bore_bin, 0o755)
    except Exception as e:
        save("FAIL_BORE", repr(e))
        sys.exit(1)
    # start overlay hub first (binds 127.0.0.1:1194)
    rc, out, _ = sh("nohup sudo python3 /kaggle/working/ovl.py hub 1194 > /tmp/ovl.log 2>&1 & "
                    "sleep 3; cat /tmp/ovl.log; pgrep -f 'ovl.py hub' | head -2")
    print("HUB_START", rc, out[:600], flush=True)
    if "hub listening" not in out:
        time.sleep(3)
        out2 = sh("cat /tmp/ovl.log; pgrep -f 'ovl.py hub' | head -2")[1]
        if "hub listening" not in out2:
            save("FAIL_HUB_START", out + "\n" + out2[-1000:])
            sys.exit(1)
    rc, out, _ = sh("nohup %s local 1194 --to bore.pub > /tmp/bore.log 2>&1 & sleep 4; cat /tmp/bore.log" % bore_bin)
    print("BORE_START", rc, out[:400], flush=True)
    mm = re.search(r"([A-Za-z0-9.-]*bore\.pub):(\d+)", out)
    if not mm:
        time.sleep(4)
        out = sh("cat /tmp/bore.log")[1]
        mm = re.search(r"([A-Za-z0-9.-]*bore\.pub):(\d+)", out)
    if not mm:
        save("FAIL_BORE_EP", out[-1500:])
        sys.exit(1)
    EP_HOST, EP_PORT = mm.group(1), mm.group(2)
    put("mn_ep", "EP", {"host": EP_HOST, "port": EP_PORT, "t": time.time()})
    print("bore endpoint %s:%s" % (EP_HOST, EP_PORT), flush=True)
    phase("VPN_WAIT_PEER")
    peers = [p for p in "@@PEERS@@".split(",") if p]
    for prow in peers:
        peer = get_row(prow, tries=720, delay=15)   # ~112 min for capacity waves
        if not peer:
            save("FAIL_PEER_WAIT", "peer readiness row missing: " + prow)
            sys.exit(1)
    rc, out, _ = sh("for i in $(seq 2 9); do ping -c 1 -W 2 10.77.0.$i >/dev/null 2>&1 && echo PING_OK_$i; done")
    nready = out.count("PING_OK_")
    if nready < len(peers):
        save("FAIL_PEER_PING", "only %d/%d peers pinging\n%s" % (nready, len(peers), out[-1200:]))
        sys.exit(1)
    save("RUNNING_VPN_OK", "hub up, %d peers reachable" % nready)
else:
    ep_s = get_row("mn_ep", tries=720, delay=15)
    if not ep_s:
        save("FAIL_EP_WAIT", "hub never posted mn_ep")
        sys.exit(1)
    ep = json.loads(ep_s)
    my_ip = "10.77.0.%d" % (RANK + 1)
    rc, out, _ = sh("nohup sudo python3 /kaggle/working/ovl.py spoke %s %s %s > /tmp/ovl.log 2>&1 & "
                    "sleep 3; cat /tmp/ovl.log" % (ep["host"], ep["port"], my_ip))
    print("SPOKE_START", rc, out[:600], flush=True)
    # wait for tun + hub ping
    ok = False
    for i in range(40):
        rc2, out2, _ = sh("ip -4 addr show ovl0 2>/dev/null | head -3; "
                          "ping -c 1 -W 2 10.77.0.1 >/dev/null 2>&1 && echo PING_HUB_OK; "
                          "tail -3 /tmp/ovl.log")
        if "PING_HUB_OK" in out2:
            ok = True
            break
        print("[spoke tun wait %d/40]" % (i + 1), flush=True)
        time.sleep(6)
    if not ok:
        save("FAIL_HUB_PING", out2[-1500:])
        sys.exit(1)
    put(ROWPEER, "READY", {"rank": RANK, "t": time.time()})
    save("RUNNING_VPN_OK", "spoke %s up, hub reachable" % my_ip)

# ------------------------------------------------------------------ model download
phase("DOWNLOAD")
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HOME"] = "/kaggle/working/hf_cache"
try:
    from huggingface_hub import list_repo_files, hf_hub_download
    files = list_repo_files(MODEL, token=HF_TOKEN)
    sfiles = [f for f in files if f.endswith((".safetensors", ".json", ".txt", ".model"))]
    if "@@DL_MODE@@" == "smallonly":
        # 27B: only metadata now; vLLM's loader pulls just this stage's
        # safetensors shards on demand (xet off) => per-node disk ~stage size
        sfiles = [f for f in sfiles if not f.endswith(".safetensors")]
    NOTES.append({"files": sfiles[:10], "n": len(sfiles)})
    got = []
    for f in sfiles:
        p0 = time.time()
        hf_hub_download(MODEL, f, token=HF_TOKEN, cache_dir="/kaggle/working/hf_cache")
        got.append(f)
        put(ROW, "RUNNING_DL_FILE", {"file": f, "got": len(got), "total": len(sfiles),
                                     "t": round(time.time() - T0, 1), "rank": RANK})
        log("dl %s %.1fs" % (f, time.time() - p0))
    phase("DOWNLOAD_OK", "%d files" % len(got))
except Exception as e:
    import traceback
    save("FAIL_DOWNLOAD", str(e) + "\n" + traceback.format_exc()[-1500:])
    sys.exit(1)

phase("PRE_SERVE")
rc, out, _ = sh("python3 -c \"import vllm.entrypoints.llm; print('vllm-import-ok')\"", timeout=300)
if rc != 0:
    sh("pip uninstall -y torchaudio 2>&1|tail -1", timeout=180)
    rc, out, _ = sh("python3 -c \"import vllm.entrypoints.llm; print('vllm-import-ok-after-uninstall')\"", timeout=300)
    if rc != 0:
        sh("pip install --quiet torchaudio 2>&1|tail -2", timeout=600)
        rc, out, _ = sh("python3 -c \"import vllm.entrypoints.llm; print('vllm-import-ok-after-reinstall')\"", timeout=300)
    if rc != 0:
        save("FAIL_IMPORT", out[-4000:])
        sys.exit(1)
save("RUNNING_PRE_SERVE_OK", out[-500:])

phase("PREFLIGHT")
sh_df = sh("df -h /dev/shm /tmp /kaggle/working; mount | grep -E '/dev/shm' | head -2")[1]
sh_rem = sh("mount -o remount,size=4G /dev/shm 2>&1 && echo REMOUNT_OK || echo REMOUNT_FAIL")[1]
save("RUNNING_PREFLIGHT", "df:" + sh_df[:400] + " remount:" + sh_rem[:120])

# serve gate: hub announces readiness so the spoke never dials a dead TCPStore
if RANK == 0:
    put("mn_gate", "GATE", {"t": time.time()})
else:
    phase("GATE_WAIT")
    gate = get_row("mn_gate", tries=120, delay=15)   # ~30 min
    if not gate:
        save("FAIL_GATE_WAIT", "hub never posted mn_gate")
        sys.exit(1)
    save("RUNNING_GATE_OK", "hub gate seen")

# ------------------------------------------------------------------ serve
phase("SERVE")
SERVE_SRC = '''
import os, time, json, sys

RANK = %d
NNODES = %d
MASTER = "%s"
MASTER_PORT = %d
MODEL = "%s"

def main():
    from vllm import LLM, SamplingParams
    common = dict(
        model=MODEL, dtype="bfloat16",
        tensor_parallel_size=2, pipeline_parallel_size=%d,
        nnodes=NNODES, node_rank=RANK,
        master_addr=MASTER, master_port=MASTER_PORT,
        distributed_executor_backend="mp",
        gpu_memory_utilization=0.92, max_model_len=4096,
        trust_remote_code=True, download_dir="/kaggle/working/hf_cache",
        skip_mm_profiling=True, enforce_eager=True,
    )
    t0 = time.time()
    llm = LLM(**common)
    t_load = time.time() - t0
    if RANK == 0:
        sp = SamplingParams(max_tokens=64, temperature=0.7)
        t1 = time.time()
        outs = llm.generate(["Write a one-line greeting from a friendly robot."], sp)
        t_gen = time.time() - t1
        ntok = len(outs[0].outputs[0].token_ids)
        print("VLLM_RESULT " + json.dumps({"t_load_s": round(t_load, 1), "t_gen_s": round(t_gen, 2),
              "tokens": ntok, "tok_per_s": round(ntok / t_gen, 2),
              "text": outs[0].outputs[0].text[:200]}), flush=True)
    else:
        print("WORKER_HOLD engine up, holding", flush=True)
        time.sleep(@@HOLDSECS@@)

if __name__ == "__main__":
    main()
''' % (RANK, NNODES, MASTER, MASTER_PORT, MODEL, int("@@PP@@"))

with open("/kaggle/working/serve_mn.py", "w") as fh:
    fh.write(SERVE_SRC)
if os.environ.get("MN_ROLE") == "compile_check":
    import py_compile
    py_compile.compile("/kaggle/working/serve_mn.py", doraise=True)
    print("SERVE_COMPILE_OK")
    sys.exit(0)

# nccl via overlay
rc, out, _ = sh("ip route get 10.77.0.1 2>/dev/null | head -1")
DEV = ""
mdev = re.search(r"dev\s+(\S+)", out)
if mdev:
    DEV = mdev.group(1)
log("overlay dev: " + DEV)
save("RUNNING_SERVE_PRE", "dev=%s" % DEV)

envp = ("HF_TOKEN=%s HF_HUB_DISABLE_XET=1 HF_HOME=/kaggle/working/hf_cache "
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        "VLLM_LOGGING_LEVEL=INFO NCCL_SOCKET_IFNAME=%s NCCL_IB_DISABLE=1 NCCL_PROTO=Simple "
        "NCCL_DEBUG=WARN NCCL_P2P_DISABLE=0 NCCL_SHM_DISABLE=0 NCCL_SOCKET_NTHREADS=4 "
        "TORCH_NCCL_WAIT_TIMEOUT_MINUTES=30 " % (HF_TOKEN, DEV))
rc, out, dt = sh(envp + "python3 /kaggle/working/serve_mn.py "
                 "> /kaggle/working/serve_mn.log 2>&1; echo RC=$?", timeout=3000)
log_tail = sh("tail -c 10000 /kaggle/working/serve_mn.log")[1]
log_head = sh("head -c 6000 /kaggle/working/serve_mn.log")[1]
errlines = sh("grep -n -m 30 -Ei 'error|not supported|failed|bootstrap|timed out|nccl|connection|refused' "
              "/kaggle/working/serve_mn.log")[1]
NOTES.append({"serve_rc": out.strip()[-40:]})
mm = re.search(r"VLLM_RESULT (\{.*\})", log_head + log_tail)
if (RANK == 0) and mm and out.strip().endswith("RC=0"):
    save("DONE", log_tail[-4000:], extra={"result": json.loads(mm.group(1)), "vllm": VER})
    put("mn_result", "DONE", {"result": json.loads(mm.group(1)), "vllm": VER,
                              "tag": TAG, "elapsed": round(time.time() - T0, 1)})
    print("SUCCESS", mm.group(1), flush=True)
    sys.exit(0)
if (RANK != 0) and out.strip().endswith("RC=0"):
    save("DONE_WORKER", "worker held and exited cleanly", extra={"vllm": VER})
    print("WORKER_DONE", flush=True)
    sys.exit(0)

# fail: chunks + classified tag
try:
    blob = open("/kaggle/working/serve_mn.log", "rb").read()
    txt = blob.decode("utf-8", "replace")
    n = len(txt)
    for i in range(3):
        part = txt[i * n // 3:(i + 1) * n // 3]
        put(ROW + "_l%d" % (i + 1), "LOG%d" % (i + 1), {"off": i * n // 3, "len": len(part),
                                                        "part": part[:15000], "rank": RANK})
except Exception as e:
    save("LOGCAP_FAIL", str(e))

low = (log_head + log_tail + errlines + out).lower()
tag = "FAIL_SERVE"
if "out of memory" in low:
    tag = "FAIL_OOM"
elif "skipframe" in low or "bf16 is not supported" in low:
    tag = "FAIL_COMPILE"
elif "connection refused" in low or "timed out" in low or "bootstrap" in low:
    tag = "FAIL_DIST"
elif "no route" in low or "network is unreachable" in low:
    tag = "FAIL_NET"
save(tag, "ERRLINES:\n" + errlines[:3000] + "\n\nHEAD:\n" + log_head[:3500] +
     "\n\nTAIL:\n" + log_tail[-4500:], extra={"vllm": VER})
sys.exit(1)
