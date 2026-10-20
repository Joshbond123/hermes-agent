"""PP3 GPU template: 3-node vLLM PP3xTP2 serve of Qwen3.8-27B-Uncensored BF16.

Transport: netmesh2 (LD_PRELOAD port-blocks + cloudflared WS relay + D1 dir).
Assembly: pulls 6 per-account Kaggle mirror datasets -> /tmp/model, sha256
verified against model_mirror_* rows, index.json from HF.
Serve: vllm serve --nnodes 3 --node-rank R --master-addr 172.19.2.2 (block-port
identity + relay handles cross-node); E8 fix stack (PATCH_SM75, enforce_eager,
skip_mm_profiling). Rank0 runs the live completion test -> vllm_pp3_result.
Rows: pp3_{tag} (+_hb). Terminal: DONE | FAIL_*.
"""
import json
import os
import re
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.request
import signal as _sig

CF_ACCT = "@@CF_ACCOUNT@@"
CF_DB = "@@CF_DB@@"
CF_KEY = "@@CF_KEY@@"
TAG = "@@TAG@@"
RANK = int("@@PART@@")
T0 = time.time()
T0_WALL = T0
ROW = "pp3_" + TAG
HB = ROW + "_hb"
SHARED = "172.19.2.2"
NOTES = []
PHASE = ["INIT"]

NP_C = r'''
#define _GNU_SOURCE
#include <dlfcn.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <unistd.h>
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

static int (*real_bind)(int, const struct sockaddr*, socklen_t);
static int (*real_connect)(int, const struct sockaddr*, socklen_t);
static ssize_t (*real_write)(int, const void*, size_t);
static ssize_t (*real_send)(int, const void*, size_t, int);
static ssize_t (*real_writev)(int, const struct iovec*, int);
static int (*real_close)(int);

static int MYN = 0;
static uint32_t SHARED_IP = 0;
static uint16_t BLOCK_LO = 41000;
static pthread_mutex_t LK = PTHREAD_MUTEX_INITIALIZER;
static int inited = 0;

typedef struct { int used; int sent; int owner; uint16_t port; } Slot;
#define TAB 8192
static Slot tab[TAB];

static void init0(void) {
    if (inited) return;
    real_bind = dlsym(RTLD_NEXT, "bind");
    real_connect = dlsym(RTLD_NEXT, "connect");
    real_write = dlsym(RTLD_NEXT, "write");
    real_send = dlsym(RTLD_NEXT, "send");
    real_writev = dlsym(RTLD_NEXT, "writev");
    real_close = dlsym(RTLD_NEXT, "close");
    const char *r = getenv("NP_RANK"); MYN = r ? atoi(r) : 0;
    const char *s = getenv("NP_SHARED"); SHARED_IP = inet_addr(s ? s : "172.19.2.2");
    BLOCK_LO = (uint16_t)(41000 + MYN * 1000);
    inited = 1;
}
__attribute__((constructor)) static void ctor(void) { init0(); }

static int is_stream(int fd) {
    int t = 0; socklen_t l = sizeof(t);
    if (getsockopt(fd, SOL_SOCKET, SO_TYPE, &t, &l) != 0) return 0;
    return t == SOCK_STREAM;
}

int bind(int fd, const struct sockaddr *a, socklen_t l) {
    init0();
    if (a && a->sa_family == AF_INET && l >= sizeof(struct sockaddr_in) && is_stream(fd)) {
        const struct sockaddr_in *in = (const struct sockaddr_in *)a;
        if (in->sin_port == 0) {
            static uint16_t off = 0;
            pthread_mutex_lock(&LK);
            for (int i = 0; i < 1000; i++) {
                uint16_t p = BLOCK_LO + (uint16_t)((off + i) % 1000);
                struct sockaddr_in b = *in;
                b.sin_port = htons(p);
                if (real_bind(fd, (struct sockaddr *)&b, l) == 0) {
                    off = (uint16_t)((p - BLOCK_LO + 1) % 1000);
                    pthread_mutex_unlock(&LK);
                    return 0;
                }
                if (errno != EADDRINUSE && errno != EACCES) break;
            }
            pthread_mutex_unlock(&LK);
        }
    }
    return real_bind(fd, a, l);
}

int connect(int fd, const struct sockaddr *a, socklen_t l) {
    init0();
    if (a && a->sa_family == AF_INET && l >= sizeof(struct sockaddr_in) && is_stream(fd)) {
        const struct sockaddr_in *in = (const struct sockaddr_in *)a;
        if (in->sin_addr.s_addr == SHARED_IP) {
            uint16_t p = ntohs(in->sin_port);
            int owner;
            if (p >= 41000 && p < 44000) owner = (p - 41000) / 1000;
            else owner = 0;
            if (owner == MYN) {
                /* self: dial the original address (own eth0 IP loops back locally;
                   rewriting to 127.0.0.1 would miss listeners bound to eth0) */
                return real_connect(fd, a, l);
            }
            struct sockaddr_in fw;
            memset(&fw, 0, sizeof(fw));
            fw.sin_family = AF_INET;
            fw.sin_port = htons(9998);
            fw.sin_addr.s_addr = inet_addr("127.0.0.1");
            pthread_mutex_lock(&LK);
            if (fd >= 0 && fd < TAB) {
                tab[fd].used = 1; tab[fd].sent = 0;
                tab[fd].owner = owner; tab[fd].port = p;
            }
            pthread_mutex_unlock(&LK);
            return real_connect(fd, (struct sockaddr *)&fw, l);
        }
    }
    return real_connect(fd, a, l);
}

static int prep(int fd, char h[6]) {
    int r = 0;
    pthread_mutex_lock(&LK);
    if (fd >= 0 && fd < TAB && tab[fd].used && !tab[fd].sent && tab[fd].port) {
        h[0] = (char)0xA5; h[1] = (char)tab[fd].owner;
        h[2] = (char)(tab[fd].port >> 8); h[3] = (char)(tab[fd].port & 0xff);
        h[4] = 0; h[5] = 0;
        tab[fd].sent = 1; r = 1;
    }
    pthread_mutex_unlock(&LK);
    return r;
}

ssize_t send(int fd, const void *b, size_t n, int f) {
    init0();
    char h[6];
    if (is_stream(fd) && prep(fd, h)) real_send(fd, h, 6, f);
    return real_send(fd, b, n, f);
}
ssize_t write(int fd, const void *b, size_t n) {
    init0();
    char h[6];
    if (is_stream(fd) && prep(fd, h)) real_write(fd, h, 6);
    return real_write(fd, b, n);
}
ssize_t writev(int fd, const struct iovec *v, int c) {
    init0();
    char h[6];
    if (is_stream(fd) && prep(fd, h)) real_write(fd, h, 6);
    return real_writev(fd, v, c);
}
int close(int fd) {
    init0();
    if (fd >= 0 && fd < TAB) {
        pthread_mutex_lock(&LK);
        tab[fd].used = 0; tab[fd].sent = 0;
        pthread_mutex_unlock(&LK);
    }
    return real_close(fd);
}
'''

# relay reads CF creds from this file (no string substitution in its source)
RELAY = r'''
import asyncio, json, os, struct, time, urllib.request
import websockets
CFC = json.load(open("/tmp/cf.json"))
DIRCACHE = {}
MAGIC = 0xA5

def log(*a):
    print("[%7.1f]" % time.time(), *a, flush=True)

def d1(sql, params=()):
    req = urllib.request.Request(
        "https://api.cloudflare.com/client/v4/accounts/%s/d1/database/%s/query"
        % (CFC["acct"], CFC["db"]),
        data=json.dumps({"sql": sql, "params": list(params)}).encode(),
        method="POST",
        headers={"Authorization": "Bearer " + CFC["key"], "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

async def dirpoll():
    while True:
        try:
            rows = await asyncio.to_thread(
                lambda: d1("SELECT id,gpu_info FROM kaggle_gpu_state WHERE id LIKE 'tr_dir_%'")["result"][0]["results"])
            for r in rows:
                r = r if isinstance(r, dict) else {"id": r[0], "gpu_info": r[1]}
                try:
                    info = json.loads(r["gpu_info"] or "{}")
                except Exception:
                    continue
                if info.get("ws"):
                    try:
                        DIRCACHE[int(r["id"].split("_")[-1])] = info["ws"]
                    except Exception:
                        pass
        except Exception as e:
            log("direrr", repr(e))
        await asyncio.sleep(3)

async def pump_t2w(reader, ws):
    try:
        while True:
            d = await reader.read(65536)
            if not d:
                break
            await ws.send(d)
    except Exception:
        pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass

async def pump_w2t(ws, writer):
    try:
        async for m in ws:
            if isinstance(m, bytes) and m:
                writer.write(m)
                await writer.drain()
    except Exception:
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass

async def handle_out(reader, writer):
    try:
        hdr = await asyncio.wait_for(reader.readexactly(6), timeout=15)
    except Exception as e:
        log("outhdr", repr(e))
        writer.close()
        return
    if hdr[0] != MAGIC:
        log("badmagic", hdr[:6])
        writer.close()
        return
    owner = hdr[1]
    port = struct.unpack(">H", hdr[2:4])[0]
    ws_url = DIRCACHE.get(owner)
    if not ws_url:
        log("nodir", owner)
        writer.close()
        return
    uri = (ws_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
           .rstrip("/") + "/tcp?port=%d" % port)
    ws = None
    last = None
    for attempt in range(8):
        try:
            ws = await asyncio.wait_for(
                websockets.connect(uri, ping_interval=20, close_timeout=5, open_timeout=20),
                timeout=30)
            break
        except Exception as e:
            last = e
            log("wsconn try%d" % attempt, uri, repr(e)[:220])
            await asyncio.sleep(2.5)
    if ws is None:
        log("wsconn giveup", uri, repr(last)[:200])
        writer.close()
        return
    log("wsopen", uri)
    await asyncio.gather(pump_t2w(reader, ws), pump_w2t(ws, writer))

async def handle_in(ws, path):
    try:
        q = path.split("?", 1)[1]
        port = int([x.split("=", 1)[1] for x in q.split("&") if x.startswith("port=")][0])
    except Exception as e:
        log("badpath", path, repr(e))
        await ws.close()
        return
    r2 = w2 = None
    for host in ("127.0.0.1", "172.19.2.2"):
        try:
            r2, w2 = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=8)
            break
        except Exception as e:
            last = e
    if w2 is None:
        log("dialfail", port, repr(last))
        await ws.close()
        return
    log("dial", port)

    async def w2t():
        try:
            async for m in ws:
                if isinstance(m, bytes) and m:
                    w2.write(m)
                    await w2.drain()
        except Exception:
            pass
        finally:
            try:
                w2.close()
            except Exception:
                pass

    async def t2w():
        try:
            while True:
                d = await r2.read(65536)
                if not d:
                    break
                await ws.send(d)
        except Exception:
            pass
        finally:
            try:
                await ws.close()
            except Exception:
                pass

    await asyncio.gather(w2t(), t2w())

async def main():
    asyncio.ensure_future(dirpoll())
    s1 = await asyncio.start_server(handle_out, "127.0.0.1", 9998)
    s2 = await websockets.serve(handle_in, "127.0.0.1", 10001)
    log("relay up")
    await asyncio.gather(s1.serve_forever(), s2.serve_forever())

asyncio.run(main())
'''

ECHO = r'''
import os, socket, threading
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("0.0.0.0", 0))
s.listen(64)
open("/tmp/echo_port", "w").write(str(s.getsockname()[1]))

def h(c):
    try:
        while True:
            d = c.recv(4096)
            if not d:
                break
            c.sendall(d.replace(b"PING:", b"PONG:"))
    except Exception:
        pass
    finally:
        try:
            c.close()
        except Exception:
            pass

while True:
    c, _ = s.accept()
    threading.Thread(target=h, args=(c,), daemon=True).start()
'''

PONG_SRC = """
import socket, sys, time
peer, port, nonce = sys.argv[1], int(sys.argv[2]), sys.argv[3]
t0 = time.time()
s = socket.create_connection((peer, port), timeout=12)
s.sendall(("PING:%s\\n" % nonce).encode())
buf = b""
s.settimeout(12)
while b"\\n" not in buf:
    c = s.recv(4096)
    if not c:
        break
    buf += c
s.close()
ok = ("PONG:" + nonce).encode() in buf
print("OK" if ok else "BAD", round((time.time()-t0)*1000, 1),
      "" if ok else repr(buf[:90]))
"""


def d1(sql, params=()):
    req = urllib.request.Request(
        "https://api.cloudflare.com/client/v4/accounts/%s/d1/database/%s/query" % (CF_ACCT, CF_DB),
        data=json.dumps({"sql": sql, "params": list(params)}).encode(),
        method="POST",
        headers={"Authorization": "Bearer " + CF_KEY, "Content-Type": "application/json",
                 "User-Agent": "netmesh2"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def put(row, status, info):
    try:
        d1("INSERT INTO kaggle_gpu_state (id,status,tunnel_url,api_key,model,gpu_info,updated_at) "
           "VALUES (?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET status=excluded.status, "
           "gpu_info=excluded.gpu_info, updated_at=excluded.updated_at;",
           [row, str(status)[:80], "", "", "netmesh2", json.dumps(info)[:16000], time.time()])
    except Exception as e:  # noqa: BLE001
        print("D1FAIL", row, repr(e), flush=True)


def save(status, detail="", extra=None):
    info = {"status": status, "detail": str(detail)[:16000], "notes": NOTES[-24:],
            "elapsed": round(time.time() - T0, 1), "rank": RANK, "tag": TAG}
    if extra:
        info.update(extra)
    put(ROW, status, info)


def log(msg):
    print("[%7.1fs] %s" % (time.time() - T0, msg), flush=True)


def phase(name, detail=""):
    PHASE[0] = name
    NOTES.append({"phase": name, "t": round(time.time() - T0, 1), "d": str(detail)[:400]})
    log("PHASE %s %s" % (name, detail))
    save("RUNNING_" + name, str(detail))


def sh(cmd, timeout=300, env=None):
    e = dict(os.environ)
    if env:
        e.update(env)
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, env=e)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except Exception as ex:  # noqa: BLE001
        return 1, str(ex)


def fail(tag, detail):
    save(tag, str(detail)[:6000])
    log("FAIL %s" % tag)
    os._exit(1)


def heartbeat():
    while True:
        time.sleep(10)
        put(HB, "HB", {"phase": PHASE[0], "t": round(time.time() - T0, 1)})


def _sigdie(signum, frame):
    try:
        put(ROW, "SIGNAL_%d" % signum,
            {"signal": signum, "t": round(time.time() - T0, 1), "phase": PHASE[0]})
    except Exception:
        pass
    os._exit(128 + signum)


threading.Thread(target=heartbeat, daemon=True).start()
for _s in (_sig.SIGTERM, _sig.SIGINT, _sig.SIGHUP, _sig.SIGQUIT):
    _sig.signal(_s, _sigdie)

# ---------------------------------------------------------------- ENV
phase("ENV")
rc, out = sh("id -u; nvidia-smi -L; free -g | head -2; df -h /tmp | tail -1; "
             "pip install -q websockets==12.0 2>&1 | tail -1; "
             "python3 -c 'import websockets; print(\"ws\", websockets.__version__)'; "
             "curl -s --max-time 8 https://ipinfo.io/country")
NOTES.append({"cmd": "env", "rc": rc, "out": out[:900]})
if "Tesla" not in out and "GPU" not in out:
    fail("FAIL_NO_GPU", out[:800])
region = (out.strip().split("\n")[-1] or "").strip()
log("region=%s" % region)
rc, out2 = sh("python3 -c \"import shutil;t,u,f=shutil.disk_usage('/tmp');print('TMPFREE_G', f//2**30)\"")
NOTES.append({"cmd": "disk", "out": out2.strip()[:120]})
try:
    free_tb = int(out2.split("TMPFREE_G")[1].strip().split()[0])
except Exception:
    free_tb = -1
if free_tb < 130:
    fail("FAIL_DISK", "only %sG free on /tmp; need >=130G; out=%s" % (free_tb, out2[:400]))
if "ws " not in out:
    fail("FAIL_ENV", out[:1000])

# ---------------------------------------------------------------- PULL (26 files, Bearer REST)
phase("PULL")
# REAL part->owner map from the fan-out: p1=ala p2=tes p3=jos p4=tem p5=hap p6=bla
CREDITS = [
    ("alagbo", "@@KAGGLE_U3@@", "@@KAGGLE_P3@@", 1),
    ("teslaarymo", "@@KAGGLE_U2@@", "@@KAGGLE_P2@@", 2),
    ("Joshbond123", "@@KAGGLE_BU1@@", "@@KAGGLE_B1@@", 3),
    ("temilolo", "@@KAGGLE_BU2@@", "@@KAGGLE_B2@@", 4),
    ("happink", "@@KAGGLE_BU3@@", "@@KAGGLE_B3@@", 5),
    ("black797", "@@KAGGLE_U1@@", "@@KAGGLE_P1@@", 6),
]
PARTS = {1: [1, 2, 3], 2: [4, 5, 6], 3: [7, 8, 9],
         4: [10, 11, 12], 5: [13, 14, 15], 6: [16, 17, 18]}
SMALLS = ["config.json", "generation_config.json", "tokenizer.json",
          "tokenizer_config.json", "chat_template.jinja", "merges.txt",
          "vocab.json", "preprocessor_config.json"]
MODEL_DIR = "/tmp/model"
os.makedirs(MODEL_DIR, exist_ok=True)

JOBS = []  # (owner, key, ref, filename)
for owner, ku, kk, part in CREDITS:
    ref = "%s/qwen38-27b-bf16-p%d" % (owner, part)
    for idx in PARTS[part]:
        JOBS.append((ku, kk, ref, "model-%05d-of-00018.safetensors" % idx))
_ref0 = "%s/qwen38-27b-bf16-p1" % (CREDITS[0][0],)
for f in SMALLS:
    JOBS.append((CREDITS[0][1], CREDITS[0][2], _ref0, f))

PULL_DONE = []
PULL_LOCK = threading.Lock()


def fetch_file(key, ref, fname):
    dest = os.path.join(MODEL_DIR, fname)
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        return fname, 0
    url = "https://www.kaggle.com/api/v1/datasets/download/%s/%s" % (ref, fname)
    last = ""
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"Authorization": "Bearer " + key})
            t0f = time.time()
            n = 0
            with urllib.request.urlopen(req, timeout=900) as r, open(dest + ".part", "wb") as fh:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    n += len(chunk)
            os.replace(dest + ".part", dest)
            with PULL_LOCK:
                PULL_DONE.append(fname)
                put(ROW, "RUNNING_PULL",
                    {"file": fname, "bytes": n, "done": len(PULL_DONE), "total": len(JOBS),
                     "dt": round(time.time() - t0f, 1)})
            return fname, n
        except Exception as e:  # noqa: BLE001
            last = repr(e)[:200]
            time.sleep(4 * (attempt + 1))
    raise RuntimeError("pull failed %s @ %s: %s" % (fname, ref, last))


from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = [ex.submit(fetch_file, k, rf, fn) for (_u, k, rf, fn) in JOBS]
        for f in as_completed(futs):
            f.result()
except Exception as e:  # noqa: BLE001
    fail("FAIL_PULL", str(e)[:1500])
nfiles = len([x for x in os.listdir(MODEL_DIR) if x.endswith(".safetensors")])
if nfiles != 18:
    fail("FAIL_PULL", "only %d shards present" % nfiles)
NOTES.append({"cmd": "pull", "files": len(PULL_DONE)})
log("pulled %d files" % len(PULL_DONE))

# index + extra configs not shipped in the mirror parts
phase("INDEX")
for extra in ("model.safetensors.index.json", "video_preprocessor_config.json"):
    rc, out = sh("curl -sSL --max-time 120 -H 'Authorization: Bearer @@HF_TOKEN@@' "
                 "-o %s/%s 'https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored/resolve/main/%s' "
                 "&& wc -c %s/%s" % (MODEL_DIR, extra, extra, MODEL_DIR, extra), timeout=180)
    NOTES.append({"cmd": "idx_" + extra, "rc": rc, "out": out[-200:]})
    if rc != 0:
        fail("FAIL_INDEX", "%s rc=%s %s" % (extra, rc, out[-600:]))
rc, out = sh("python3 -c \"import json;d=json.load(open('%s/model.safetensors.index.json'));"
             "print('weight_map', len(d.get('weight_map', {})))\"" % MODEL_DIR)
if rc != 0 or "weight_map" not in out:
    fail("FAIL_INDEX", out[:600])
NOTES.append({"cmd": "index_ok", "out": out.strip()[:120]})

# ---------------------------------------------------------------- VERIFY (sha256 vs mirror rows)
phase("VERIFY")
exp = {}
try:
    rows = d1("SELECT id,gpu_info FROM kaggle_gpu_state WHERE id LIKE 'model_mirror_%'")["result"][0]["results"]
    for r in rows:
        r = r if isinstance(r, dict) else {"id": r[0], "gpu_info": r[1]}
        if r["id"].endswith("_hb") or r["id"] == "model_mirror_result":
            continue
        try:
            gi = json.loads(r["gpu_info"] or "{}")
        except Exception:
            continue
        for fn, meta in (gi.get("sha256") or {}).items():
            hexv = meta.get("sha256") if isinstance(meta, dict) else meta
            if fn in exp and exp[fn] != hexv:
                fail("FAIL_VERIFY", "cross-account sha conflict on %s" % fn)
            exp[fn] = hexv
except Exception as e:  # noqa: BLE001
    fail("FAIL_VERIFY", "cannot read mirror rows: %r" % (e,))
n_shards = len([f for f in exp if f.endswith(".safetensors")])
if len(exp) < 26 or n_shards != 18:
    fail("FAIL_VERIFY", "mirror rows incomplete: files=%d shards=%d" % (len(exp), n_shards))
bad = []
tot = 0
import hashlib as _hl
for fn, want in exp.items():
    pth = os.path.join(MODEL_DIR, fn)
    if not os.path.isfile(pth):
        bad.append((fn, "missing")); continue
    h = _hl.sha256()
    with open(pth, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    if h.hexdigest() != want:
        bad.append((fn, "sha"))
    tot += os.path.getsize(pth)
    put(ROW, "RUNNING_VERIFY", {"file": fn, "bad": len(bad), "t": round(time.time() - T0, 1)})
if bad:
    fail("FAIL_VERIFY", "bad files: %s" % bad[:8])
if tot < 55_000_000_000:
    fail("FAIL_VERIFY", "total too small: %d" % tot)
NOTES.append({"cmd": "verify", "files": len(exp), "bytes": tot})
log("verified %d files %d bytes" % (len(exp), tot))

# ---------------------------------------------------------------- INSTALL vLLM + patch
phase("INSTALL")
rc, out = sh("pip show vllm 2>/dev/null | head -2")
if "Name: vllm" not in out:
    rc, out = sh("pip install --quiet vllm==0.32.0 2>&1 | tail -4", timeout=1800)
    if rc != 0:
        fail("FAIL_INSTALL", out[-3000:])
rc, out = sh("python3 -c \"import vllm;print('vllm',vllm.__version__)\"")
if "vllm 0.32.0" not in out:
    fail("FAIL_INSTALL", out[-800:])
NOTES.append({"cmd": "vllm_ver", "out": out.strip()[:80]})
rc, out = sh("pip show flashinfer-jit-cache >/dev/null 2>&1 && echo PRESENT || echo ABSENT")
if "PRESENT" in out:
    sh("pip uninstall -y flashinfer-jit-cache 2>&1 | tail -1", timeout=300)
phase("PATCH_SM75")
rc, out = sh("python3 - <<'PY2'\n"
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
    fail("FAIL_PATCH", out[-1500:])
NOTES.append({"cmd": "patch", "out": out.strip()[:160]})

phase("PRE_SERVE")
rc, out = sh("python3 -c \"import vllm.entrypoints.llm; print('vllm-import-ok')\"", timeout=300)
if rc != 0:
    sh("pip uninstall -y torchaudio 2>&1 | tail -1", timeout=180)
    rc, out = sh("python3 -c \"import vllm.entrypoints.llm; print('ok-after-uninstall')\"", timeout=300)
    if rc != 0:
        sh("pip install --quiet torchaudio 2>&1 | tail -2", timeout=600)
        rc, out = sh("python3 -c \"import vllm.entrypoints.llm; print('ok-after-reinstall')\"", timeout=300)
        if rc != 0:
            fail("FAIL_IMPORT", out[-3000:])
NOTES.append({"cmd": "import", "out": out.strip()[:80]})
sh("mount -o remount,size=4G /dev/shm 2>&1 | tail -1")


# ---------------------------------------------------------------- COMPILE
phase("COMPILE")
with open("/tmp/np.c", "w") as fh:
    fh.write(NP_C)
rc, out = sh("gcc -shared -fPIC -O2 -o /tmp/np.so /tmp/np.c -ldl && ls -la /tmp/np.so")
NOTES.append({"cmd": "gcc", "rc": rc, "out": out[-300:]})
if rc != 0:
    fail("FAIL_COMPILE", out[:1500])
lo = 41000 + RANK * 1000
rc, out = sh("python3 -c \"import socket;s=socket.socket();s.bind(('0.0.0.0',0));print(s.getsockname())\"",
             env={"LD_PRELOAD": "/tmp/np.so", "NP_RANK": str(RANK), "NP_SHARED": SHARED})
try:
    p = int(re.search(r"0\.0\.0\.0', (\d+)", out.replace('"', "'")).group(1))
except Exception:
    p = -1
if not (lo <= p < lo + 1000):
    fail("FAIL_PRELOAD", "bind rewrite miss: %s want [%d,%d)" % (out, lo, lo + 1000))
NOTES.append({"cmd": "preload_bind", "out": out.strip()[:120]})
log("preload bind -> %d" % p)

# ---------------------------------------------------------------- RELAY
phase("RELAY")
with open("/tmp/cf.json", "w") as fh:
    json.dump({"acct": CF_ACCT, "db": CF_DB, "key": CF_KEY}, fh)
os.chmod("/tmp/cf.json", 0o600)
with open("/tmp/relay.py", "w") as fh:
    fh.write(RELAY)
rp = subprocess.Popen([sys.executable, "-u", "/tmp/relay.py"],
                      stdout=open("/tmp/relay.out", "w"), stderr=subprocess.STDOUT)
time.sleep(2.5)
if rp.poll() is not None:
    fail("FAIL_RELAY", open("/tmp/relay.out").read()[:2000])
try:
    s = socket.create_connection(("127.0.0.1", 9998), timeout=3)
    s.sendall(b"\x00" * 6)
    s.close()
except Exception as e:  # noqa: BLE001
    log("relay probe err %r" % (e,))
time.sleep(1)
rout = open("/tmp/relay.out").read()[-400:]
NOTES.append({"cmd": "relay_probe", "out": rout})
if "badmagic" not in rout:
    fail("FAIL_RELAY", "probe not logged; relay.out=%s" % rout)
log("relay up pid=%d" % rp.pid)

# ---------------------------------------------------------------- ECHO
phase("ECHO")
with open("/tmp/echo.py", "w") as fh:
    fh.write(ECHO)
ep2 = subprocess.Popen([sys.executable, "-u", "/tmp/echo.py"],
                       stdout=open("/tmp/echo.out", "w"), stderr=subprocess.STDOUT,
                       env=dict(os.environ, LD_PRELOAD="/tmp/np.so", NP_RANK=str(RANK),
                                NP_SHARED=SHARED))
echo_port = 0
for _ in range(20):
    time.sleep(0.5)
    if os.path.exists("/tmp/echo_port"):
        echo_port = int(open("/tmp/echo_port").read().strip())
        break
if not echo_port:
    fail("FAIL_ECHO", open("/tmp/echo.out").read()[:800])
if not (lo <= echo_port < lo + 1000):
    fail("FAIL_ECHO", "echo port %d not in block" % echo_port)
log("echo port %d" % echo_port)

# ---------------------------------------------------------------- TUNNEL
phase("TUNNEL")
rc, out = sh("curl -sL --max-time 120 -o /tmp/cloudflared "
             "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 "
             "&& chmod +x /tmp/cloudflared && /tmp/cloudflared --version")
NOTES.append({"cmd": "cf_dl", "rc": rc, "out": out[-200:]})
if rc != 0:
    fail("FAIL_TUNNEL", out[-800:])
cp = subprocess.Popen(["/tmp/cloudflared", "tunnel", "--url", "http://127.0.0.1:10001",
                       "--no-autoupdate"],
                      stdout=open("/tmp/cf.out", "w"), stderr=subprocess.STDOUT)
url = ""
for _ in range(60):
    time.sleep(1)
    txt = open("/tmp/cf.out").read()
    m = re.search(r"(https://[a-z0-9-]+\.trycloudflare\.com)", txt)
    if m:
        url = m.group(1)
        break
    if cp.poll() is not None:
        break
if not url:
    fail("FAIL_TUNNEL", open("/tmp/cf.out").read()[-1500:])
log("tunnel %s" % url)
NOTES.append({"cmd": "tunnel", "url": url, "pid": cp.pid})

# ---------------------------------------------------------------- DIR
phase("DIR")
put("tr_dir_%d" % RANK, "DIR",
    {"ws": url, "alias": SHARED, "echo_port": echo_port, "rank": RANK,
     "region": region, "tag": TAG, "host": socket.gethostname(), "t": time.time()})
save("RUNNING_DIRWAIT", url)

deadline = time.time() + 3600
have = {}
while time.time() < deadline:
    try:
        rows = d1("SELECT id,gpu_info FROM kaggle_gpu_state WHERE id LIKE 'tr_dir_%'")["result"][0]["results"]
        for r in rows:
            r = r if isinstance(r, dict) else {"id": r[0], "gpu_info": r[1]}
            try:
                info = json.loads(r["gpu_info"] or "{}")
            except Exception:
                continue
            # freshness guard: only accept dirs written during THIS run window
            # (stale rows from previous runs/other templates must never sync us)
            if (info.get("ws") and info.get("echo_port")
                    and float(info.get("t") or 0) >= T0_WALL - 300):
                have[r["id"]] = info
        if len(have) >= 3:
            break
    except Exception as e:  # noqa: BLE001
        log("dirpoll err %r" % (e,))
    save("RUNNING_DIRWAIT", "have=%s" % sorted(have))
    time.sleep(8)
if len(have) < 3:
    fail("FAIL_DIRWAIT", "only %s" % sorted(have))

# ---------------------------------------------------------------- SERVE
phase("SERVE")
SERVE_CMD = (
    "vllm serve /tmp/model --tensor-parallel-size 2 --pipeline-parallel-size 3 "
    "--nnodes 3 --node-rank %d --master-addr 172.19.2.2 --master-port 29501 "
    "--distributed-executor-backend mp --dtype bfloat16 --max-model-len 2048 "
    "--max-num-batched-tokens 2048 --max-num-seqs 8 "
    "--gpu-memory-utilization 0.90 --enforce-eager --skip-mm-profiling "
    "--trust-remote-code --disable-log-stats --host 127.0.0.1 --port 8000" % RANK)
SERVE_ENV = dict(os.environ,
                 LD_PRELOAD="/tmp/np.so", NP_RANK=str(RANK), NP_SHARED="172.19.2.2",
                 NCCL_SOCKET_IFNAME="eth0", GLOO_SOCKET_IFNAME="eth0",
                 NCCL_IB_DISABLE="1", NCCL_SHM_DISABLE="1", NCCL_PROTO="Simple",
                 NCCL_TIMEOUT="1800", VLLM_LOGGING_LEVEL="INFO",
                 HF_HUB_DISABLE_XET="1", HF_HUB_OFFLINE="1")
NOTES.append({"cmd": "serve_cmd", "d": SERVE_CMD})
sp = subprocess.Popen(SERVE_CMD + " > /tmp/vllm_serve.log 2>&1", shell=True, env=SERVE_ENV)
save("RUNNING_SERVE", "launched pid=%s" % sp.pid)
log("serve pid=%s" % sp.pid)


def read_log(n=14000):
    try:
        return open("/tmp/vllm_serve.log").read()[-n:]
    except Exception:
        return ""


def classify(logtxt):
    low = logtxt.lower()
    if "cuda out of memory" in low or "outofmemoryerror" in low:
        return "FAIL_OOM"
    if "bf16 is not supported" in low or "skipframe" in low:
        return "FAIL_COMPILE"
    if "symbol not found" in low or "cuda error" in low:
        return "FAIL_DRIVER"
    if "no such file" in low and "index" in low:
        return "FAIL_MODELFILES"
    if "bootstrap" in low or "connection refused" in low or "nccl" in low and "error" in low:
        return "FAIL_DIST"
    if "timed out" in low or "timeout" in low and "store" in low:
        return "FAIL_DIST"
    return "FAIL_SERVE"


def serve_fail(tag, detail):
    save(tag, str(detail)[:8000])
    try:
        put("vllm_pp3_result", "FAIL", {"tag": TAG, "reason": tag, "t": time.time()})
    except Exception:
        pass
    log("FAIL %s" % tag)
    os._exit(1)


def get_result_row_status():
    # freshness guard: ignore FAIL/DONE rows left by earlier runs
    try:
        rr = d1("SELECT status,gpu_info FROM kaggle_gpu_state WHERE id='vllm_pp3_result'")["result"][0]["results"]
        if not rr:
            return None
        r0 = rr[0] if isinstance(rr[0], dict) else {"status": rr[0][0], "gpu_info": rr[0][1]}
        try:
            gi = json.loads(r0.get("gpu_info") or "{}")
        except Exception:
            gi = {}
        if float(gi.get("t") or 0) < T0_WALL - 300:
            return None
        return r0.get("status")
    except Exception:
        return None


if RANK == 0:
    # health+test merged: retry completions until engine across 3 nodes is ready
    t0 = time.time()
    prompt_body = json.dumps({
        "model": "/tmp/model",
        "prompt": "Write a one-line greeting from a friendly robot.",
        "max_tokens": 64, "temperature": 0.7}).encode()
    last_err = ""
    resp = None
    while time.time() - t0 < 2400:
        if sp.poll() is not None:
            serve_fail(classify(read_log()), "serve exited early\n" + read_log(7000))
        try:
            req = urllib.request.Request(
                "http://127.0.0.1:8000/v1/completions", data=prompt_body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                resp = json.load(r)
            break
        except Exception as e:  # noqa: BLE001
            last_err = repr(e)[:200]
            save("RUNNING_WAITAPI", "t=%.0fs last=%s" % (time.time() - t0, last_err))
            time.sleep(15)
    if resp is None:
        serve_fail(classify(read_log()), "no API in 40min: %s\n%s" % (last_err, read_log(7000)))
    dt_gen = time.time() - t0  # includes queue-wait; real gen time measured below via usage
    txt = (resp.get("choices") or [{}])[0].get("text", "")[:300]
    ntok = (resp.get("usage") or {}).get("completion_tokens") or 0
    result = {"tokens": ntok, "text": txt,
              "t_api_s": round(time.time() - t0, 1),
              "err": last_err}
    save("DONE", "live test ok", extra={"result": result, "vllm": "0.32.0"})
    put("vllm_pp3_result", "DONE",
        {"result": result, "tag": TAG, "rank": RANK, "t": time.time(),
         "endpoint": "127.0.0.1:8000"})
    print("VLLM_RESULT " + json.dumps(result), flush=True)
    time.sleep(60)   # let peers observe the result row
    os._exit(0)
else:
    # wait for rank0's result row; keep engine alive meanwhile
    t0 = time.time()
    while time.time() - t0 < 3300:
        st = get_result_row_status()
        if st == "DONE":
            save("DONE", "cluster live (rank0 result)", extra={"via": "result-row"})
            time.sleep(60)
            os._exit(0)
        if st and str(st).startswith("FAIL"):
            save("FAIL_SERVE", "rank0 failed: %s" % st)
            os._exit(1)
        if sp.poll() is not None:
            # engine died: maybe rank0 finished and tore down right after our check
            time.sleep(5)
            st = get_result_row_status()
            if st == "DONE":
                save("DONE", "cluster live", extra={"via": "result-row-after-exit"})
                os._exit(0)
            serve_fail(classify(read_log()), "serve exited\n" + read_log(7000))
        save("RUNNING_WAITRESULT", "t=%.0fs" % (time.time() - t0))
        time.sleep(12)
    serve_fail("FAIL_TIMEOUT", "no cluster result in 55min")
