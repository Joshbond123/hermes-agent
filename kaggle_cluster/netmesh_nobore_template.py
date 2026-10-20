"""Netmesh v2 (CPU): LD_PRELOAD transport test — no NET_ADMIN required.

Why: sessions are root but lack CAP_NET_ADMIN (ip/iptables EPERM), and all
sessions share private IP 172.19.2.2. Design:
  - libnp.so hooks bind(port0)->per-node block [41000+N*1000, +999] so
    advertised ports encode node identity;
  - connect() to the shared IP -> rerouted to local fwd :9998, with
    (owner,port) stashed; first send() prepends a 6B header;
  - fwd routes to owner's bore tunnel (endpoint dir in D1 rows tr_dir_N);
  - bore-side fwd dials the local listener (block port) -> raw splice.
Rows: tr_state_{tag} (+_hb), tr_dir_{rank}. Terminal: TR_DONE | FAIL_*.
"""
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request

CF_ACCT = "@@CF_ACCOUNT@@"
CF_DB = "@@CF_DB@@"
CF_KEY = "@@CF_KEY@@"
TAG = "@@TAG@@"
RANK = int("@@PART@@")
T0 = time.time()
ROW = "tr_state_" + TAG
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
static int (*real_getsockname)(int, struct sockaddr*, socklen_t*);

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
            else owner = 0;                      /* master/rendezvous convention */
            if (owner == MYN) {                  /* self: dial locally, unhooked */
                struct sockaddr_in b = *in;
                b.sin_addr.s_addr = inet_addr("127.0.0.1");
                return real_connect(fd, (struct sockaddr *)&b, l);
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
           [row, str(status)[:80], "", "", "netmesh", json.dumps(info)[:16000], time.time()])
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
    t0 = time.time()
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, env=e)
        return p.returncode, (p.stdout or "") + (p.stderr or ""), time.time() - t0
    except Exception as ex:  # noqa: BLE001
        return 1, str(ex), time.time() - t0


def fail(tag, detail):
    save(tag, str(detail)[:4000])
    log("FAIL %s" % tag)
    os._exit(1)


def heartbeat():
    while True:
        time.sleep(10)
        put(HB, "HB", {"phase": PHASE[0], "t": round(time.time() - T0, 1)})


threading.Thread(target=heartbeat, daemon=True).start()

import signal as _sig
def _sigdie(signum, frame):
    try:
        put(ROW, "SIGNAL_%d" % signum,
            {"signal": signum, "t": round(time.time() - T0, 1), "phase": PHASE[0],
             "notes": NOTES[-8:]})
    except Exception:
        pass
    os._exit(128 + signum)
for _s in (_sig.SIGTERM, _sig.SIGINT, _sig.SIGHUP, _sig.SIGQUIT):
    _sig.signal(_s, _sigdie)


# ---------------------------------------------------------------- ENV
phase("ENV")
rc, out, _ = sh("id -u; which gcc cc python3; "
                "grep -E 'CapEff|CapPrm' /proc/self/status; "
                "curl -s --max-time 8 https://ipinfo.io/country")
NOTES.append({"cmd": "sysinfo", "rc": rc, "out": out[:700]})
if "not found" in out or rc != 0:
    fail("FAIL_ENV", out[:1200])
lines = out.split("\n")
if lines[0].strip() != "0":
    fail("FAIL_ENV", "not root: %s" % out[:300])
region = (lines[-1] or "").strip()
caps = [l for l in lines if l.startswith("Cap")]
NOTES.append({"cmd": "caps", "out": caps})
log("region=%s caps=%s" % (region, caps))

# ---------------------------------------------------------------- COMPILE
phase("COMPILE")
with open("/tmp/np.c", "w") as fh:
    fh.write(NP_C)
rc, out, _ = sh("gcc -shared -fPIC -O2 -o /tmp/np.so /tmp/np.c -ldl && ls -la /tmp/np.so")
NOTES.append({"cmd": "gcc", "rc": rc, "out": out[-400:]})
if rc != 0:
    fail("FAIL_COMPILE", out[:1500])

# preload self-test: bind(port0) lands in this rank's block
rc, out, _ = sh('python3 -c "import socket;s=socket.socket();s.bind((\'0.0.0.0\',0));'
                'print(s.getsockname())"', env={"LD_PRELOAD": "/tmp/np.so", "NP_RANK": str(RANK),
                                                 "NP_SHARED": SHARED})
try:
    p = int(re.search(r"0\.0\.0\.0', (\d+)", out.replace('"', "'")).group(1))
except Exception:
    p = -1
lo = 41000 + RANK * 1000
if not (lo <= p < lo + 1000):
    fail("FAIL_PRELOAD", "bind rewrite miss: out=%s want in [%d,%d)" % (out, lo, lo + 1000))
NOTES.append({"cmd": "preload_bind", "out": out.strip()[:120], "block": [lo, lo + 1000]})
log("preload bind -> %d" % p)

# ---------------------------------------------------------------- FWD daemon
phase("FWD")
FWD = r'''
import asyncio, json, socket, struct, time, urllib.request
REDIR = 9998; BORE = 10000
CF_A = "%(cfa)s"; CF_D = "%(cfd)s"; CF_K = "%(cfk)s"
MAGIC = 0xA5
DIRCACHE = {}   # rank -> bore endpoint
logf = open("/tmp/fwd.log", "a", buffering=1)

def d1(sql, params=()):
    req = urllib.request.Request(
        "https://api.cloudflare.com/client/v4/accounts/%%s/d1/database/%%s/query" %% (CF_A, CF_D),
        data=json.dumps({"sql": sql, "params": list(params)}).encode(),
        method="POST",
        headers={"Authorization": "Bearer " + CF_K, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

async def dirpoll():
    while True:
        try:
            rows = d1("SELECT id,gpu_info FROM kaggle_gpu_state WHERE id LIKE 'tr_dir_%%'")["result"][0]["results"]
            for r in rows:
                r = r if isinstance(r, dict) else {"id": r[0], "gpu_info": r[1]}
                try:
                    info = json.loads(r["gpu_info"] or "{}")
                except Exception:
                    continue
                if info.get("bore"):
                    try:
                        DIRCACHE[int(r["id"].split("_")[-1])] = info["bore"]
                    except Exception:
                        pass
        except Exception as e:
            logf.write("direrr %%s\\n" %% repr(e))
        await asyncio.sleep(3)

async def pump(a, b):
    try:
        while True:
            d = await a.read(65536)
            if not d:
                break
            b.write(d)
            await b.drain()
    except Exception:
        pass
    finally:
        try:
            b.close()
        except Exception:
            pass

async def handle_outbound(reader, writer):
    # from local LD_PRELOAD'd app: 6B header (magic, owner, port) then raw
    try:
        hdr = await asyncio.wait_for(reader.readexactly(6), timeout=15)
    except Exception as e:
        logf.write("outhdr %%s\\n" %% repr(e))
        writer.close()
        return
    if hdr[0] != MAGIC:
        logf.write("badmagic %%s\\n" %% hdr[:6])
        writer.close()
        return
    owner = hdr[1]
    port = struct.unpack(">H", hdr[2:4])[0]
    logf.write("hdr owner=%%d port=%%d\\n" %% (owner, port))
    bore = DIRCACHE.get(owner)
    if not bore:
        logf.write("nodir owner=%%d\\n" %% owner)
        writer.close()
        return
    try:
        bh, bp = bore.rsplit(":", 1)
        r2, w2 = await asyncio.wait_for(asyncio.open_connection(bh, int(bp)), timeout=15)
    except Exception as e:
        logf.write("boreconn %%s\\n" %% repr(e))
        writer.close()
        return
    w2.write(bytes([MAGIC]) + struct.pack(">H", port))
    await w2.drain()
    await asyncio.gather(pump(reader, w2), pump(w2, reader))

async def handle_inbound(reader, writer):
    # from bore: 3B header (magic, port) then raw -> dial local listener
    try:
        hdr = await asyncio.wait_for(reader.readexactly(3), timeout=15)
    except Exception as e:
        logf.write("inhdr %%s\\n" %% repr(e))
        writer.close()
        return
    if hdr[0] != MAGIC:
        writer.close()
        return
    port = struct.unpack(">H", hdr[1:3])[0]
    w2 = None
    for host in ("127.0.0.1", "0.0.0.0"):
        try:
            r2, w2 = await asyncio.wait_for(asyncio.open_connection(
                host if host != "0.0.0.0" else "127.0.0.1", port), timeout=5)
            break
        except Exception:
            continue
    if w2 is None:
        logf.write("localdial %%d\\n" %% port)
        writer.close()
        return
    await asyncio.gather(pump(reader, w2), pump(w2, reader))

async def main():
    asyncio.ensure_future(dirpoll())
    s1 = await asyncio.start_server(handle_outbound, "127.0.0.1", REDIR)
    s2 = await asyncio.start_server(handle_inbound, "0.0.0.0", BORE)
    logf.write("fwd up\\n")
    await asyncio.gather(s1.serve_forever(), s2.serve_forever())

asyncio.run(main())
''' % {"cfa": CF_ACCT, "cfd": CF_DB, "cfk": CF_KEY}
with open("/tmp/fwd.py", "w") as fh:
    fh.write(FWD)
p = subprocess.Popen([sys.executable, "-u", "/tmp/fwd.py"],
                     stdout=open("/tmp/fwd.out", "w"), stderr=subprocess.STDOUT)
time.sleep(2)
if p.poll() is not None:
    fail("FAIL_FWD", open("/tmp/fwd.out").read()[:1500])
log("fwd pid=%d" % p.pid)
NOTES.append({"cmd": "fwd_pid", "pid": p.pid})

ep = "SKIP"

# ---------------------------------------------------------------- ECHO (preload'd, block port)
phase("ECHO")
ECHO_PY = r'''
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
            c.sendall(d)
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
with open("/tmp/echo.py", "w") as fh:
    fh.write(ECHO_PY)
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
lo = 41000 + RANK * 1000
if not (lo <= echo_port < lo + 1000):
    fail("FAIL_ECHO", "echo port %d not in block [%d,%d)" % (echo_port, lo, lo + 1000))
log("echo port %d (pid %d)" % (echo_port, ep2.pid))
NOTES.append({"cmd": "echo", "port": echo_port, "pid": ep2.pid})

# ---------------------------------------------------------------- DIR
phase("DIR")
put("tr_dir_%d" % RANK, "DIR",
    {"alias": SHARED, "bore": ep, "tag": TAG, "rank": RANK, "region": region,
     "echo_port": echo_port, "host": socket.gethostname(), "t": time.time()})
save("RUNNING_DIRWAIT", ep)

# ALIVE loop: prove session survives without bore (A/B cancel isolation)
for i in range(12):
    save("RUNNING_ALIVE", "alive %d/12" % (i + 1))
    time.sleep(5)
save("TR_ALIVE", "alive w/o bore", extra={"region": region, "endpoint": ep})
print("TR_ALIVE", flush=True)
os._exit(0)
