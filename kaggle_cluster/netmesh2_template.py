"""Netmesh2: cross-node transport over cloudflared WS relay (no NET_ADMIN needed).

Per node (rank = @@PART@@):
  libnp.so (LD_PRELOAD): bind(port0)->block [41000+rank*1000,+999] encodes
  node id in advertised ports; connect() to shared IP 172.19.2.2 reroutes
  to local relay :9998, first send() prepends (owner,port) 6B header.
  relay.py: :9998 -> wss to owner's cloudflared quick tunnel (/tcp?port=N)
  -> owner's :10001 WS server -> dial 127.0.0.1:N -> raw splice.
  bore.pub is TOXIC (session SIGKILL ~T+30s, verified); cloudflared survives.
Rows: tr_state_{tag} (+_hb), tr_dir_{rank} {ws,echo_port,region}.
Terminal: TR_DONE | FAIL_*. Signals -> SIGNAL_{n} (SIGKILL leaves none).
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
    try:
        r2, w2 = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", port), timeout=8)
    except Exception as e:
        log("dialfail", port, repr(e))
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
rc, out = sh("id -u; pip install -q websockets==12.0 2>&1 | tail -1; "
             "python3 -c 'import websockets; print(\"ws\", websockets.__version__)'; "
             "curl -s --max-time 8 https://ipinfo.io/country")
NOTES.append({"cmd": "env", "rc": rc, "out": out[:600]})
region = (out.strip().split("\n")[-1] or "").strip()
log("region=%s" % region)
if "ws " not in out:
    fail("FAIL_ENV", out[:1000])

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

deadline = time.time() + 720
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
            if info.get("ws") and info.get("echo_port"):
                have[r["id"]] = info
        if len(have) >= 3:
            break
    except Exception as e:  # noqa: BLE001
        log("dirpoll err %r" % (e,))
    save("RUNNING_DIRWAIT", "have=%s" % sorted(have))
    time.sleep(8)
if len(have) < 3:
    fail("FAIL_DIRWAIT", "only %s" % sorted(have))

# ---------------------------------------------------------------- PONG
time.sleep(RANK * 15)   # stagger: avoid simultaneous handshake bursts on quick tunnels
phase("PONG")
with open("/tmp/pong.py", "w") as fh:
    fh.write(PONG_SRC)
pongs = {}
errs = []
for rk in range(3):
    if rk == RANK:
        continue
    info = have.get("tr_dir_%d" % rk, {})
    peer_ip = info.get("alias") or SHARED
    peer_port = int(info.get("echo_port") or 0)
    rtt_list = []
    okn = 0
    for i in range(5):
        nonce = "n%d-%d-%d" % (RANK, rk, int(time.time()) % 1000000)
        rc, out = sh("python3 /tmp/pong.py %s %d %s" % (peer_ip, peer_port, nonce),
                     env={"LD_PRELOAD": "/tmp/np.so", "NP_RANK": str(RANK), "NP_SHARED": SHARED})
        if rc == 0 and out.startswith("OK"):
            try:
                rtt_list.append(float(out.split()[1]))
            except Exception:
                pass
            okn += 1
        else:
            errs.append("r%d.t%d rc=%s out=%s" % (rk, i, rc, out.replace(chr(10), " ")[:220]))
            log("pong %d try%d fail %s" % (rk, i, out[:150]))
        time.sleep(0.5)
    pongs[str(rk)] = {"ok": okn, "rtt_ms": rtt_list, "port": peer_port}
    log("pong rank%d: ok=%d rtt=%s" % (rk, okn, rtt_list))

bad = [k for k, v in pongs.items() if v["ok"] == 0]
rout = ""
try:
    rout = open("/tmp/relay.out").read()[-4500:]
except Exception:
    pass
if bad:
    fail("FAIL_PONG", json.dumps(pongs) + "\nERRS:\n" + "\n".join(errs)[:1500] + "\nRELAY:\n" + rout)
save("TR_DONE", "mesh ok", extra={"pongs": pongs, "region": region, "endpoint": url,
                                  "echo_port": echo_port})
print("TR_DONE", json.dumps(pongs), flush=True)
# hold: lagging peers still need our tunnel/echo to finish their own checks
hold_t0 = time.time()
while time.time() - hold_t0 < 120:
    save("TR_DONE", "holding for peers", extra={"pongs": pongs, "region": region,
                                                "endpoint": url, "echo_port": echo_port})
    time.sleep(15)
os._exit(0)
