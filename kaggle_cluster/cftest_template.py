"""Cloudflared quick-tunnel survival probe (CPU): does Kaggle kill cloudflared like bore?

Row: cf_alive_{tag} (+_hb). Terminal TR_CF_ALIVE (URL +90s survival) | FAIL_*.
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request

CF_ACCT = "@@CF_ACCOUNT@@"
CF_DB = "@@CF_DB@@"
CF_KEY = "@@CF_KEY@@"
TAG = "@@TAG@@"
T0 = time.time()
ROW = "cf_alive_" + TAG
HB = ROW + "_hb"
NOTES = []
PHASE = ["INIT"]


def d1(sql, params=()):
    req = urllib.request.Request(
        "https://api.cloudflare.com/client/v4/accounts/%s/d1/database/%s/query" % (CF_ACCT, CF_DB),
        data=json.dumps({"sql": sql, "params": list(params)}).encode(),
        method="POST",
        headers={"Authorization": "Bearer " + CF_KEY, "Content-Type": "application/json",
                 "User-Agent": "cftest"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def put(row, status, info):
    try:
        d1("INSERT INTO kaggle_gpu_state (id,status,tunnel_url,api_key,model,gpu_info,updated_at) "
           "VALUES (?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET status=excluded.status, "
           "gpu_info=excluded.gpu_info, updated_at=excluded.updated_at;",
           [row, str(status)[:80], "", "", "cftest", json.dumps(info)[:16000], time.time()])
    except Exception as e:  # noqa: BLE001
        print("D1FAIL", row, repr(e), flush=True)


def save(status, detail="", extra=None):
    info = {"status": status, "detail": str(detail)[:8000], "notes": NOTES[-20:],
            "elapsed": round(time.time() - T0, 1), "tag": TAG}
    if extra:
        info.update(extra)
    put(ROW, status, info)


def log(msg):
    print("[%7.1fs] %s" % (time.time() - T0, msg), flush=True)


def sh(cmd, timeout=300):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except Exception as e:  # noqa: BLE001
        return 1, str(e)


def fail(tag, detail):
    save(tag, str(detail)[:3000])
    os._exit(1)


def heartbeat():
    while True:
        time.sleep(10)
        put(HB, "HB", {"phase": PHASE[0], "t": round(time.time() - T0, 1)})


threading.Thread(target=heartbeat, daemon=True).start()

PHASE[0] = "ENV"
log("ENV")
rc, out = sh("curl -sL --max-time 120 -o /tmp/cloudflared "
             "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 "
             "&& chmod +x /tmp/cloudflared && /tmp/cloudflared --version")
NOTES.append({"cmd": "dl", "rc": rc, "out": out[-300:]})
if rc != 0:
    save("FAIL_DL", out[-1000:])
    os._exit(1)

PHASE[0] = "ORIGIN"
# trivial origin so cloudflared has something to point at
ORIGIN = ("import http.server, socketserver\nclass H(http.server.BaseHTTPRequestHandler):\n"
          "    def do_GET(self):\n"
          "        self.send_response(200); self.end_headers(); self.wfile.write(b'pong')\n"
          "    def log_message(self, *a):\n"
          "        pass\n"
          "socketserver.TCPServer(('127.0.0.1', 18080), H).serve_forever()\n")
open("/tmp/origin.py", "w").write(ORIGIN)
subprocess.Popen([sys.executable, "-u", "/tmp/origin.py"],
                 stdout=open("/tmp/origin.out", "w"), stderr=subprocess.STDOUT)
time.sleep(1)
p = subprocess.Popen(["/tmp/cloudflared", "tunnel", "--url", "http://127.0.0.1:18080",
                      "--no-autoupdate"],
                     stdout=open("/tmp/cf.out", "w"), stderr=subprocess.STDOUT)
url = ""
for _ in range(45):
    time.sleep(1)
    txt = open("/tmp/cf.out").read()
    m = re.search(r"(https://[a-z0-9-]+\.trycloudflare\.com)", txt)
    if m:
        url = m.group(1)
        break
    if p.poll() is not None:
        break
if not url:
    save("FAIL_URL", open("/tmp/cf.out").read()[-1500:])
    os._exit(1)
log("url %s" % url)
PHASE[0] = "ALIVE"
save("RUNNING_CF", url, extra={"url": url, "pid": p.pid})

# let cloudflared finish routing, then self-check via curl (records code + body)
time.sleep(15)
self_ok = 0
tries = []
for i in range(9):
    rc, out = sh("curl -sS -m 25 -o /tmp/self.out -w 'HTTP%%{http_code}' '%s/' 2>&1; echo; "
                 "head -c 64 /tmp/self.out 2>/dev/null" % url)
    body = open("/tmp/self.out", "rb").read(64) if os.path.exists("/tmp/self.out") else b""
    cfrun = "cf_alive" if p.poll() is None else "cf_dead(%s)" % p.poll()
    tries.append({"i": i, "rc": rc, "out": out[:120], "cf": cfrun})
    if body == b"pong":
        self_ok += 1
        break
    log("self try %d rc=%s out=%s cf=%s" % (i, rc, out[:100], cfrun))
    time.sleep(8)
save("RUNNING_ALIVE", "self_ok=%d tries=%s" % (self_ok, json.dumps(tries)[:600]),
     extra={"url": url})

if self_ok >= 1:
    save("TR_CF_ALIVE", "cloudflared survived 60s + self-fetch ok x%d" % self_ok,
         extra={"url": url, "self_ok": self_ok})
    print("TR_CF_ALIVE", url, flush=True)
    os._exit(0)
cf_tail = ""
try:
    cf_tail = open("/tmp/cf.out").read()[-900:]
except Exception:
    pass
save("FAIL_SELF", "tunnel up (%s) self_ok=%d\nTRIES:%s\nCFOUT:%s"
     % (url, self_ok, json.dumps(tries)[:800], cf_tail), extra={"url": url})
os._exit(1)
