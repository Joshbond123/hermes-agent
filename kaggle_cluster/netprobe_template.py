"""Blackthorn netprobe v2 — nonce-disambiguated cross-session connectivity + region + bandwidth.

Injected at push time: @@TAG@@, @@KAGGLE_USER@@, @@KAGGLE_TOKEN@@,
@@CF_ACCOUNT@@, @@CF_DB@@, @@CF_KEY@@.  Stdlib only.
"""
import json
import socket
import subprocess
import threading
import time
import urllib.request
import urllib.error

CF_ACCT = "@@CF_ACCOUNT@@"
CF_DB = "@@CF_DB@@"
CF_KEY = "@@CF_KEY@@"
TAG = "@@TAG@@"
MYTOK = "@@KAGGLE_TOKEN@@"
MYUSER = "@@KAGGLE_USER@@"
ROW = "np_" + TAG
PEER_ROW = "np_p2" if TAG == "p1" else "np_p1"
PORT = 29501
T0 = time.time()
HOSTID = socket.gethostname()


def log(msg):
    print("[%6.1fs] %s" % (time.time() - T0, msg), flush=True)


def d1(sql, params=()):
    req = urllib.request.Request(
        "https://api.cloudflare.com/client/v4/accounts/%s/d1/database/%s/query" % (CF_ACCT, CF_DB),
        data=json.dumps({"sql": sql, "params": list(params)}).encode(),
        method="POST",
        headers={"Authorization": "Bearer " + CF_KEY, "Content-Type": "application/json",
                 "User-Agent": "blackthorn-netprobe2"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def d1_put(status, info):
    d1("INSERT INTO kaggle_gpu_state (id, status, tunnel_url, api_key, model, gpu_info, updated_at) "
       "VALUES (?, ?, '', '', ?, ?, ?) ON CONFLICT(id) DO UPDATE SET status=excluded.status, "
       "gpu_info=excluded.gpu_info, updated_at=excluded.updated_at;",
       [ROW, status, "netprobe2", json.dumps(info), time.time()])


def d1_get(row):
    res = d1("SELECT status, gpu_info FROM kaggle_gpu_state WHERE id = ? LIMIT 1;", [row])
    rows = res["result"][0].get("results") or []
    if not rows:
        return None
    try:
        return rows[0]["status"], json.loads(rows[0]["gpu_info"] or "{}")
    except Exception:
        return rows[0]["status"], {"raw": rows[0].get("gpu_info")}


def sh(cmd, timeout=8):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (p.stdout + p.stderr).strip()
    except Exception as e:
        return "ERR " + str(e)


# ------------------------------------------------------------------ local info
info = {
    "tag": TAG,
    "host": HOSTID,
    "addrs": sh("ip -4 -o addr show | sed 's#^##'"),
    "routes": sh("ip route"),
    "arp": sh("cat /proc/net/arp"),
    "egress": None,
}
try:
    with urllib.request.urlopen("https://ipinfo.io/json", timeout=10) as r:
        info["egress"] = json.loads(r.read().decode()[:600])
except Exception as e:
    info["egress_err"] = str(e)
log("local: addrs=%s egress=%s" % (info["addrs"].replace("\n", " | ")[:200],
                                   json.dumps(info["egress"] or {})[:160]))

# ------------------------------------------------------------------ listener: replies PONG|HOSTID|TAG|stamp
def serve():
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", PORT))
    srv.listen(64)

    def loop():
        while True:
            c, addr = srv.accept()

            def handle(c=c, addr=addr):
                try:
                    c.settimeout(15)
                    data = c.recv(256)
                    if data.startswith(b"PING"):
                        c.sendall(b"PONG|" + HOSTID.encode() + b"|" + TAG.encode() + b"|" + data[4:])
                    c.close()
                except Exception:
                    pass

            threading.Thread(target=handle, daemon=True).start()

    threading.Thread(target=loop, daemon=True).start()


try:
    serve()
    info["listening"] = True
except Exception as e:
    info["listening"] = False
    info["listen_err"] = str(e)


def probe_once(addr, port, expect_tag, with_bw=False):
    r = {"addr": addr, "port": port}
    t0 = time.time()
    try:
        s = socket.create_connection((addr, port), timeout=5)
        peer_name = list(s.getpeername())
        r["getpeername"] = peer_name
        t_conn = time.time()
        stamp = ("%.6f" % t0).encode()
        s.sendall(b"PING" + stamp)
        buf = b""
        while len(buf) < 80 and (not buf.startswith(b"PONG") or buf.count(b"|") < 3):
            chunk = s.recv(256)
            if not chunk:
                break
            buf += chunk
        t_done = time.time()
        parts = buf.split(b"|", 3)
        resp = {"ok": buf.startswith(b"PONG"), "rtt_ms": round((t_done - t0) * 1000, 2),
                "connect_ms": round((t_conn - t0) * 1000, 2)}
        if buf.startswith(b"PONG") and len(parts) >= 3:
            resp["peer_host"] = parts[1].decode(errors="replace")
            resp["peer_tag"] = parts[2].decode(errors="replace")
            resp["same_host_as_me"] = resp["peer_host"] == HOSTID
            resp["is_peer"] = resp["peer_tag"] == expect_tag and resp["peer_host"] != HOSTID
            resp["self"] = resp["peer_host"] == HOSTID
        if with_bw and resp.get("ok") and resp.get("is_peer"):
            try:
                s.settimeout(60)
                payload = b"\xa5" * (1 << 20)  # 1 MiB
                bw0 = time.time()
                sent = 0
                for _ in range(64):
                    s.sendall(payload)
                    sent += len(payload)
                s.shutdown(socket.SHUT_WR)
                # drain until close/timeout
                while True:
                    try:
                        if not s.recv(1 << 16):
                            break
                    except Exception:
                        break
                bw1 = time.time()
                r["bw_MBps"] = round(sent / (bw1 - bw0) / 1e6, 1)
            except Exception as e:
                r["bw_err"] = str(e)[:120]
        s.close()
        r.update(resp)
    except Exception as e:
        r["err"] = str(e)[:120]
    return r


def candidates_from(info):
    c = []
    # interface networks' .1/.3 (docker gateways/neighbours)
    for line in (info.get("addrs") or "").splitlines():
        for token in line.split():
            if "/" in token and token[0].isdigit():
                base = token.split("/")[0]
                parts = base.split(".")
                if len(parts) == 4:
                    for last in ("1", "2", "3", "254", "253"):
                        c.append(".".join(parts[:3] + [last]))
    for line in (info.get("routes") or "").splitlines():
        toks = line.split()
        if "via" in toks:
            i = toks.index("via")
            if i + 1 < len(toks):
                c.append(toks[i + 1])
    # de-dup, drop obvious self
    seen, out = set(), []
    for x in c:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out[:18]


# ------------------------------------------------------------------ announce + test
info["phase"] = "announcing"
d1_put("ANNOUNCING2", info)

deadline = time.time() + 300
peer = None
while time.time() < deadline:
    got = d1_get(PEER_ROW)
    if got and (got[1].get("addrs") or got[1].get("egress")):
        peer = got[1]
        break
    time.sleep(3)

results = {}
if peer:
    peer_ip = None
    for line in (peer.get("addrs") or "").splitlines():
        for token in line.split():
            if token.startswith("inet ") or ("/" in token and token[0].isdigit() and "inet" in line):
                pass
    # parse peer inet: "inet 172.19.2.2/16 brd ..."
    import re as _re
    m = _re.search(r"inet\s+(\d+\.\d+\.\d+\.\d+)/", peer.get("addrs") or "")
    peer_ip = m.group(1) if m else None
    results["peer_ip"] = peer_ip
    results["peer_host_id"] = peer.get("host")
    results["peer_egress"] = peer.get("egress")
    if peer_ip:
        results["direct_peer"] = probe_once(peer_ip, PORT, peer.get("tag") or "", with_bw=True)
    # sweep (nonce-aware: we record whoever answers, incl. new hosts)
    sweep = {}
    for addr in candidates_from(info):
        if addr == peer_ip:
            continue
        r = probe_once(addr, PORT, peer.get("tag") or "")
        sweep[addr] = r
    results["sweep"] = sweep
else:
    results["error"] = "peer never announced"

info["results"] = results
info["phase"] = "done"
d1_put("DONE2", info)
log("results: %s" % json.dumps(results)[:800])
time.sleep(3)
