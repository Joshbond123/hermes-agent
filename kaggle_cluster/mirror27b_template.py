"""Qwen3.8-27B-Uncensored BF16 mirror -> Kaggle dataset v5 (dataset-metadata.json).

Downloads a 3-shard PART of orcarouter/Qwen3.8-27B-Uncensored (HF_TOKEN auth,
xet disabled), sha256s it, and publishes it as a private dataset
qwen38-27b-bf16-p{PART} on this account.  Injected at push: @@CF_*@@ @@HF_TOKEN@@ @@TAG@@ @@PART@@
Rows: model_mirror_{tag} (status/notes), model_mirror_{tag}_hb (30s heartbeat).
Disk budget: /kaggle/working = 20G; part = 3 shards (~8.6G) + CLI zip (~8.6G).
"""
import base64
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error

CF_ACCT = "@@CF_ACCOUNT@@"
KUSER = "@@KUSER@@"
KKEY = "@@KKEY@@"
CF_DB = "@@CF_DB@@"
CF_KEY = "@@CF_KEY@@"
HF_TOKEN = "@@HF_TOKEN@@"
TAG = "@@TAG@@"
PART = int("@@PART@@")
MODEL = "orcarouter/Qwen3.8-27B-Uncensored"
T0 = time.time()
ROW = "model_mirror_" + TAG
HB = ROW + "_hb"
NOTES = []
PHASE = ["INIT"]

SHARD_TMPL = "model-%05d-of-00018.safetensors"
PARTS = {1: [1, 2, 3], 2: [4, 5, 6], 3: [7, 8, 9],
         4: [10, 11, 12], 5: [13, 14, 15], 6: [16, 17, 18]}
SMALL = ["config.json", "generation_config.json", "tokenizer.json",
         "tokenizer_config.json", "chat_template.jinja", "merges.txt",
         "vocab.json", "preprocessor_config.json"]
STAGE = "/kaggle/working/qwen38-27b-bf16-p%d" % PART
DS_SLUG = "qwen38-27b-bf16-p%d" % PART


def d1(sql, params=()):
    req = urllib.request.Request(
        "https://api.cloudflare.com/client/v4/accounts/%s/d1/database/%s/query" % (CF_ACCT, CF_DB),
        data=json.dumps({"sql": sql, "params": list(params)}).encode(),
        method="POST",
        headers={"Authorization": "Bearer " + CF_KEY, "Content-Type": "application/json",
                 "User-Agent": "mirror27b"})
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


def save(status, detail="", extra=None):
    info = {"status": status, "detail": str(detail)[:16000], "notes": NOTES[-24:],
            "elapsed": round(time.time() - T0, 1), "part": PART, "tag": TAG}
    if extra:
        info.update(extra)
    put(ROW, status, info)


def log(msg):
    print("[%7.1fs] %s" % (time.time() - T0, msg), flush=True)


def phase(name, detail=""):
    PHASE[0] = name
    NOTES.append({"phase": name, "t": round(time.time() - T0, 1), "d": str(detail)[:400]})
    log("PHASE %s" % name)
    save("RUNNING_" + name, str(detail))


def heartbeat():
    while True:
        time.sleep(30)
        put(HB, "HB", {"phase": PHASE[0], "t": round(time.time() - T0, 1)})


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


# ---------------------------------------------------------------- phases
phase("ENV")
# session carries no KAGGLE_USERNAME/KEY env: stage credentials locally (never logged)
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
with open(os.path.expanduser("~/.kaggle/kaggle.json"), "w") as fh:
    json.dump({"username": KUSER, "key": KKEY}, fh)
os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)
rc, out, _ = sh("python3 -V; df -h /kaggle/working | tail -1; "
                "kaggle --version 2>&1 | tail -1; "
                "kaggle datasets list -s qwen38-27b-bf16-p%d 2>&1 | tail -4; "
                "pip cache purge 2>/dev/null; rm -rf /root/.cache/pip 2>/dev/null; "
                "df -h /kaggle/working | tail -1" % PART)
NOTES.append({"cmd": "env/prelist", "rc": rc, "out": out[-500:]})
save("RUNNING_ENV", out[-1200:])

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HOME"] = "/kaggle/working/hf_cache_m"

phase("DOWNLOAD")
shards = [SHARD_TMPL % i for i in PARTS[PART]]
os.makedirs(STAGE, exist_ok=True)
try:
    from huggingface_hub import hf_hub_download
    got = []
    for f in shards + [s for s in SMALL]:
        p0 = time.time()
        try:
            hf_hub_download(MODEL, f, token=HF_TOKEN, local_dir=STAGE)
        except Exception as e:  # small optional file (e.g. vocab.json absent)
            log("skip %s (%s)" % (f, str(e)[:80]))
            continue
        got.append(f)
        save("RUNNING_DL_FILE", json.dumps({"file": f, "n": len(got), "want": len(shards)}))
        log("dl %s %.1fs" % (f, time.time() - p0))
    phase("DOWNLOAD_OK", "%d files" % len(got))
except Exception as e:
    import traceback
    save("FAIL_DOWNLOAD", str(e) + "\n" + traceback.format_exc()[-1500:])
    os._exit(1)

phase("HASH")
hashes = {}
for f in sorted(os.listdir(STAGE)):
    p = os.path.join(STAGE, f)
    if not os.path.isfile(p):
        continue
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    hashes[f] = {"sha256": h.hexdigest(), "bytes": os.path.getsize(p)}
total = sum(v["bytes"] for v in hashes.values())
with open(os.path.join(STAGE, "MANIFEST.json"), "w") as fh:
    json.dump({"model": MODEL, "part": PART, "files": hashes}, fh, indent=1)
save("RUNNING_HASH", json.dumps({"files": len(hashes), "bytes": total})[:400])
# Real shards are uneven (HF API ground truth): exact per-part totals =
# shard bytes + 22,917,122 (8 small files). A flat >=8GiB guard false-FAILs
# parts 3/5 (8,197,072,384 shard bytes = 7.66 GiB but byte-correct).
EXPECT = {1: 9575524954, 2: 10090783834, 3: 8219989506,
          4: 10090783858, 5: 8219989506, 6: 9503437850}
want = EXPECT[PART]
if abs(total - want) > 1_000_000:
    save("FAIL_SIZE", "size mismatch: %d bytes (expected %d +/-1MB)" % (total, want))
    os._exit(1)

phase("PACKAGE")
# kaggle CLI 2.x refuses to create without dataset-metadata.json in the folder.
with open(os.path.join(STAGE, "dataset-metadata.json"), "w") as fh:
    json.dump({"title": DS_SLUG, "id": "%s/%s" % (KUSER, DS_SLUG),
               "licenses": [{"name": "CC0-1.0"}]}, fh, indent=1)
pkg_ok = False
for cattempt in range(3):
    rc, out, dt = sh("kaggle datasets create -p %s -q" % STAGE, timeout=7200)
    NOTES.append({"cmd": "create%d" % cattempt, "rc": rc, "t": round(dt, 1), "out": out[-700:]})
    if rc == 0 and "error:" not in out.lower():
        pkg_ok = True
        save("RUNNING_PACKAGE_OK", "create %.0fs %s" % (dt, out[-400:]))
        break
    # Failed: dataset may already exist (reuse it, verified against `want`
    # later) or the API returned a transient error (e.g. empty body when
    # several creates run concurrently) — retry.
    tb = None
    try:
        req = urllib.request.Request(
            "https://www.kaggle.com/api/v1/datasets/view/%s/%s" % (KUSER, DS_SLUG),
            headers={"Authorization": "Bearer " + KKEY, "User-Agent": "mirror27b"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            tb = json.load(resp).get("totalBytes")
    except Exception as e:  # noqa: BLE001
        NOTES.append({"cmd": "dup_view%d" % cattempt, "err": repr(e)[:200]})
    if isinstance(tb, int):
        pkg_ok = True
        save("RUNNING_PACKAGE_OK", "create rejected rc=%s; existing dataset totalBytes=%s reused"
             % (rc, tb))
        break
    save("RUNNING_PACKAGE_RETRY", "create attempt %d rc=%s; retrying" % (cattempt, rc))
    time.sleep(30)
if not pkg_ok:
    save("FAIL_PACKAGE", "create failed after retries rc=%s\n%s" % (rc, out[-3000:]))
    os._exit(1)

phase("VERIFY")
# Dataset creation is async server-side (create returns "being created" and
# list/files 403 for a while after). Poll REST view (fast, authoritative)
# up to ~15 min; CLI files listing is logged as evidence every 4th try.
def view_total():
    auths = ["Bearer " + KKEY,
             "Basic " + base64.b64encode(("%s:%s" % (KUSER, KKEY)).encode()).decode()]
    last = ""
    for auth in auths:
        try:
            req = urllib.request.Request(
                "https://www.kaggle.com/api/v1/datasets/view/%s/%s" % (KUSER, DS_SLUG),
                headers={"Authorization": auth, "User-Agent": "mirror27b"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp).get("totalBytes"), last
        except Exception as e:  # noqa: BLE001
            last = repr(e)[:300]
    return None, last

ok = False
tb = None
detail = ""
for attempt in range(20):
    tb, detail = view_total()
    if isinstance(tb, int) and want <= tb <= want + 100000:
        ok = True
        break
    if attempt % 4 == 0:
        rc, out, _ = sh("kaggle datasets files %s/%s" % (KUSER, DS_SLUG), timeout=120)
        nlisted = sum(1 for line in out.splitlines() if ".safetensors" in line)
        NOTES.append({"cmd": "verify_try%d" % attempt, "tb": tb, "files_rc": rc, "n": nlisted})
        save("RUNNING_VERIFY", "try%d tb=%s n=%d %s" % (attempt, tb, nlisted, detail))
    time.sleep(45)
if not ok:
    save("FAIL_VERIFY", "tb=%s want=%d last=%s" % (tb, want, detail))
    os._exit(1)

save("MIRROR_DONE", "part=%d files=%d bytes=%d" % (PART, len(hashes), total),
     extra={"dataset": "kaggle.com/datasets/%s/%s" % (KUSER, DS_SLUG),
            "sha256": {k: v["sha256"] for k, v in hashes.items()},
            "bytes": total})
put("model_mirror_result", "MIRROR_DONE",
    {"part": PART, "tag": TAG, "dataset": DS_SLUG, "bytes": total,
     "sha256": {k: v["sha256"] for k, v in hashes.items()}, "t": time.time()})
print("MIRROR_DONE part", PART, flush=True)
os._exit(0)
