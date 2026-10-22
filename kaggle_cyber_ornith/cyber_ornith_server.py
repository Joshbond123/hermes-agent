# ==============================================================================
# 🛡️ Qwen3.8-27B-Uncensored — Kaggle dual-T4 API Server (cache-aware fast start)
# ==============================================================================
# Model : ressl/Qwen3.8-27B-uncensored-GGUF (Q4_K_M ~16.8 GB)
# Engine: llama.cpp CUDA (built on-device) pinned to GPU 0 (split-mode none, layers that
#         do not fit spill to system RAM — never onto GPU 1)
# Tunnel: stable Cloudflare named tunnel on blacktornagent.ing.ng (quick tunnel only as fallback)
# Auth  : Bearer API key
# GPUs  : T4 #0 hosts the model ONLY; T4 #1 is the agent computer (commands, files, GPU tasks)
#
# Boot order (cache first, never re-download if possible):
#   1. Persistent /kaggle/working GGUF or ollama store
#   2. Attached private Kaggle dataset
#   3. Hugging Face resumable download (then publish as dataset)
# Engine prefers CUDA llama-server (stable on large GGUF). Ollama is fallback only.
# Status is reported to D1 at every stage.
# ==============================================================================

import concurrent.futures
import contextlib
import threading
import hashlib
import json
import os
import re
import shutil
import subprocess
import zipfile
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# --------------------- CONFIGURATION ---------------------
import secrets as _secrets
# Gateway bearer key: injected, or generated per boot and published to D1 (never a constant in source).
API_KEY = (os.environ.get("QWEN38_API_KEY") or os.environ.get("CYBER_ORNITH_API_KEY")
           or ("bt-" + _secrets.token_urlsafe(32)))
MODEL_QUANT = os.environ.get("QWEN38_QUANT", os.environ.get("CYBER_ORNITH_QUANT", "Q4_K_M"))
MODEL_ALIAS = "Qwen3.8-27B-Uncensored"
OLLAMA_MODEL_NAME = "qwen38:uncensored"
GGUF_BASENAME = "Qwen3.8-27B-uncensored-{quant}.gguf"
HF_REPO = "ressl/Qwen3.8-27B-uncensored-GGUF"
HF_GGUF_URL = f"https://huggingface.co/{HF_REPO}/resolve/main/{GGUF_BASENAME.format(quant=MODEL_QUANT)}"
FALLBACK_REPO = "orcarouter/Qwen3.8-27B-Uncensored-GGUF"
# Sizes from HF API (bytes). SHA left empty when unknown — size check still applies.
MODEL_SHA256 = {
    "Q4_K_M": "",
    "Q4_K_S": "",
    "Q5_K_M": "",
    "Q6_K": "",
    "Q8_0": "",
    "IQ4_XS": "",
}
MODEL_BYTES = {
    "Q4_K_M": 16810714496,
    "Q4_K_S": 15825298816,
    "Q5_K_M": 19535701376,
    "Q6_K": 22430999936,
    "Q8_0": 29047084416,
    "IQ4_XS": 15309039200,
}

GATEWAY_PORT = 8000
OLLAMA_PORT = 11434
# Dual T4 (~15GB each): keep model on GPU, limit context to avoid OOM on KV cache
OLLAMA_NUM_PARALLEL = int(os.environ.get("OLLAMA_NUM_PARALLEL", "1"))
OLLAMA_MAX_LOADED_MODELS = "1"
OLLAMA_FLASH_ATTENTION = "1"
DEFAULT_NUM_CTX = int(os.environ.get("QWEN38_NUM_CTX", "8192"))
DEFAULT_NUM_GPU_LAYERS = int(os.environ.get("QWEN38_NUM_GPU_LAYERS", "999"))  # offload all layers to GPU
TELEMETRY_TOPIC = "qwen38_kaggle_blackthorn_8492"
MAX_RUNTIME_SECONDS = int(os.environ.get("MAX_RUNTIME_SECONDS", str(11 * 3600)))

KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME", "")
KAGGLE_API_TOKEN = os.environ.get("KAGGLE_API_TOKEN", "")
CACHE_DATASET_SLUG = os.environ.get("QWEN38_CACHE_DATASET", f"{KAGGLE_USERNAME}/qwen38-27b-uncensored-model-cache")
CACHE_DATASET_TITLE = "Qwen3.8-27B-Uncensored Model Cache"

INPUT_ROOT = "/kaggle/input"
WORK_ROOT = "/kaggle/working"
# Prefer /kaggle/tmp for the large GGUF when present (more free space than working)
_tmp = "/kaggle/tmp" if os.path.isdir("/kaggle/tmp") else WORK_ROOT
PERSIST_MODELS_DIR = f"{WORK_ROOT}/ollama_models"
ROLE = (os.environ.get("ORNITH_ROLE") or "model").strip().lower()      # "model" | "computer"
COMPUTER_WORKSPACE = os.environ.get("COMPUTER_WORKSPACE") or f"{WORK_ROOT}/blackthorn_workspace"
STATE_DATASET = (os.environ.get("COMPUTER_STATE_DATASET") or "").strip()   # e.g. josh787/blackthorn-computer-state
STATE_DIR = "/tmp/computer_state"
STATE_MANIFEST = f"{STATE_DIR}/manifest.json"
CHECKPOINT_INTERVAL_S = int(os.environ.get("COMPUTER_CHECKPOINT_S") or "120")
BLIND_DISK_DIR = f"{_tmp}/model_cache"
UPLOAD_DIR = f"{_tmp}/cache_upload"  # same filesystem as model_cache: hardlink, no extra 16 GB copy
MARKER_PATH = f"{BLIND_DISK_DIR}/.verified"
LOG_PATH = "/tmp/qwen38_uncensored.log"

DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen3.8-27B-Uncensored running on Blackthorn. You are a capable, direct assistant with full tool access via Hermes. Be concise, accurate, and action-oriented."
    "and CLI reasoning engine developed by DuoNeural. Always reason step-by-step "
    "inside <think>...</think> tags before providing your final response."
)

_LOGS: list = []


# --------------------- STATUS / TELEMETRY ---------------------
def log(msg: str) -> None:
    # The gateway's owner-only /logs endpoint reads this same boot log. Never
    # write Kaggle access tokens or bearer headers to notebook output or disk.
    safe = re.sub(r"KGAT_[A-Za-z0-9_-]+|github_pat_[A-Za-z0-9_]+|rnd_[A-Za-z0-9]+", "[redacted]", str(msg))
    safe = re.sub(r"(?i)Bearer\s+[^\s]+", "Bearer [redacted]", safe)
    line = f"[{time.strftime('%H:%M:%S')}] {safe}"
    _LOGS.append(line)
    del _LOGS[:-400]
    try:
        with open("/tmp/qwen38.log", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass
    print(line, flush=True)


_LAST_TUNNEL_URL = ""


# --------------------- LIVE PROGRESS (real bytes, never guesses) ---------------------
# Long boot steps (model download, unpacking, ollama pull) used to be silent. The Render-side watchdog treats a boot
# stage that stays quiet for 4 minutes as stuck and re-pushes the kernel, so a slow-but-healthy download could be killed
# by its own watchdog. This reporter publishes the REAL bytes transferred and refreshes `updated_at` only while the bytes
# keep advancing: moving => alive, frozen for PROGRESS_STALL_SECONDS => stop refreshing so a genuine hang is still healed.
# Everything here is best-effort: reporting must never be able to break a boot.
BOOT_STAGES = {
    "CHECKING_ENVIRONMENT", "INSTALLING_DEPS", "CHECKING_CACHE", "CACHE_MISS", "DOWNLOADING_MODEL", "MODEL_DOWNLOADED",
    "VERIFYING_MODEL", "STARTING_OLLAMA", "LOADING_MODEL", "STARTING_GATEWAY", "TUNNEL_ONLINE", "WARMING_GPU", "CACHE_HIT",
}
PROGRESS_EVERY = 10.0
PROGRESS_STALL_SECONDS = 240.0
STAGE_MAX_SECONDS = 1500.0  # a non-measurable stage may stay "alive" this long before it stops refreshing


class _Progress:
    def __init__(self) -> None:
        self.status = ""
        self.since = 0.0
        self.label = ""
        self.total = 0
        self.probe = None
        self.last_bytes = -1
        self.last_moved = 0.0


PROGRESS = _Progress()


def progress_stage(status: str) -> None:
    PROGRESS.status, PROGRESS.since = status, time.time()
    PROGRESS.label, PROGRESS.total, PROGRESS.probe, PROGRESS.last_bytes, PROGRESS.last_moved = "", 0, None, -1, time.time()


@contextlib.contextmanager
def progress_probe(label: str, total: int, probe):
    """While the block runs, publish ``probe()`` (bytes done so far) out of ``total``."""
    PROGRESS.label, PROGRESS.total, PROGRESS.probe = label, int(total or 0), probe
    PROGRESS.last_bytes, PROGRESS.last_moved = -1, time.time()
    try:
        yield
    finally:
        PROGRESS.probe = None


def dir_bytes(path: str) -> int:
    total = 0
    try:
        for root, _dirs, files in os.walk(path):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    pass
    except OSError:
        pass
    return total


def file_bytes(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _d1_exec(sql: str, params: list) -> None:
    acct, db, tok = (os.environ.get(k, "") for k in ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_D1_DATABASE_ID", "CLOUDFLARE_API_TOKEN"))
    if not (acct and db and tok):
        return
    req = urllib.request.Request(
        f"https://api.cloudflare.com/client/v4/accounts/{acct}/d1/database/{db}/query",
        data=json.dumps({"sql": sql, "params": params}).encode("utf-8"),
        headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json", "User-Agent": "HermesKaggleGPU/1.0"},
        method="POST",
    )
    urllib.request.urlopen(req, timeout=8)


def progress_tick(now: float = 0.0) -> bool:
    """Publish one progress sample. Returns True when `updated_at` was refreshed (the stage is alive)."""
    now = now or time.time()
    status = PROGRESS.status
    if status not in BOOT_STAGES:
        return False
    done = None
    if PROGRESS.probe is not None:
        try:
            done = int(PROGRESS.probe())
        except Exception:
            done = None
    if done is not None:
        if done > PROGRESS.last_bytes:
            PROGRESS.last_bytes, PROGRESS.last_moved = done, now
        alive = (now - PROGRESS.last_moved) <= PROGRESS_STALL_SECONDS
    else:
        alive = (now - PROGRESS.since) <= STAGE_MAX_SECONDS
    detail = {"stage": status, "label": PROGRESS.label, "elapsed": int(now - PROGRESS.since)}
    if done is not None and PROGRESS.total:
        detail.update(bytes_done=done, bytes_total=PROGRESS.total)
    if not alive:
        detail["stalled"] = True
    try:
        if alive:
            _d1_exec("UPDATE kaggle_gpu_state SET detail = ?, updated_at = ? WHERE id = 'primary' AND status = ?",
                     [json.dumps(detail), now, status])
        else:
            _d1_exec("UPDATE kaggle_gpu_state SET detail = ? WHERE id = 'primary' AND status = ?", [json.dumps(detail), status])
    except Exception:
        return False
    return alive


def _progress_loop() -> None:
    while True:
        time.sleep(PROGRESS_EVERY)
        try:
            progress_tick()
        except Exception:
            pass


def start_progress_reporter() -> None:
    threading.Thread(target=_progress_loop, daemon=True, name="blackthorn-progress").start()



def notify_workspace(status: str, tunnel_url: str = "", extra: dict = None) -> None:
    """Publish live status + tunnel URL to the Blackthorn workspace (ntfy + D1).

    The tunnel URL is sticky: a later status update never blanks the URL the
    backend routes chat traffic to (that would look like "Connection Lost"
    while the GPU is perfectly healthy).
    """
    global _LAST_TUNNEL_URL
    try:
        progress_stage(status)
    except Exception:
        pass
    if tunnel_url:
        _LAST_TUNNEL_URL = tunnel_url
    elif status in ("SERVICE_DOWN", "TUNNEL_ERROR", "BOOT_FAILED", "GPU_STOPPED_SAVING_QUOTA"):
        _LAST_TUNNEL_URL = ""
    else:
        tunnel_url = _LAST_TUNNEL_URL
    payload = {
        "status": status,
        "tunnel_url": tunnel_url,
        "model": MODEL_ALIAS,
        "quant": MODEL_QUANT,
        "timestamp": int(time.time()),
    }
    if extra:
        payload.update(extra)
    # The ntfy topic is public. The gateway key is never published there; it is stored in D1 only.
    payload.pop("api_key", None)
    try:
        req = urllib.request.Request(
            f"https://ntfy.sh/{TELEMETRY_TOPIC}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Title": f"Qwen3.8-27B: {status}"},
        )
        urllib.request.urlopen(req, timeout=8)
    except Exception:
        pass

    try:
        _cf_acct = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
        _cf_db = os.environ.get("CLOUDFLARE_D1_DATABASE_ID", "")
        _cf_tok = os.environ.get("CLOUDFLARE_API_TOKEN", "")
        if not (_cf_acct and _cf_db and _cf_tok):
            raise RuntimeError("Cloudflare D1 credentials are missing from the notebook environment")
        cf_url = (
            f"https://api.cloudflare.com/client/v4/accounts/{_cf_acct}"
            f"/d1/database/{_cf_db}/query"
        )
        cf_sql = (
            "INSERT OR REPLACE INTO kaggle_gpu_state (id, status, tunnel_url, api_key, model, gpu_info, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?);"
        )
        cf_params = [
            "computer" if ROLE == "computer" else "primary",
            status,
            tunnel_url or "",
            API_KEY,
            MODEL_ALIAS,
            (extra or {}).get("gpu") or (extra or {}).get("gpu_info") or ((extra or {}).get("error") and ("FAILED: " + str((extra or {}).get("error"))[:150])) or "2x NVIDIA Tesla T4",
            time.time(),
        ]
        cf_req = urllib.request.Request(
            cf_url,
            data=json.dumps({"sql": cf_sql, "params": cf_params}).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {_cf_tok}",
                "Content-Type": "application/json",
                "User-Agent": "HermesKaggleGPU/1.0",
            },
            method="POST",
        )
        urllib.request.urlopen(cf_req, timeout=8)
    except Exception as _d1_exc:
        _msg = f"D1 status publish failed: {type(_d1_exc).__name__}: {str(_d1_exc)[:160]}"
        if _msg != globals().get("_LAST_D1_ERR"):
            globals()["_LAST_D1_ERR"] = _msg
            log(_msg)

    log(f"status → {status} {json.dumps(extra) if extra else ''}")


def fail(status: str, message: str) -> None:
    """Publish a terminal failure together with its reason (surfaced in the UI)."""
    notify_workspace(status, extra={"error": message[:400], "gpu_info": f"FAILED: {message[:150]}"})
    log(f"❌ {status}: {message}")


# --------------------- MODEL CACHE DISCOVERY ---------------------
def _healthy_file(path: str, expected_bytes: int) -> bool:
    """True when a file exists at full size. Never raises."""
    try:
        if not os.path.isfile(path):
            return False
        size = os.path.getsize(path)
        if expected_bytes and size != expected_bytes:
            log(f"   size mismatch for {os.path.basename(path)}: {size} != {expected_bytes}")
            return False
        return size > 1024 * 1024
    except OSError:
        return False


def _sha256(path: str, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _marker_ok(path: str, expected_sha: str) -> bool:
    try:
        with open(MARKER_PATH) as fh:
            data = json.load(fh)
        return data.get("path") == path and data.get("sha256") == expected_sha
    except Exception:
        return False


def _write_marker(path: str, expected_sha: str) -> None:
    try:
        os.makedirs(BLIND_DISK_DIR, exist_ok=True)
        with open(MARKER_PATH, "w") as fh:
            json.dump({"path": path, "sha256": expected_sha, "at": time.time()}, fh)
    except Exception:
        pass


def find_local_store_model() -> str:
    """Model blob already materialised in the persistent ollama store.

    Ollama names blobs ``sha256-<digest of the file contents>``, so a blob that
    was created from our GGUF matches the published GGUF digest exactly.
    """
    digest = MODEL_SHA256.get(MODEL_QUANT, "")
    expected = MODEL_BYTES.get(MODEL_QUANT, 0)
    candidates = []
    if digest:
        candidates.append(os.path.join(PERSIST_MODELS_DIR, "blobs", f"sha256-{digest}"))
    blobs_dir = os.path.join(PERSIST_MODELS_DIR, "blobs")
    if os.path.isdir(blobs_dir):
        for name in os.listdir(blobs_dir):
            if name.startswith("sha256-"):
                candidates.append(os.path.join(blobs_dir, name))
    for path in candidates:
        if _healthy_file(path, expected):
            return path
    return ""


def _scan_input_tree(max_depth: int = 5) -> list:
    """Every plausible mount of the cache dataset, newest Kaggle layouts included."""
    hits = []
    if not os.path.isdir(INPUT_ROOT):
        return hits
    target = GGUF_BASENAME.format(quant=MODEL_QUANT)
    try:
        for root, dirs, files in os.walk(INPUT_ROOT):
            if root.count(os.sep) - INPUT_ROOT.count(os.sep) > max_depth:
                dirs[:] = []
                continue
            for name in files:
                if name == target or name.endswith(".gguf"):
                    hits.append(os.path.join(root, name))
    except Exception as exc:
        log(f"⚠️ input scan note: {exc}")
    return hits


def find_dataset_gguf() -> str:
    """GGUF shipped inside an attached Kaggle dataset (read-only mount).

    The mount path has moved between Kaggle releases
    (``/kaggle/input/<slug>/`` vs ``/kaggle/input/datasets/<owner>/<slug>/``), so
    the whole input tree is scanned instead of guessing one path.
    """
    expected = MODEL_BYTES.get(MODEL_QUANT, 0)
    candidates = _scan_input_tree()
    log(f"🔎 Scanning {INPUT_ROOT}: {len(candidates)} .gguf candidate(s)")
    for path in candidates[:20]:
        log(f"   • {path} ({os.path.getsize(path) if os.path.exists(path) else '?'} bytes)")
    # exact-size match first, then any file that at least looks like the model
    for path in candidates:
        if _healthy_file(path, expected):
            log(f"✅ Persistent model found in dataset mount: {path}")
            return path
    for path in candidates:
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        if expected and abs(size - expected) <= max(1024 * 1024, expected * 0.01):
            log(f"✅ Persistent model found (size within 1%): {path}")
            return path
    return ""


def blob_path_for(gguf_path: str) -> str:
    """Where ollama stores the model blob once the GGUF is registered."""
    return os.path.join(PERSIST_MODELS_DIR, "blobs", f"sha256-{MODEL_SHA256.get(MODEL_QUANT, '')}")


def download_from_cache_dataset(dest: str) -> str:
    """Pull the model from our own Kaggle dataset (datacenter-to-datacenter, no HF egress)."""
    expected = MODEL_BYTES.get(MODEL_QUANT, 0)
    zip_part = dest + ".zip.part"
    url = f"https://www.kaggle.com/api/v1/datasets/download/{CACHE_DATASET_SLUG}"
    log(f"⬇️  Checking the Kaggle dataset copy first: {CACHE_DATASET_SLUG}")
    for attempt in (1, 2):
        with progress_probe("Downloading the model archive", expected, lambda: file_bytes(zip_part)):
            proc = subprocess.run(["curl", "-fL", "--retry", "2", "--retry-delay", "2", "-C", "-",
                                   "-H", f"Authorization: Bearer {KAGGLE_API_TOKEN}",
                                   "-o", zip_part, url], capture_output=True, text=True)
        if proc.returncode != 0:
            log(f"   dataset fetch rc={proc.returncode}: {(proc.stderr or '')[-160:]}")
            if os.path.exists(zip_part):
                try:
                    os.remove(zip_part)  # a rejected/partial archive cannot be resumed
                except OSError:
                    pass
            continue
        try:
            with zipfile.ZipFile(zip_part) as zf:
                member = next((n for n in zf.namelist() if n.lower().endswith(".gguf")), "")
                if not member:
                    log("   archive carries no .gguf member — falling back")
                    break
                part = dest + ".part"
                with progress_probe("Unpacking the model", expected, lambda: file_bytes(part)):
                    with zf.open(member) as src, open(part, "wb") as dst:
                        shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
        except zipfile.BadZipFile:
            log("   incomplete archive on disk — retrying")
            if attempt == 2:
                try:
                    os.remove(zip_part)
                except OSError:
                    pass
            continue
        try:
            os.remove(zip_part)
        except OSError:
            pass
        part = dest + ".part"
        size = os.path.getsize(part) if os.path.exists(part) else 0
        if expected and size != expected:
            log(f"   extracted size mismatch ({size} vs {expected})")
            try:
                os.remove(part)
            except OSError:
                pass
            continue
        os.replace(part, dest)
        log("✅ Model pulled from the Kaggle dataset copy")
        return dest
    return ""


def _probe_remote_size(url: str) -> int:
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Qwen38Blackthorn/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            return int(resp.headers.get("Content-Length") or resp.headers.get("x-linked-size") or 0)
    except Exception as exc:
        log(f"   size probe note: {exc}")
        return 0


def _parallel_download(url: str, dest_partial: str, connections: int = 8) -> bool:
    """Ranged download over several connections — Kaggle's NIC is far faster than
    a single HTTP stream, so 8 workers noticeably cut the 5.6 GB transfer."""
    total = _probe_remote_size(url)
    if total <= 0:
        return False
    chunk = total // connections
    procs = []
    try:
        with open(dest_partial, "wb") as fh:
            fh.truncate(total)
        for i in range(connections):
            start = i * chunk
            end = total - 1 if i == connections - 1 else (start + chunk - 1)
            part = f"{dest_partial}.part{i}"
            procs.append((subprocess.Popen(
                ["curl", "-fsL", "--retry", "2", "--retry-delay", "2", "-r", f"{start}-{end}",
                 "-o", part, url],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL), part, start))
        for proc, part, start in procs:
            if proc.wait() != 0:
                return False
            want = min(chunk, total - start)
            got = os.path.getsize(part) if os.path.exists(part) else 0
            if got != want:
                log(f"   range {start} returned {got} bytes, expected {want} — falling back")
                return False
            with open(part, "rb") as src, open(dest_partial, "r+b") as dst:
                dst.seek(start)
                shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
            os.remove(part)
        return os.path.getsize(dest_partial) == total
    except Exception as exc:
        log(f"   ranged download note: {exc}")
        return False
    finally:
        for proc, part, _ in procs:
            if proc.poll() is None:
                proc.kill()
            if os.path.exists(part):
                try:
                    os.remove(part)
                except OSError:
                    pass


def download_model(dest: str) -> str:
    """Resumable, verified download. Returns (path, verified_sha_ok)."""
    expected = MODEL_BYTES.get(MODEL_QUANT, 0)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    partial = dest + ".part"
    for attempt in range(1, 5):
        if attempt == 1:
            log(f"⚡ Single-stream resumable download — {HF_GGUF_URL}")
        if attempt > 1 and os.path.exists(partial) and expected and os.path.getsize(partial) != expected:
            os.remove(partial)  # never resume a bad partial file
        have = os.path.getsize(partial) if os.path.exists(partial) else 0
        log(f"⬇️  Download attempt {attempt}: {HF_GGUF_URL} (resume at {have} bytes)")
        with progress_probe("Downloading the model", expected, lambda: file_bytes(partial)):
            proc = subprocess.run(["curl", "-fL", "--retry", "8", "--retry-delay", "5", "--retry-all-errors",
                                   "--connect-timeout", "30", "--max-time", "0",
                                   "-C", "-", "-o", partial, HF_GGUF_URL], capture_output=True, text=True)
        if proc.returncode == 0:
            size = os.path.getsize(partial)
            if not expected or size == expected:
                os.replace(partial, dest)
                return dest, "verified"
            log(f"   size mismatch: got {size}, expected {expected}")
        else:
            log(f"   curl rc={proc.returncode}: {(proc.stderr or '')[-200:]}")
        time.sleep(3)

    log("⚠️ Primary HF download failed; trying the imatrix mirror (size-checked only)")
    mirror = (f"https://huggingface.co/{FALLBACK_REPO}/resolve/main/"
              f"Qwen3.8-27B-Uncensored.i1-{MODEL_QUANT}.gguf")
    if os.path.exists(partial):
        os.remove(partial)
    subprocess.run(["curl", "-fL", "--retry", "3", "-C", "-", "-o", partial, mirror], check=True)
    if os.path.getsize(partial) < 3 * 1024 * 1024 * 1024:
        raise RuntimeError(f"mirror download too small: {os.path.getsize(partial)} bytes")
    os.replace(partial, dest)
    return dest, "mirror"


def cache_dataset_is_ready(expected_bytes: int = 0) -> bool:
    """The published cache dataset already holds a full-size model?"""
    expected = expected_bytes or MODEL_BYTES.get(MODEL_QUANT, 0)
    try:
        req = urllib.request.Request(
            f"https://www.kaggle.com/api/v1/datasets/view/{CACHE_DATASET_SLUG}",
            headers={"Authorization": f"Bearer {KAGGLE_API_TOKEN}", "User-Agent": "Qwen38Blackthorn/1.0"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8")) or {}
        total = int(data.get("totalBytesNullable") or data.get("totalBytes") or 0)
        return bool(total and (expected <= 0 or total >= expected))
    except Exception as exc:
        log(f"cache dataset check note: {exc}")
        return False


def _publish_once(gguf_path: str, gpu_out: str) -> None:
    try:
        if cache_dataset_is_ready():
            log("✅ Cache dataset already holds the model — skipping republish")
            notify_workspace("CACHE_DATASET_READY", extra={"dataset": CACHE_DATASET_SLUG, "gpu": gpu_out})
        else:
            publish_cache_dataset(gguf_path)
    except Exception as exc:  # the publish is best-effort; it must never take the running engine down
        log(f"⚠️ cache publish thread: {type(exc).__name__}: {exc}")


def _start_cache_publish(gguf_path: str, gpu_out: str = "") -> None:
    """Publish a permanent Kaggle dataset copy so the next boot skips HF.

    Runs synchronously so a short-lived session still leaves a durable cache.
    Failures are logged and never abort the boot.
    """
    try:
        if cache_dataset_is_ready():
            notify_workspace("CACHE_DATASET_READY", extra={"dataset": CACHE_DATASET_SLUG, "gpu": gpu_out})
            log(f"✅ Model cache already published: {CACHE_DATASET_SLUG}")
            return
        log(f"📦 Publishing permanent model cache to {CACHE_DATASET_SLUG} …")
        publish_cache_dataset(gguf_path)
    except Exception as exc:
        log(f"cache publish note: {exc}")


def publish_cache_dataset(gguf_path: str) -> None:
    """One-time: publish the verified GGUF as a private Kaggle dataset so future
    sessions mount it instantly instead of downloading 5.6 GB again."""
    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        target = os.path.join(UPLOAD_DIR, os.path.basename(gguf_path))
        if not _healthy_file(target, MODEL_BYTES.get(MODEL_QUANT, 0)):
            if os.path.exists(target):
                os.remove(target)
            log("📦 Seeding cache dataset upload dir (hardlink/copy of verified GGUF)...")
            try:
                os.link(gguf_path, target)
            except OSError:
                shutil.copy2(gguf_path, target)
        with open(os.path.join(UPLOAD_DIR, "dataset-metadata.json"), "w") as fh:
            json.dump(
                {
                    "title": CACHE_DATASET_TITLE,
                    "id": CACHE_DATASET_SLUG,
                    "licenses": [{"name": "other"}],
                    "isPrivate": True,
                },
                fh,
            )
        env = _kaggle_cli_env()  # KGAT bearer token, not legacy KAGGLE_KEY Basic auth.

        if shutil.which("kaggle") is None:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "--no-input", "kaggle"],
                check=False,
            )
        # Version the dataset when it already exists, otherwise create it.
        create = subprocess.run(
            ["kaggle", "datasets", "create", "-p", UPLOAD_DIR, "--dir-mode", "skip", "-q"],
            env=env, capture_output=True, text=True,
        )
        if create.returncode != 0:
            log(f"   create note: {create.stderr.strip()[:200]}")
            version = subprocess.run(
                ["kaggle", "datasets", "version", "-p", UPLOAD_DIR, "-q",
                 "-m", f"Qwen3.8-27B {MODEL_QUANT} verified {time.strftime('%Y-%m-%d')}", "--dir-mode", "skip"],
                env=env, capture_output=True, text=True,
            )
            if version.returncode != 0:
                log(f"⚠️ Cache dataset publish failed: {version.stderr.strip()[:200]}")
                return
        log(f"✅ Cache dataset published: {CACHE_DATASET_SLUG}")
        notify_workspace("CACHE_DATASET_PUBLISHED", extra={"dataset": CACHE_DATASET_SLUG})
    except Exception as exc:
        log(f"⚠️ Cache dataset publish skipped: {exc}")


# --------------------- DEPENDENCY SETUP (skip when present) ---------------------
def ensure_python_deps() -> None:
    missing = []
    for module, package in (("fastapi", "fastapi"), ("uvicorn", "uvicorn"),
                            ("httpx", "httpx"), ("pydantic", "pydantic")):
        try:
            __import__(module)
        except Exception:
            missing.append(package)
    if missing:
        log(f"📦 Installing missing python packages: {missing}")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--no-input", *missing],
            check=False,
        )
    else:
        log("✅ Python deps already present — skipping pip install")


def ensure_ollama() -> str:
    binary = shutil.which("ollama")
    if binary:
        log(f"✅ Ollama already installed: {binary}")
        return binary
    log("🦙 Installing Ollama (not present in this fresh container)...")
    for tool in ("zstd", "pciutils", "lshw"):
        if shutil.which(tool) is None:
            subprocess.run("apt-get update -qq && apt-get install -y -qq zstd pciutils lshw",
                           shell=True, check=False)
            break
    subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
    binary = shutil.which("ollama") or "/usr/local/bin/ollama"
    return binary


def start_ollama_daemon(ollama_bin: str) -> None:
    """Idempotent: bring the engine up as early as possible (during the download)."""
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{OLLAMA_PORT}/api/tags", timeout=2)
        log("✅ Ollama daemon already serving")
        return
    except Exception:
        pass
    os.makedirs(PERSIST_MODELS_DIR, exist_ok=True)
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"127.0.0.1:{OLLAMA_PORT}"
    env["OLLAMA_ORIGINS"] = "*"
    env["OLLAMA_KEEP_ALIVE"] = "-1"
    env["OLLAMA_MODELS"] = PERSIST_MODELS_DIR
    # Engine placement. The model is the only thing this engine serves, and it needs both T4s: the 27B Q4_K_M
    # weights (~17.5 GB) do not fit on one 15 GB T4. Pinning the daemon to GPU 0 left ~23% of the weights in system
    # RAM and GPU 1 idle, which measured ~3 tokens/s. Spread the layers across both cards instead.
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["OLLAMA_SCHED_SPREAD"] = "1"
    # The Modelfile's num_ctx only applies to models created from it. A registry pull (the fallback path) ran at
    # 4096 tokens while the agent budgets for DEFAULT_NUM_CTX, so set the window for every model the daemon loads.
    env["OLLAMA_CONTEXT_LENGTH"] = str(DEFAULT_NUM_CTX)
    env["OLLAMA_NUM_PARALLEL"] = str(OLLAMA_NUM_PARALLEL)
    subprocess.Popen([ollama_bin, "serve"], env=env,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{OLLAMA_PORT}/api/tags", timeout=2)
            log("✅ Ollama daemon up")
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError("ollama serve did not come up in 60s")


def ensure_ollama_serving() -> str:
    """Install (only if missing) *and* start the engine — runs during the download."""
    binary = ensure_ollama()
    start_ollama_daemon(binary)
    return binary


INROK_BIN_DIR = "/tmp/inrok_bin"
INROK_BIN = os.path.join(INROK_BIN_DIR, "inrok")
# Inrok shows a browser interstitial on shared sites unless non-browser clients send this header.
INROK_INTERSTITIAL_HEADERS = {"skip_zrok_interstitial": "true"}


def inrok_enabled() -> bool:
    return bool((os.environ.get("INROK_API_KEY") or "").strip())


def ensure_inrok() -> str:
    """Install the Inrok CLI (official installer, SHA-256 verified downloads) into /tmp."""
    if os.path.exists(INROK_BIN):
        return INROK_BIN
    os.makedirs(INROK_BIN_DIR, exist_ok=True)
    log("🌐 Installing the Inrok CLI...")
    subprocess.run(["sh", "-c", "curl -fsSL https://inrok.in/install.sh | INROK_BIN_DIR=" + INROK_BIN_DIR + " sh"],
                   check=True, timeout=300, capture_output=True)
    if not os.path.exists(INROK_BIN):
        raise RuntimeError("inrok install did not produce " + INROK_BIN)
    return INROK_BIN


# Inrok names are held by the account until the fabric releases them. A fixed name collides with the
# share left behind by the previous kernel (409 shareConflict), so every boot gets its own name. The
# tunnel URL is published to D1 on each start, so routing does not depend on the name.
_INROK_BOOT_SUFFIX = _secrets.token_hex(3)


def inrok_tunnel_name() -> str:
    explicit = (os.environ.get("INROK_TUNNEL_NAME") or "").strip()
    if explicit:
        return explicit
    base = "blackthorn" if ROLE != "computer" else "blackthorn-computer"
    return f"{base}-{_INROK_BOOT_SUFFIX}"


_INROK_MAX_STALE_RELEASE = 6


def stale_inrok_names(status_text: str, role: str, current: str) -> list:
    """Names of same-role shares left behind by earlier kernels, safe to release before this boot.

    The Inrok beta allows 3 live tunnels per account and a dead kernel's share keeps its slot. Only names
    this role issues are chosen: the model uses ``blackthorn`` or ``blackthorn-<6 hex>``, the computer uses
    ``blackthorn-computer-<6 hex>``. The current boot's name is never returned. Bounded to a few names.
    """
    if role == "computer":
        pattern = r"^blackthorn-computer-[0-9a-f]{6}$"
    else:
        pattern = r"^blackthorn(-[0-9a-f]{6})?$"
    found = []
    for line in (status_text or "").splitlines()[1:]:
        cols = line.split()
        if not cols:
            continue
        name = cols[0]
        if name != current and re.match(pattern, name) and name not in found:
            found.append(name)
    return found[:_INROK_MAX_STALE_RELEASE]


def release_stale_inrok_shares(binary: str, role: str, current: str) -> list:
    """Stop same-role stale shares so this boot's tunnel fits the account limit. Returns the names stopped."""
    status = subprocess.run([binary, "status"], capture_output=True, text=True, timeout=60)
    stopped = []
    for name in stale_inrok_names(status.stdout, role, current):
        res = subprocess.run([binary, "stop", name], capture_output=True, text=True, timeout=60)
        if res.returncode == 0:
            stopped.append(name)
        log("inrok: released stale share " + name + " (rc=" + str(res.returncode) + ")")
    return stopped


def _inrok_gateway_answers(public: str) -> bool:
    """True only when the gateway itself answers through the tunnel.

    Inrok serves its own HTML pages (interstitial, 'tunnel offline') through the same host, so an HTML
    body or a 404 is NOT the gateway. The gateway answers 200 JSON, or 401/403 for a bad key.
    """
    req = urllib.request.Request(public + "/health", headers=dict(INROK_INTERSTITIAL_HEADERS))
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            body = resp.read(4096)
            return resp.status == 200 and not body.lstrip().startswith(b"<")
    except urllib.error.HTTPError as exc:
        body = exc.read(4096) if hasattr(exc, "read") else b""
        return exc.code in (401, 403) and not body.lstrip().startswith(b"<")
    except Exception:
        return False


def launch_inrok() -> tuple:
    """Expose the local gateway at https://<name>.share.inrok.in and return (proc, url).

    The key goes to `inrok login` on stdin (never argv, never logged). Returns (None, "") when the
    tunnel does not answer in time, so the caller can report TUNNEL_ERROR instead of guessing.
    """
    key = (os.environ.get("INROK_API_KEY") or "").strip()
    name = inrok_tunnel_name()
    binary = ensure_inrok()
    login = subprocess.run([binary, "login"], input=key + "\n", capture_output=True, text=True, timeout=90)
    if login.returncode != 0:
        msg = (login.stderr or login.stdout or "").replace(key, "***")[-300:]
        log("❌ inrok login failed: " + msg)
        return None, ""
    release_stale_inrok_shares(binary, ROLE, name)
    handle = open("/tmp/inrok_" + name + ".log", "a")
    proc = subprocess.Popen([binary, "http", str(GATEWAY_PORT), "--name", name],
                            stdout=handle, stderr=subprocess.STDOUT)
    public = "https://" + name + ".share.inrok.in"
    deadline = time.time() + 150
    while time.time() < deadline:
        if proc.poll() is not None:
            log("❌ inrok tunnel exited early (see /tmp/inrok_" + name + ".log)")
            return None, ""
        if _inrok_gateway_answers(public):
            log("Inrok tunnel up: " + public)
            return proc, public
        time.sleep(2)
    log("❌ inrok tunnel did not answer within 150s: " + public)
    try:
        proc.terminate()
    except Exception:
        pass
    return None, ""


def ensure_cloudflared() -> str:
    if inrok_enabled():
        return "inrok"  # Inrok replaces cloudflared; nothing to download
    for candidate in ("/tmp/cloudflared", "/usr/local/bin/cloudflared"):
        if os.path.exists(candidate):
            log(f"✅ cloudflared present: {candidate}")
            return candidate
    binary = "/tmp/cloudflared"
    log("🌐 Fetching cloudflared...")
    subprocess.run([
        "curl", "-fsSL",
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
        "-o", binary,
    ], check=True)
    os.chmod(binary, 0o755)
    return binary


def ollama_model_present() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{OLLAMA_PORT}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        names = [m.get("name", "") for m in (data.get("models") or [])]
        return any(n.split(":")[0] == OLLAMA_MODEL_NAME.split(":")[0] for n in names)
    except Exception:
        return False





# ---------------------------------------------------------------------------
# Tunnel watchdog: detect dead public URL and relaunch inrok/cloudflared.
# A 502 on the public edge usually means the tunnel process died while the
# local uvicorn is still healthy — restart the tunnel without wiping workspace.
# ---------------------------------------------------------------------------
_tunnel_watch_stop = threading.Event()
_tunnel_watch_thread = None
_tunnel_restart_lock = threading.Lock()


def _local_health_ok() -> bool:
    try:
        import urllib.request
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/health",
            headers={"Authorization": f"Bearer {API_KEY}"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/healthz", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False


def _public_tunnel_ok(url: str) -> bool:
    if not url:
        return False
    try:
        import urllib.request
        req = urllib.request.Request(
            url.rstrip("/") + "/health",
            headers={"Authorization": f"Bearer {API_KEY}"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            return resp.status == 200
    except Exception:
        try:
            import urllib.request
            req = urllib.request.Request(url.rstrip("/") + "/healthz", method="GET")
            with urllib.request.urlopen(req, timeout=12) as resp:
                return resp.status < 500
        except Exception:
            return False


def restart_tunnel_safe(reason: str = "") -> str:
    """Restart the public tunnel once; never deletes COMPUTER_WORKSPACE."""
    with _tunnel_restart_lock:
        print(f"[tunnel-watch] restarting tunnel ({reason})", flush=True)
        try:
            url = start_tunnel()
            print(f"[tunnel-watch] tunnel up: {url}", flush=True)
            return url
        except Exception as exc:
            print(f"[tunnel-watch] restart failed: {exc}", flush=True)
            return ""


def start_tunnel_watchdog(interval_s: float = 45.0) -> None:
    global _tunnel_watch_thread
    if _tunnel_watch_thread and _tunnel_watch_thread.is_alive():
        return

    def _loop() -> None:
        consecutive_fail = 0
        while not _tunnel_watch_stop.is_set():
            try:
                url = globals().get("PUBLIC_URL") or globals().get("TUNNEL_URL") or ""
                if not url:
                    # try env / module state
                    url = os.environ.get("BLACKTHORN_PUBLIC_URL", "")
                local_ok = _local_health_ok()
                if local_ok and url and not _public_tunnel_ok(url):
                    consecutive_fail += 1
                    print(f"[tunnel-watch] public URL unhealthy ({consecutive_fail}) url={url[:60]}", flush=True)
                    if consecutive_fail >= 2:
                        restart_tunnel_safe("public health failed")
                        consecutive_fail = 0
                elif local_ok:
                    consecutive_fail = 0
            except Exception as exc:
                print(f"[tunnel-watch] error: {exc}", flush=True)
            _tunnel_watch_stop.wait(interval_s)

    _tunnel_watch_thread = threading.Thread(target=_loop, name="tunnel-watch", daemon=True)
    _tunnel_watch_thread.start()
    print("[tunnel-watch] started", flush=True)


def ensure_llama_server_bin() -> str:
    """Build or reuse a CUDA-enabled llama-server for dual T4."""
    binary = "/tmp/llama-server"
    if os.path.isfile(binary) and os.access(binary, os.X_OK):
        # quick sanity: prefer CUDA build
        try:
            out = subprocess.check_output([binary, "--version"], stderr=subprocess.STDOUT, timeout=10).decode()
            if "CUDA" in out or "ggml" in out.lower():
                log("Reusing existing llama-server binary")
                return binary
        except Exception:
            pass
    build_dir = "/tmp/llama.cpp"
    log("Building llama.cpp with CUDA (one-time, ~5-12 min on T4)...")
    if not os.path.isdir(build_dir):
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp", build_dir],
            check=True, timeout=180,
        )
    # Install minimal build deps if missing
    subprocess.run(
        ["bash", "-c", "which cmake || (apt-get update -qq && apt-get install -y -qq cmake build-essential)"],
        check=False, timeout=180,
    )
    cmake_cmd = [
        "cmake", "-B", "build",
        "-DGGML_CUDA=ON",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLAMA_CURL=OFF",
    ]
    subprocess.run(cmake_cmd, cwd=build_dir, check=True, timeout=300)
    subprocess.run(
        ["cmake", "--build", "build", "--config", "Release", "-j", "4"],
        cwd=build_dir, check=True, timeout=900,
    )
    src = os.path.join(build_dir, "build", "bin", "llama-server")
    if not os.path.isfile(src):
        # older layout
        for root, dirs, files in os.walk(os.path.join(build_dir, "build")):
            if "llama-server" in files:
                src = os.path.join(root, "llama-server")
                break
    if not os.path.isfile(src):
        raise RuntimeError("llama-server binary not found after CUDA build")
    shutil.copy2(src, binary)
    os.chmod(binary, 0o755)
    log("✅ CUDA llama-server ready at " + binary)
    return binary


def start_llama_server(gguf_path: str) -> None:
    """Start OpenAI-compatible llama-server across BOTH GPUs for maximum inference speed.

    The agent's computer runs on its own dedicated hosts (josh787 / alagbo), so this
    machine hosts the model only: all 2x T4 work for it. The 27B Q4 weights (~15.7 GiB)
    fit fully in the combined VRAM with room for the KV cache, which gives the fastest
    prefill and first token. If a smaller or split-only arrangement must be used, the
    -ngl ladder still degrades gracefully (99 -> 56 -> 40 layers on GPU, rest in RAM).
    """
    binary = ensure_llama_server_bin()
    # Context window: prefer env, default 8192 to match the agent's context budget.
    try:
        n_ctx = int(os.environ.get("QWEN38_NUM_CTX") or os.environ.get("BLACKTHORN_CONTEXT_TOKENS") or "8192")
    except ValueError:
        n_ctx = 8192
    n_ctx = max(4096, min(n_ctx, 32768))
    log(f"Starting CUDA llama-server on :{OLLAMA_PORT} with {gguf_path} (all GPUs, ctx={n_ctx})")
    logf_path = "/tmp/llama_server.log"
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)      # both T4s serve the model

    def _launch(ngl: str):
        cmd = [
            binary,
            "-m", gguf_path,
            "-ngl", ngl,
            "-c", str(n_ctx),
            "--host", "0.0.0.0",
            "--port", str(OLLAMA_PORT),
            "-np", "1",
            "--flash-attn", "on",
            "-b", "512",
            "-ub", "256",
            "--mlock",
        ]
        extra = (os.environ.get("LLAMA_EXTRA_ARGS") or "").strip()
        if extra:
            cmd += extra.split()
        logf = open(logf_path, "w")
        return subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)

    def _wait(proc, seconds: int) -> bool:
        for i in range(seconds // 2):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{OLLAMA_PORT}/health", timeout=3)
                return True
            except Exception:
                if proc.poll() is not None:
                    return False  # crashed (e.g. CUDA OOM) — caller may retry with fewer layers
                if i % 15 == 0:
                    log(f"  waiting for llama-server... ({i*2}s)")
                time.sleep(2)
        return False

    # The 27B Q4 weights (~15.7 GiB) can exceed one T4 once the KV cache is reserved.
    # Ask for full offload first; if the engine cannot start, retry with fewer GPU layers.
    # Every attempt is pinned to GPU 0: weights never land on GPU 1, no matter the fallback.
    for ngl in ("99", "56", "40"):
        proc = _launch(ngl)
        if _wait(proc, 240):
            log("llama-server healthy (ngl=" + ngl + ")")
            with open("/tmp/engine_mode.txt", "w") as fh:
                fh.write("llama")
            return
        log(f"llama-server did not come up with -ngl {ngl}; stopping it and retrying")
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        time.sleep(2)
    try:
        with open(logf_path) as lf:
            tail = lf.read()[-800:]
        log("llama-server log tail: " + tail)
    except Exception:
        pass
    raise RuntimeError("llama-server failed to start — see /tmp/llama_server.log")



def ollama_pull_model(ollama_bin: str) -> bool:
    """Pull model from Ollama registry (more reliable than raw HF GGUF on Kaggle)."""
    global OLLAMA_MODEL_NAME
    tags = [
        OLLAMA_MODEL_NAME,
        "orcarouter/qwen3.8-27b-uncensored:q4_K_M",
        "orcarouter/qwen3.8-27b-uncensored:latest",
        "huihui_ai/qwen3-abliterated:latest",
    ]
    for tag in tags:
        log(f"Trying ollama pull {tag}")
        notify_workspace("DOWNLOADING_MODEL", extra={"note": f"ollama pull {tag}"})
        try:
            with progress_probe("Pulling the model from the Ollama registry", MODEL_BYTES.get(MODEL_QUANT, 0), lambda: dir_bytes(PERSIST_MODELS_DIR)):
                proc = subprocess.run([ollama_bin, "pull", tag], capture_output=True, text=True, timeout=3600)
            if proc.returncode == 0:
                OLLAMA_MODEL_NAME = tag
                log(f"ollama pull succeeded: {tag}")
                return True
            log(f"pull failed {tag}: {(proc.stderr or proc.stdout or '')[-200:]}")
        except Exception as e:
            log(f"pull exception {tag}: {e}")
    return False


def register_model(ollama_bin: str, gguf_path: str) -> None:
    """Prefer CUDA llama-server for large GGUF stability on dual T4.
    Ollama create is attempted only when disk allows; failures fall through
    cleanly to llama-server without aborting the boot.
    """
    # Kaggle's CUDA toolkit/compiler changes between images; a fresh llama.cpp
    # CMake build can fail before any model serves. Reuse the already-running
    # Ollama engine by default, with an explicit opt-in for custom llama.cpp.
    prefer_llama = os.environ.get("PREFER_LLAMA", "0") == "1"
    if prefer_llama and gguf_path and os.path.isfile(gguf_path):
        try:
            log("Trying CUDA llama-server on dual T4")
            start_llama_server(gguf_path)
            return
        except (OSError, RuntimeError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            log(f"CUDA llama-server unavailable ({type(exc).__name__}); using Ollama")

    if ollama_model_present():
        log("✅ Model already registered in the ollama store — skipping create")
        with open("/tmp/engine_mode.txt", "w") as fh:
            fh.write("ollama")
        return
    # Free space check — ollama create can need ~2x the GGUF size briefly
    try:
        usage = shutil.disk_usage(WORK_ROOT)
        free_gb = usage.free / (1024**3)
        log(f"Disk free on {WORK_ROOT}: {free_gb:.1f} GB")
        if free_gb < 20:  # avoid a guaranteed out-of-space copy
            if prefer_llama:
                start_llama_server(gguf_path)
                return
            log("Low working-disk space; attempting Ollama import of the cached GGUF")
        # purge any partial blobs
        blobs = os.path.join(PERSIST_MODELS_DIR, "blobs")
        if os.path.isdir(blobs):
            for name in os.listdir(blobs):
                try:
                    os.remove(os.path.join(blobs, name))
                except Exception:
                    pass
            log("Purged partial ollama blobs to reclaim space")
    except Exception as e:
        log(f"disk check note: {e}")
    modelfile = "/tmp/Modelfile"
    with open(modelfile, "w") as fh:
        fh.write(f"FROM {gguf_path}\n")
        fh.write(f"PARAMETER num_ctx {DEFAULT_NUM_CTX}\n")
        fh.write(f"PARAMETER num_gpu 99\n")
        fh.write("PARAMETER temperature 0.6\n")
        fh.write("PARAMETER num_batch 256\n")
        fh.write("PARAMETER num_thread 8\n")
        fh.write('SYSTEM """' + DEFAULT_SYSTEM_PROMPT + '"""\n')
    log(f"🧱 Registering model from {gguf_path} (one-time)...")
    try:
        proc = subprocess.run(
            [ollama_bin, "create", OLLAMA_MODEL_NAME, "-f", modelfile],
            capture_output=True, text=True, timeout=600,
        )
    except Exception as e:
        log(f"ollama create exception: {e}")
        start_llama_server(gguf_path)
        return
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()[:400]
        log(f"ollama create failed (rc={proc.returncode}): {err}")
        start_llama_server(gguf_path)
        return
    with open("/tmp/engine_mode.txt", "w") as fh:
        fh.write("ollama")
    log("✅ Model registered in ollama")


# --------------------- GATEWAY + TUNNEL ---------------------
def gateway_code() -> str:
    return '''
import json, time, os, asyncio
import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

API_KEY = "''' + API_KEY + '''"
MODEL_ALIAS = "''' + MODEL_ALIAS + '''"
OLLAMA_MODEL = "''' + OLLAMA_MODEL_NAME + '''"
OLLAMA_URL = "http://127.0.0.1:''' + str(OLLAMA_PORT) + '''"
DEFAULT_SYSTEM_PROMPT = """''' + DEFAULT_SYSTEM_PROMPT + '''"""
START_TIME = time.time()
BOOT_STATE = json.load(open("/tmp/boot_state.json")) if os.path.exists("/tmp/boot_state.json") else {}
LOGS = "/tmp/qwen38.log"
CKPT_DIR = "/tmp/computer_state"

app = FastAPI(title="Qwen3.8-27B-Uncensored API Server", version="4.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


async def verify_api_key(authorization: str = Header(default=None), x_api_key: str = Header(default=None)):
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    elif x_api_key:
        token = x_api_key.strip()
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key.")
    return token


@app.get("/")
@app.get("/health")
async def health():
    """Liveness plus the honest model state; boot state is written after gateway startup."""
    try:
        with open("/tmp/boot_state.json") as fh:
            current_boot = json.load(fh)
    except (OSError, ValueError):
        current_boot = BOOT_STATE
    model_loaded = False
    engine = "unknown"
    try:
        if os.path.exists("/tmp/engine_mode.txt"):
            engine = open("/tmp/engine_mode.txt").read().strip()
    except Exception:
        pass
    try:
        async with httpx.AsyncClient(timeout=4) as client:
            if engine == "llama":
                h = await client.get(f"{OLLAMA_URL}/health")
                model_loaded = h.status_code == 200
            else:
                ps = await client.get(f"{OLLAMA_URL}/api/ps")
                if ps.status_code == 200:
                    names = [str(m.get("name") or m.get("model") or "") for m in (ps.json().get("models") or [])]
                    model_loaded = any(n.split(":")[0] == OLLAMA_MODEL.split(":")[0] for n in names)
    except Exception:
        model_loaded = bool(current_boot.get("warmup_ok"))
    return {
        "status": "online",
        "model": MODEL_ALIAS,
        "backend_model": OLLAMA_MODEL,
        "model_loaded": model_loaded,
        "gpu": current_boot.get("gpu", ""),
        "gpu_roles": ({"0": "computer", "1": "computer"} if (os.environ.get("ORNITH_ROLE") or "model") == "computer"
                      else {"0": "model", "1": "model"}),
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "boot": {k: current_boot.get(k) for k in ("cache_source", "boot_seconds", "download_skipped", "warmup_ok")},
        "endpoints": ["/v1/models", "/v1/chat/completions", "/chat", "/logs", "/health", "/computer/info", "/computer/exec", "/computer/list_files", "/computer/read_file", "/computer/write_file", "/computer/fetch_url"],
        "auth_required": True,
    }


@app.get("/logs", dependencies=[Depends(verify_api_key)])
async def logs():
    try:
        with open(LOGS, "r", errors="ignore") as fh:
            return {"lines": fh.read().splitlines()[-200:]}
    except Exception as exc:
        return {"lines": [], "error": str(exc)}


@app.get("/v1/models")
async def list_models(dependencies=[Depends(verify_api_key)]):
    return {"object": "list", "data": [{"id": MODEL_ALIAS, "object": "model",
            "created": int(START_TIME), "owned_by": "DuoNeural",
            "backing_gguf": OLLAMA_MODEL}]}


def thinking_disabled(body: dict) -> bool:
    """True when the caller asked for no private reasoning (Blackthorn sends chat_template_kwargs.enable_thinking=false)."""
    kwargs = body.get("chat_template_kwargs")
    if isinstance(kwargs, dict) and kwargs.get("enable_thinking") is False:
        return True
    return body.get("think") is False


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, dependencies=[Depends(verify_api_key)]):
    body = await request.json()
    messages = body.get("messages", [])
    if not any(m.get("role") == "system" for m in messages):
        messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}] + messages
    engine = "ollama"
    try:
        if os.path.exists("/tmp/engine_mode.txt"):
            engine = open("/tmp/engine_mode.txt").read().strip()
    except Exception:
        pass
    if engine == "llama":
        body["model"] = body.get("model") or MODEL_ALIAS
        body.pop("keep_alive", None)
    else:
        body["model"] = OLLAMA_MODEL
        body.setdefault("keep_alive", -1)
    body["messages"] = messages
    if thinking_disabled(body):
        # Ollama's OpenAI-compatible route ignores chat_template_kwargs and "think"; reasoning_effort="none" is honoured
        # (measured: 151 tokens / 12.4 s instead of 262 tokens / 20.6 s for the same answer, tool calls intact).
        body["reasoning_effort"] = "none"
    if body.get("stream"):
        async def event_generator():
            # Cloudflare edges cut an origin that stays silent for ~100s (HTTP 524).
            # A comment frame opens the response immediately and repeats while the model thinks.
            # Forwarder design: a reader task feeds a queue; the generator only *waits* on the
            # queue, so a 15s timeout never cancels the in-flight upstream read (the previous
            # asyncio.wait_for(agen.__anext__()) corrupted the stream on every ping).
            yield b": blackthorn-open\\n\\n"
            timeout = httpx.Timeout(connect=10.0, read=900.0, write=60.0, pool=10.0)
            q = asyncio.Queue(maxsize=64)
            DONE = object()

            async def pump(resp):
                try:
                    async for chunk in resp.aiter_bytes():
                        await q.put(chunk)
                except Exception as exc:
                    await q.put(("error", repr(exc)))
                finally:
                    await q.put(DONE)

            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", f"{OLLAMA_URL}/v1/chat/completions", json=body) as resp:
                    if resp.status_code != 200:
                        err = await resp.aread()
                        yield b"data: " + err + b"\\n\\n"
                        return
                    reader = asyncio.create_task(pump(resp))
                    try:
                        while True:
                            try:
                                item = await asyncio.wait_for(q.get(), timeout=15)
                            except asyncio.TimeoutError:
                                yield b": ping\\n\\n"
                                continue
                            if item is DONE:
                                break
                            if isinstance(item, tuple) and item and item[0] == "error":
                                yield ('data: {"error": {"message": "upstream: " + item[1][:200] + "}}\\n\\n').encode()
                                break
                            if item:
                                yield item
                    finally:
                        if not reader.done():
                            reader.cancel()
        return StreamingResponse(event_generator(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    timeout = httpx.Timeout(connect=10.0, read=900.0, write=60.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(f"{OLLAMA_URL}/v1/chat/completions", json=body)
        data = resp.json()
        if isinstance(data, dict):
            data["model"] = MODEL_ALIAS
        return JSONResponse(content=data, status_code=resp.status_code)


@app.post("/chat")
async def simple_chat(request: Request, dependencies=[Depends(verify_api_key)]):
    body = await request.json()
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": body.get("system", DEFAULT_SYSTEM_PROMPT)},
            {"role": "user", "content": body.get("prompt") or body.get("message") or ""},
        ],
        "temperature": body.get("temperature", 0.6),
        "max_tokens": body.get("max_tokens", 1024),
        "stream": False,
    }
    timeout = httpx.Timeout(connect=10.0, read=900.0, write=60.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(f"{OLLAMA_URL}/v1/chat/completions", json=payload)
        data = resp.json()
        try:
            reply = data["choices"][0]["message"]["content"]
        except Exception:
            reply = ""
        return {"model": MODEL_ALIAS, "response": reply}


# ---- Kaggle Computer Workspace API ----
import pathlib as _pathlib
import asyncio as _asyncio
COMPUTER_ROOT = "/kaggle/working/blackthorn_workspace"
_pathlib.Path(COMPUTER_ROOT).mkdir(parents=True, exist_ok=True)

def _csafe(rel: str):
    # The agent computer is a real root machine: relative paths start in the workspace,
    # absolute paths are used as given. No jail - legitimate agent work (system files,
    # installs, /kaggle/input datasets) must not be blocked. Destructive commands are
    # still refused at the tool layer.
    rel = (rel or ".").strip()
    if not rel:
        rel = "."
    path = _pathlib.Path(rel).expanduser()
    if not path.is_absolute():
        path = _pathlib.Path(COMPUTER_ROOT) / path
    return path.resolve()

@app.get("/computer/info")
async def computer_info(_=Depends(verify_api_key)):
    import platform, shutil
    gpu = ""
    try:
        gpu = __import__("subprocess").check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader"],
            text=True, timeout=5).strip()
    except Exception:
        gpu = "unavailable"
    du = shutil.disk_usage(COMPUTER_ROOT)
    try:
        import getpass, os as _os
        who = {"user": getpass.getuser(), "uid": _os.getuid(), "euid": _os.geteuid()}
    except Exception:
        who = {}
    return {
        "workspace": COMPUTER_ROOT,
        "hostname": platform.node(),
        "identity": who,
        "root_access": bool(who.get("euid") == 0),
        "gpu_roles": {"0": "model", "1": "computer"},
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpu": gpu,
        "disk_free_gb": round(du.free / 1e9, 2),
        "disk_total_gb": round(du.total / 1e9, 2),
        "environment": "kaggle-computer",
        "cuda_available": bool(gpu and "Tesla" in gpu),
    }

def _tagged_processes() -> list:
    """Live processes started through /computer/exec (tag inherited by every child, daemonised or not)."""
    found = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/environ", "rb") as fh:
                tags = fh.read().split(b"\\0")
            if b"BLACKTHORN_EXEC_TAG=1" not in tags:
                continue
            with open(f"/proc/{entry}/stat") as fh:
                state = fh.read().rsplit(")", 1)[-1].split()[0]
            if state in ("Z", "X"):
                continue
            with open(f"/proc/{entry}/cmdline", "rb") as fh:
                cmd = fh.read().replace(b"\\0", b" ").decode("utf-8", "replace").strip()[:160]
            found.append({"pid": int(entry), "state": state, "cmd": cmd})
        except OSError:
            continue  # process exited while we were reading it
    return found


@app.get("/computer/activity")
async def computer_activity(_=Depends(verify_api_key)):
    procs = _tagged_processes()
    return {"active_jobs": len(procs), "processes": procs[:50], "ts": time.time()}


@app.post("/computer/checkpoint")
async def computer_checkpoint(_=Depends(verify_api_key)):
    """Publish the workspace now. The notebook process owns the checkpoint code, so this writes a request file
    the notebook's main loop answers (within ~15 s), and waits a bounded time for the result."""
    os.makedirs(CKPT_DIR, exist_ok=True)
    req_id = str(time.time_ns())
    tmp = os.path.join(CKPT_DIR, "checkpoint.request.tmp")
    with open(tmp, "w") as fh:
        fh.write(req_id)
    os.replace(tmp, os.path.join(CKPT_DIR, "checkpoint.request"))
    deadline = time.time() + 200
    while time.time() < deadline:
        try:
            with open(os.path.join(CKPT_DIR, "checkpoint.result")) as fh:
                res = json.load(fh)
            if res.get("id") == req_id:
                return res
        except (OSError, ValueError):
            pass
        await asyncio.sleep(1)
    return {"ok": False, "error": "the notebook did not finish the checkpoint within 200 s"}



@app.post("/computer/exec")
async def computer_exec(request: Request, _=Depends(verify_api_key)):
    body = await request.json()
    cmd = str(body.get("command") or "").strip()
    if not cmd:
        raise HTTPException(status_code=400, detail="command required")
    # Long jobs are legitimate: a build or a test suite must be allowed to finish.
    timeout = min(int(body.get("timeout_seconds") or 60), 3600)
    cwd = COMPUTER_ROOT
    if body.get("cwd"):
        try:
            cwd = str(_csafe(str(body["cwd"])))
        except Exception:
            cwd = COMPUTER_ROOT
    env = dict(os.environ)
    env["HOME"] = cwd if os.path.isdir(cwd) else COMPUTER_ROOT
    env["PWD"] = cwd
    env["BLACKTHORN_WORKSPACE"] = COMPUTER_ROOT
    # GPU 0 hosts llama-server. Computer work is pinned to physical GPU 1 only.
    env["CUDA_VISIBLE_DEVICES"] = "1"
    env["BLACKTHORN_GPU_ROLE"] = "computer"
    env["BLACKTHORN_EXEC_TAG"] = "1"   # inherited by background jobs; read by /computer/activity
    def _run():
        return __import__("subprocess").run(
            cmd, shell=True, cwd=cwd, env=env,
            capture_output=True, text=True, timeout=timeout,
        )
    loop = _asyncio.get_running_loop()
    fut = loop.run_in_executor(None, _run)

    async def paced():
        # A Cloudflare edge cuts an origin silent for ~100s (HTTP 524), and commands may
        # run for minutes. Lead with whitespace (JSON parsers ignore it) and keep one
        # heartbeat while the command works; the real JSON comes as the final chunk.
        yield b" "
        while not fut.done():
            try:
                await _asyncio.wait_for(_asyncio.shield(fut), timeout=15)
            except _asyncio.TimeoutError:
                yield b" "
            except Exception:
                break
        try:
            proc = fut.result()
        except Exception as exc:
            payload = {"ok": False, "exit_code": -1,
                       "output": "error: " + type(exc).__name__ + ": " + str(exc),
                       "cwd": cwd, "workspace": COMPUTER_ROOT, "host": "kaggle-computer"}
            yield __import__("json").dumps(payload).encode("utf-8")
            return
        out = (proc.stdout or "") + (proc.stderr or "")
        if len(out) > 120000:
            out = out[:100000] + "\\n... [truncated]"
        payload = {"ok": True, "exit_code": proc.returncode, "output": out or "(no output)",
                   "cwd": cwd, "workspace": COMPUTER_ROOT, "host": "kaggle-computer"}
        yield __import__("json").dumps(payload).encode("utf-8")

    return StreamingResponse(paced(), media_type="application/json",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.post("/computer/list_files")
async def computer_list(request: Request, _=Depends(verify_api_key)):
    body = await request.json()
    path = str(body.get("path") or ".")
    try:
        target = _csafe(path)
    except HTTPException as e:
        return {"ok": False, "error": str(e.detail)}
    if not target.is_dir():
        return {"ok": False, "error": path + " is not a directory"}
    rows = []
    for entry in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))[:300]:
        try:
            size = "" if entry.is_dir() else ("  " + str(entry.stat().st_size) + " B")
        except OSError:
            size = ""
        rows.append(("DIR " if entry.is_dir() else "FILE") + " " + entry.name + size)
    return {"ok": True, "path": str(target), "listing": "\\n".join(rows) or "(empty)", "host": "kaggle-computer"}

@app.post("/computer/read_file")
async def computer_read(request: Request, _=Depends(verify_api_key)):
    body = await request.json()
    path = str(body.get("path") or "").strip()
    if not path:
        return {"ok": False, "error": "path required"}
    try:
        target = _csafe(path)
    except HTTPException as e:
        return {"ok": False, "error": str(e.detail)}
    if not target.is_file():
        return {"ok": False, "error": path + " is not a file"}
    if target.stat().st_size > 4 * 1024 * 1024:
        return {"ok": False, "error": "file larger than 4 MB"}
    data = target.read_bytes()
    try:
        content = data.decode("utf-8")
    except UnicodeDecodeError:
        return {"ok": False, "error": "binary file (" + str(len(data)) + " bytes)"}
    if len(content) > 100000:
        content = content[:100000] + "\\n... [truncated]"
    return {"ok": True, "path": str(target), "content": content, "host": "kaggle-computer"}

@app.post("/computer/write_file")
async def computer_write(request: Request, _=Depends(verify_api_key)):
    body = await request.json()
    path = str(body.get("path") or "").strip()
    content = body.get("content")
    if content is None:
        return {"ok": False, "error": "content required"}
    if not path:
        return {"ok": False, "error": "path required"}
    try:
        target = _csafe(path)
    except HTTPException as e:
        return {"ok": False, "error": str(e.detail)}
    target.parent.mkdir(parents=True, exist_ok=True)
    data = content if isinstance(content, str) else str(content)
    target.write_text(data, encoding="utf-8")
    try:
        import time as _t
        os.makedirs("/tmp/computer_state", exist_ok=True)
        with open("/tmp/computer_state/checkpoint.request", "w") as _cf:
            _cf.write(str(_t.time_ns()))
    except Exception:
        pass
    return {"ok": True, "path": str(target), "bytes": len(data.encode("utf-8")), "host": "kaggle-computer"}

@app.post("/computer/fetch_url")
async def computer_fetch(request: Request, _=Depends(verify_api_key)):
    body = await request.json()
    url = str(body.get("url") or "").strip()
    if not url.startswith(("http://", "https://")):
        return {"ok": False, "error": "url must start with http(s)://"}
    import urllib.request, re as _re
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "BlackthornKaggleComputer/1.0"})
        with urllib.request.urlopen(req, timeout=45) as resp:
            raw = resp.read(1_500_000)
            charset = resp.headers.get_content_charset() or "utf-8"
        body_text = raw.decode(charset, errors="replace")
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__ + ": " + str(exc), "host": "kaggle-computer"}
    body_text = _re.sub(r"(?is)<(script|style)[^>]*>.*?</\\1>", " ", body_text)
    body_text = _re.sub(r"(?s)<[^>]+>", " ", body_text)
    body_text = _re.sub(r"\\s+", " ", body_text).strip()
    if len(body_text) > 100000:
        body_text = body_text[:100000] + " ...[truncated]"
    return {"ok": True, "url": url, "text": body_text, "host": "kaggle-computer"}



'''


def write_gateway() -> str:
    path = "/tmp/gateway_server.py"
    with open(path, "w") as fh:
        fh.write(gateway_code())
    return path


def _computer_gateway_ok(base: str) -> bool:
    """Check authenticated computer API, not just a TCP port or generic model /health."""
    if not base:
        return False
    req = urllib.request.Request(base.rstrip("/") + "/computer/activity",
                                 headers={"Authorization": "Bearer " + API_KEY,
                                          "skip_zrok_interstitial": "true"})
    try:
        with urllib.request.urlopen(req, timeout=9) as resp:
            data = json.loads(resp.read(65536))
            return resp.status == 200 and isinstance(data.get("active_jobs"), int)
    except (urllib.error.URLError, OSError, ValueError, KeyError, TypeError):
        return False


def start_gateway():
    """Reuse an authenticated gateway if still alive; otherwise own one process."""
    if _computer_gateway_ok(f"http://127.0.0.1:{GATEWAY_PORT}"):
        return None   # existing process; do not start a competing uvicorn
    write_gateway()
    with open("/tmp/blackthorn_computer_gateway.log", "a") as output:
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "gateway_server:app", "--host", "0.0.0.0",
             "--port", str(GATEWAY_PORT), "--log-level", "warning"],
            cwd="/tmp", stdout=output, stderr=subprocess.STDOUT,
        )
    for _ in range(30):
        if _computer_gateway_ok(f"http://127.0.0.1:{GATEWAY_PORT}"):
            return proc
        if proc.poll() is not None:
            break
        time.sleep(1)
    proc.terminate()
    raise RuntimeError("Authenticated computer gateway did not come up on port 8000")


TUNNEL_HOSTNAME = (os.environ.get("BLACKTHORN_TUNNEL_HOSTNAME") or "blacktornagent.ing.ng").strip()


RELAY_CLIENT_CODE = '"""Render long-poll relay client for the Blackthorn Kaggle gateway.\n\nA dedicated thread always long-polls for work (so the hub can see the client is alive even\nwhile a response streams); worker threads execute jobs against the local gateway and stream\neach response back. Never exits: every failure is a short backoff.\n"""\nimport base64\nimport json\nimport queue\nimport sys\nimport threading\nimport time\n\nimport requests\n\nBASE = sys.argv[1].rstrip("/")\n# The published URL is the data plane (/gpu-relay/...); the control plane lives on the\n# app origin at /api/kaggle-relay/... — never ask for work through the data plane itself.\nCTRL = BASE[:-len("/gpu-relay")] if BASE.endswith("/gpu-relay") else BASE\nKEY = sys.argv[2]\nTARGET = sys.argv[3].rstrip("/")\nCHANNEL = (sys.argv[4] if len(sys.argv) > 4 else "model").strip() or "model"\nJOBS = queue.Queue()\nWORKERS = 3\n\n\ndef log(msg):\n    try:\n        with open("/tmp/relay_client.log", "a") as fh:\n            fh.write("%.1f %s\\n" % (time.time(), msg))\n    except Exception:\n        pass\n\n\ndef push_fail(job_id, text):\n    try:\n        requests.post(CTRL + "/api/kaggle-relay/push/" + job_id,\n                      headers={"X-Blackthorn-Key": KEY, "X-Relay-Status": "502",\n                               "X-Relay-Content-Type": "application/json"},\n                      data=json.dumps({"error": {"message": text}}), timeout=15)\n    except Exception:\n        pass\n\n\ndef execute(job):\n    body = job.get("body_b64") or ""\n    data = base64.b64decode(body) if body else b""\n    path = (job.get("path") or "").lstrip("/")\n    target = TARGET + "/" + path if path else TARGET + "/"\n    job_id = job["id"]\n    try:\n        upstream = requests.request(job.get("method") or "GET", target,\n                                    headers=job.get("headers") or {}, data=data,\n                                    stream=True, timeout=(10, 600))\n\n        def gen():\n            # read1 semantics: forward each piece the moment it arrives. iter_content(N)\n            # blocks until N bytes accumulate, which starves slow token streams (SSE pings\n            # never reach N) and wedges workers behind abandoned generations.\n            read1 = getattr(upstream.raw, "read1", None)\n            if read1 is not None:\n                while True:\n                    chunk = read1(65536)\n                    if not chunk:\n                        break\n                    yield chunk\n            else:\n                for chunk in upstream.iter_content(chunk_size=1):\n                    if chunk:\n                        yield chunk\n\n        requests.post(CTRL + "/api/kaggle-relay/push/" + job_id,\n                      headers={"X-Blackthorn-Key": KEY,\n                               "X-Relay-Status": str(upstream.status_code),\n                               "X-Relay-Content-Type": upstream.headers.get("Content-Type") or "application/octet-stream"},\n                      data=gen(), timeout=(10, None))\n        upstream.close()\n    except Exception as exc:\n        log("exec error " + type(exc).__name__)\n        push_fail(job_id, "relay client failed: " + type(exc).__name__)\n\n\ndef worker():\n    while True:\n        job = JOBS.get()\n        try:\n            execute(job)\n        except Exception as exc:\n            log("worker error " + type(exc).__name__)\n\n\ndef pull_loop():\n    while True:\n        try:\n            r = requests.get(CTRL + "/api/kaggle-relay/pull?channel=" + CHANNEL,\n                             headers={"X-Blackthorn-Key": KEY}, timeout=(6, 35))\n        except Exception as exc:\n            log("pull error " + type(exc).__name__)\n            time.sleep(1)\n            continue\n        if r.status_code == 204:\n            continue\n        if r.status_code != 200:\n            log("pull status " + str(r.status_code))\n            time.sleep(1)\n            continue\n        JOBS.put(r.json())\n\n\nlog("relay client started base=" + BASE + " ctrl=" + CTRL + " target=" + TARGET + " channel=" + CHANNEL)\nfor _ in range(WORKERS):\n    threading.Thread(target=worker, daemon=True).start()\npull_loop()\n'


def launch_cloudflared(cloudflared_bin: str):
    """Start the public tunnel and return (proc, public_url).

    Order (reliable first):
    1. Render long-poll relay when BLACKTHORN_RELAY_URL is set (stable through Kaggle).
    2. Inrok named share when INROK_API_KEY is set.
    3. Cloudflare named tunnel / quick tunnel as last resort.
    """
    relay_base_early = (os.environ.get("BLACKTHORN_RELAY_URL") or "").strip().rstrip("/")
    if relay_base_early:
        try:
            relay_path = "/tmp/relay_client.py"
            with open(relay_path, "w") as fh:
                fh.write(RELAY_CLIENT_CODE)
            channel = "computer" if ROLE == "computer" else "model"
            proc = subprocess.Popen(
                [sys.executable, relay_path, relay_base_early, API_KEY, f"http://127.0.0.1:{GATEWAY_PORT}", channel],
                stdout=open("/tmp/relay_client.out", "w"), stderr=subprocess.STDOUT, text=True,
            )
            public = relay_base_early + ("/computer" if channel == "computer" else "")
            # A live process is NOT a registered puller. Probe the actual gateway
            # through Render; only announce a tunnel once that host answers.
            deadline = time.time() + 55
            while time.time() < deadline:
                if proc.poll() is not None:
                    log("⚠️ relay client exited early — trying another transport")
                    break
                ready = (_computer_gateway_ok(public) if channel == "computer" else
                         _inrok_gateway_answers(public))
                if ready:
                    log(f"Stable relay: {public} (verified gateway, channel={channel})")
                    return proc, public
                time.sleep(2)
            if proc.poll() is None:
                proc.terminate()
            log("⚠️ relay did not answer its gateway probe — trying another transport")
        except Exception as exc:
            log("❌ relay launch failed: " + type(exc).__name__ + ": " + str(exc)[:200] + " — falling back")
    if inrok_enabled():
        try:
            proc, url = launch_inrok()
            if proc is not None and url:
                return proc, url
            log("⚠️ inrok did not produce a live URL — falling back to Cloudflare")
        except Exception as exc:
            log("❌ inrok launch failed: " + type(exc).__name__ + ": " + str(exc)[:200] + " — falling back")
    cf_log_path = "/tmp/cloudflared.log"
    handle = open(cf_log_path, "w")
    relay_base = (os.environ.get("BLACKTHORN_RELAY_URL") or "").strip().rstrip("/")
    token = (os.environ.get("CLOUDFLARED_TUNNEL_TOKEN") or "").strip()
    if token:
        proc = subprocess.Popen(
            [cloudflared_bin, "tunnel", "run", "--token", token,
             "--no-autoupdate", "--protocol", "http2"],
            stdout=handle, stderr=subprocess.STDOUT, text=True,
        )
        log(f"Stable tunnel: https://{TUNNEL_HOSTNAME} (named tunnel)")
        return proc, f"https://{TUNNEL_HOSTNAME}"
    if relay_base:
        relay_path = "/tmp/relay_client.py"
        with open(relay_path, "w") as fh:
            fh.write(RELAY_CLIENT_CODE)
        channel = "computer" if ROLE == "computer" else "model"
        proc = subprocess.Popen(
            [sys.executable, relay_path, relay_base, API_KEY, f"http://127.0.0.1:{GATEWAY_PORT}", channel],
            stdout=open("/tmp/relay_client.out", "w"), stderr=subprocess.STDOUT, text=True,
        )
        public = relay_base + ("/computer" if channel == "computer" else "")
        log(f"Stable relay: {public} (Render long-poll relay, channel={channel})")
        return proc, public
    proc = subprocess.Popen(
        [cloudflared_bin, "tunnel", "--url", f"http://127.0.0.1:{GATEWAY_PORT}",
         "--no-autoupdate", "--protocol", "http2"],
        stdout=handle, stderr=subprocess.STDOUT, text=True,
    )
    found = None
    deadline = time.time() + 45
    while time.time() < deadline:
        try:
            with open(cf_log_path, "r", errors="ignore") as fh:
                match = re.search(r"(https://[a-zA-Z0-9-]+\.trycloudflare\.com)", fh.read())
            if match:
                found = match.group(1)
                break
        except Exception:
            pass
        time.sleep(0.3)
    return proc, found



def _kaggle_cli_env() -> dict:
    env = dict(os.environ)
    env["KAGGLE_USERNAME"] = os.environ.get("KAGGLE_USERNAME") or KAGGLE_USERNAME
    # KGAT_ tokens use Bearer auth; KAGGLE_KEY is legacy Basic and causes 401.
    env["KAGGLE_API_TOKEN"] = os.environ.get("KAGGLE_API_TOKEN") or KAGGLE_API_TOKEN
    env.pop("KAGGLE_KEY", None)
    return env


def restore_workspace() -> bool:
    """Bring the computer workspace back from the state dataset before any work begins.

    Never deletes local data it does not understand: an existing file wins over an
    older snapshot copy, and the manifest records which generation was restored.
    """
    if not STATE_DATASET:
        log("Persistence: no COMPUTER_STATE_DATASET configured — starting with a live workspace")
        return False
    os.makedirs(STATE_DIR, exist_ok=True)
    archive_dir = os.path.join(STATE_DIR, "download")
    if os.path.isdir(archive_dir):
        shutil.rmtree(archive_dir, ignore_errors=True)
    os.makedirs(archive_dir, exist_ok=True)
    env = _kaggle_cli_env()

    def _kaggle(args):
        for cmd in (["kaggle", *args], [sys.executable, "-m", "kaggle", *args]):
            try:
                return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
            except FileNotFoundError:
                continue
        class _R:
            returncode = 127
            stderr = "kaggle CLI not installed"
            stdout = ""
        return _R()

    proc = _kaggle(["datasets", "download", "-p", archive_dir, STATE_DATASET])
    if proc.returncode != 0:
        log(f"Persistence: no snapshot to restore ({(proc.stderr or proc.stdout or '')[-160:]})")
        return False
    try:
        extracted = os.path.join(archive_dir, "unzipped")
        os.makedirs(extracted, exist_ok=True)
        def safe_extract(archive, dest):
            base = os.path.realpath(dest)
            with zipfile.ZipFile(archive) as zf:
                for item in zf.infolist():
                    name = item.filename
                    path = os.path.realpath(os.path.join(base, name))
                    if not path.startswith(base + os.sep) or (item.external_attr >> 16) & 0o170000 == 0o120000:
                        raise ValueError("snapshot contains an unsafe path")
                    if item.is_dir():
                        os.makedirs(path, exist_ok=True)
                    else:
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        with zf.open(item) as inp, open(path, "wb") as out:
                            shutil.copyfileobj(inp, out)

        for name in os.listdir(archive_dir):
            if name.endswith(".zip"):
                safe_extract(os.path.join(archive_dir, name), extracted)
        # Kaggle's --dir-mode zip puts the workspace directory in workspace.zip.
        # Unpack ONLY this known staging zip, not arbitrary user-created zip files.
        for name in ("workspace.zip", "blackthorn_workspace.zip"):
            packed = os.path.join(extracted, name)
            if os.path.isfile(packed):
                safe_extract(packed, extracted)
                os.unlink(packed)
        os.makedirs(COMPUTER_WORKSPACE, exist_ok=True)
        copied = 0
        for root, _dirs, files in os.walk(extracted):
            rel = os.path.relpath(root, extracted)
            dest = os.path.join(COMPUTER_WORKSPACE, rel) if rel != "." else COMPUTER_WORKSPACE
            os.makedirs(dest, exist_ok=True)
            for name in files:
                if name in ("dataset-metadata.json", "manifest.json") and rel == ".":
                    continue
                src_f, dst_f = os.path.join(root, name), os.path.join(dest, name)
                if not os.path.exists(dst_f):            # never overwrite newer local files
                    try:
                        os.link(src_f, dst_f)
                    except OSError:
                        shutil.copy2(src_f, dst_f)
                    copied += 1
        gen = "?"
        try:
            gen = json.load(open(os.path.join(extracted, "manifest.json"))).get("generation", "?")
        except Exception:
            pass
        log(f"✅ Persistence: restored {copied} files (generation {gen}) into {COMPUTER_WORKSPACE}")
        return True
    except Exception as exc:
        log(f"Persistence: restore failed ({type(exc).__name__}: {exc})")
    return False


# Rebuildable trees stay out of the state dataset: caches, venvs and clones are re-created by the installer
# from the registry on the next boot. Anything else larger than CHECKPOINT_MAX_FILE_BYTES is also skipped.
CHECKPOINT_EXCLUDE_DIRS = {"__pycache__", ".git", "node_modules", ".cache", "venv", "security_catalog",
                           ".ssh", ".kaggle", ".aws", "secrets", ".gnupg"}
CHECKPOINT_EXCLUDE_RELPATHS = {"tools/venv", "tools/home/.cache", "tools/home/go/pkg", "tmp"}
CHECKPOINT_SECRET_NAMES = {".env", ".env.local", "credentials", "id_rsa", "id_ed25519", "kaggle.json", "access_token"}
CHECKPOINT_MAX_FILE_BYTES = int(os.environ.get("CHECKPOINT_MAX_FILE_BYTES") or str(512 * 1024 * 1024))


def _checkpoint_ignore(directory: str, names: list) -> set:
    rel = os.path.relpath(directory, COMPUTER_WORKSPACE)
    rel = "" if rel == "." else rel.replace(os.sep, "/")
    skipped = set()
    for name in names:
        child_rel = f"{rel}/{name}" if rel else name
        full = os.path.join(directory, name)
        if name in CHECKPOINT_EXCLUDE_DIRS or child_rel in CHECKPOINT_EXCLUDE_RELPATHS:
            skipped.add(name)
        elif (os.path.islink(full) or name in CHECKPOINT_SECRET_NAMES or name.endswith((".pem", ".key", ".p12"))):
            skipped.add(name)
        elif name.endswith((".part", ".partial", ".tmp")) or name == "state.zip":
            skipped.add(name)
        elif os.path.isfile(full) and os.path.getsize(full) > CHECKPOINT_MAX_FILE_BYTES:
            skipped.add(name)
    return skipped


def _environment_inventory() -> dict:
    """Record rebuildable package names, not the container filesystem or credentials."""
    import importlib.metadata
    pip = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name", "")
        version = dist.version
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", name) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+!-]{0,100}", version):
            pip[name.lower().replace("_", "-")] = version
    try:
        result = subprocess.run(["dpkg-query", "-W", "-f", "${binary:Package}\n"],
                                capture_output=True, text=True, timeout=30)
        apt = [n for n in result.stdout.splitlines() if re.fullmatch(r"[a-z0-9][a-z0-9.+:-]{0,100}", n)] if result.returncode == 0 else []
    except (OSError, subprocess.TimeoutExpired):
        apt = []
    return {"schema": 1, "pip": pip, "apt": apt}


def restore_environment() -> dict:
    """Install only absent packages after a Kaggle session restart; never replace installed ones."""
    lock = os.path.join(COMPUTER_WORKSPACE, ".blackthorn", "environment.json")
    if not os.path.isfile(lock):
        return {"ok": True, "skipped": "no inventory"}
    try:
        with open(lock) as fh:
            old = json.load(fh)
        current = _environment_inventory()
        pip = [f"{name}=={version}" for name, version in old.get("pip", {}).items()
               if re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,100}", name)
               and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+!-]{0,100}", version)
               and name not in current["pip"]]
        apt = [name for name in old.get("apt", []) if re.fullmatch(r"[a-z0-9][a-z0-9.+:-]{0,100}", name)
               and name not in current["apt"]]
        if len(pip) > 50 or len(apt) > 50:
            return {"ok": False, "error": "more than 50 missing packages; manual review required",
                    "pip_missing": len(pip), "apt_missing": len(apt)}
        failures = []
        if apt:
            p = subprocess.run(["apt-get", "update", "-qq"], capture_output=True, timeout=300)
            if p.returncode == 0:
                p = subprocess.run(["apt-get", "install", "-y", "-qq", *apt], capture_output=True, timeout=600)
            if p.returncode:
                failures.append("apt installation failed")
        if pip:
            p = subprocess.run([sys.executable, "-m", "pip", "install", "--no-input", "--disable-pip-version-check", *pip],
                               capture_output=True, timeout=600)
            if p.returncode:
                failures.append("pip installation failed")
        return {"ok": not failures, "pip_installed": len(pip), "apt_installed": len(apt),
                "error": "; ".join(failures)}
    except (OSError, ValueError, TypeError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": type(exc).__name__}


def checkpoint_workspace(label: str = "") -> dict:
    """Publish the workspace as a new dataset version. Versions are history: nothing is deleted."""
    if not STATE_DATASET:
        return {"ok": False, "error": "no COMPUTER_STATE_DATASET configured"}
    os.makedirs(STATE_DIR, exist_ok=True)
    snap = os.path.join(STATE_DIR, "workspace")
    if os.path.isdir(snap):
        shutil.rmtree(snap, ignore_errors=True)
    shutil.copytree(COMPUTER_WORKSPACE, snap, ignore=_checkpoint_ignore, dirs_exist_ok=True)
    os.makedirs(os.path.join(snap, ".blackthorn"), exist_ok=True)
    with open(os.path.join(snap, ".blackthorn", "environment.json"), "w") as fh:
        json.dump(_environment_inventory(), fh)
    generation = int(time.time())
    files = sum(len(f) for _r, _d, f in os.walk(snap))
    json.dump({"generation": generation, "label": label[:80], "files": files, "ts": time.time()},
              open(STATE_MANIFEST, "w"))
    with open(os.path.join(snap, "dataset-metadata.json"), "w") as fh:
        json.dump({"title": "Blackthorn Computer State", "id": STATE_DATASET,
                   "isPrivate": True, "licenses": [{"name": "other"}]}, fh)
    env = _kaggle_cli_env()
    def _kaggle(args):
        for cmd in (["kaggle", *args], [sys.executable, "-m", "kaggle", *args]):
            try:
                return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
            except FileNotFoundError:
                continue
        class _R:
            returncode = 127
            stderr = "kaggle CLI not installed"
            stdout = ""
        return _R()

    proc = _kaggle(["datasets", "version", "-p", snap, "-m", f"gen {generation} {label}"[:100], "--dir-mode", "zip"])
    if proc.returncode != 0:
        proc = _kaggle(["datasets", "create", "-p", snap, "--dir-mode", "zip"])
    ok = proc.returncode == 0
    if ok:
        log(f"💾 Persistence: checkpoint generation {generation} ({files} files) published")
    else:
        log(f"Persistence: checkpoint failed: {(proc.stderr or proc.stdout or '')[-200:]}")
    return {"ok": ok, "generation": generation, "files": files, "error": "" if ok else (proc.stderr or "")[-200:]}


def _answer_checkpoint_request() -> None:
    """Gateway -> notebook hand-off for /computer/checkpoint (the gateway process cannot run this code itself)."""
    req = os.path.join(CKPT_STATE_DIR, "checkpoint.request")
    if not os.path.exists(req):
        return
    try:
        with open(req) as fh:
            req_id = fh.read().strip()
        os.remove(req)
    except OSError as exc:
        log(f"checkpoint request unreadable: {exc}")
        return
    try:
        result = checkpoint_workspace("pre-stop")
    except (OSError, ValueError, RuntimeError) as exc:
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:160]}
    result = {**result, "id": req_id}
    tmp = os.path.join(CKPT_STATE_DIR, "checkpoint.result.tmp")
    with open(tmp, "w") as fh:
        json.dump(result, fh)
    os.replace(tmp, os.path.join(CKPT_STATE_DIR, "checkpoint.result"))
    log(f"pre-stop checkpoint answered: ok={result.get('ok')}")


CKPT_STATE_DIR = "/tmp/computer_state"


TUNNEL_PROBE_EVERY_S = 60
TUNNEL_FAILS_BEFORE_RELAUNCH = 3


def _tunnel_answers(public_url: str) -> bool:
    """True when the public tunnel reaches this computer's gateway (any HTTP answer from the gateway is a yes)."""
    if not public_url:
        return False
    return _computer_gateway_ok(public_url)



EMBEDDED_SKILLS = '{"authorized-recon-basics": "---\\nname: authorized-recon-basics\\ndescription: >-\\n  Authorized reconnaissance playbook for systems you own or have written permission to test.\\n  Uses only passive checks and explicitly allowed active probes. Never targets third-party systems.\\ndomain: cybersecurity\\nsubdomain: reconnaissance\\ntags: [recon, authorized, nmap, osint]\\nversion: \\"1.0\\"\\nlicense: Apache-2.0\\n---\\n## When to Use\\nUser asks for recon, asset inventory, or port/service discovery on systems they own or authorize.\\n\\n## Prerequisites\\n- Explicit authorization in the user message or system policy\\n- Computer tools: run_command, list_files, read_file\\n- Prefer passive checks first\\n\\n## Workflow\\n1. Confirm scope: only hostnames/IPs the user authorized.\\n2. Passiveify connectivity: `ping -c 2 <host>` or `curl -I --max-time 5 <url>`.\\n3. If nmap is available and scope is authorized: `nmap -Pn -sV --top-ports 100 <host>`.\\n4. Summarize open services, versions, and risk notes. Do not exploit.\\n5. Save notes under workspace `security_notes/`.\\n\\n## Out of scope\\nExploitation, credential stuffing, scanning systems without permission.\\n\\n", "camoufox-browser-ops": "---\\nname: camoufox-browser-ops\\ndescription: >-\\n  Use the browse tool (Camoufox/Playwright when available) for navigation, DOM text, screenshots, and multi-step page interaction.\\ndomain: browser\\nsubdomain: automation\\ntags: [camoufox, browser, playwright]\\nversion: \\"1.0\\"\\nlicense: Apache-2.0\\n---\\n## Workflow\\n1. browse with action=goto and the target URL.\\n2. action=content or screenshot as needed.\\n3. On failure, retry once; fall back to computer_fetch_url or web_search.\\n4. Respect CAPTCHA limits \\u2014 report when blocked, do not loop.\\n\\n", "incident-notes": "---\\nname: incident-notes\\ndescription: >-\\n  Structure incident response notes with timeline, evidence paths, impact, and next actions.\\ndomain: cybersecurity\\nsubdomain: incident-response\\ntags: [ir, notes, timeline]\\nversion: \\"1.0\\"\\nlicense: Apache-2.0\\n---\\n## Workflow\\n1. Create workspace/incident_<date>/notes.md\\n2. Sections: Summary, Timeline, Evidence, Impact, Actions, Open questions.\\n3. Store only facts from tools; no speculation presented as fact.\\n\\n", "network-hygiene-check": "---\\nname: network-hygiene-check\\ndescription: >-\\n  Check the agent computer network posture (listening ports, outbound DNS, basic firewall state) for hygiene reporting.\\ndomain: cybersecurity\\nsubdomain: hardening\\ntags: [hygiene, ports, host]\\nversion: \\"1.0\\"\\nlicense: Apache-2.0\\n---\\n## Workflow\\n1. `ss -tulpn` or `netstat -tulpn` for listening services.\\n2. `ip a` / `hostname -I` for interfaces.\\n3. Note unexpected listeners; do not change firewall rules unless asked.\\n\\n", "secure-code-review": "---\\nname: secure-code-review\\ndescription: >-\\n  Review source code for common vulnerabilities (injection, authz, secrets, insecure crypto) without running exploits.\\ndomain: cybersecurity\\nsubdomain: application-security\\ntags: [code-review, appsec, secrets]\\nversion: \\"1.0\\"\\nlicense: Apache-2.0\\n---\\n## Workflow\\n1. List relevant files; read entry points and auth paths.\\n2. Check for injection sinks, hardcoded secrets, weak crypto, missing authz.\\n3. Report findings with file paths, severity, and remediation.\\n4. Do not run destructive tests.\\n\\n", "static-malware-triage": "---\\nname: static-malware-triage\\ndescription: >-\\n  Safe static analysis of suspicious files (hashes, strings, metadata) without executing samples.\\ndomain: cybersecurity\\nsubdomain: malware-analysis\\ntags: [malware, static, triage, hashes]\\nversion: \\"1.0\\"\\nlicense: Apache-2.0\\n---\\n## When to Use\\nUser provides a suspicious file for analysis.\\n\\n## Safety\\nNever execute the sample. Work only with hashes, file, strings, and read-only inspection.\\n\\n## Workflow\\n1. `sha256sum <file>` and `file <file>`.\\n2. `strings -n 8 <file> | head -200` (cap output).\\n3. Note packers, URLs, IPs, suspicious APIs from strings only.\\n4. Report findings with severity; recommend dynamic analysis only in an isolated lab the user controls.\\n\\n", "tavily-research": "---\\nname: tavily-research\\ndescription: >-\\n  Multi-query Tavily research with source comparison and citation hygiene.\\ndomain: research\\nsubdomain: web\\ntags: [tavily, research]\\nversion: \\"1.0\\"\\nlicense: Apache-2.0\\n---\\n## Workflow\\n1. Break the question into 1\\u20133 search queries.\\n2. Call web_search for each; keep URLs and dates.\\n3. Synthesize with explicit citations; flag conflicts.\\n\\n", "web-evidence-research": "---\\nname: web-evidence-research\\ndescription: >-\\n  Research a claim with Tavily web_search and optional browser fetch; preserve URLs and dates; compare sources.\\ndomain: research\\nsubdomain: web\\ntags: [tavily, evidence, citations]\\nversion: \\"1.0\\"\\nlicense: Apache-2.0\\n---\\n## Workflow\\n1. web_search with a precise query.\\n2. Extract titles, URLs, published dates when present.\\n3. Optionally browse 1\\u20133 top sources for detail.\\n4. Compare claims; mark uncertain items.\\n5. Never invent sources.\\n\\n"}'

def seed_skills_to_workspace() -> None:
    """Write curated skill packs into the computer workspace for skill_search/skill_load."""
    try:
        root = Path(COMPUTER_WORKSPACE) / "skills"
        root.mkdir(parents=True, exist_ok=True)
        data = json.loads(EMBEDDED_SKILLS)
        for name, body in data.items():
            d = root / name
            d.mkdir(parents=True, exist_ok=True)
            (d / "SKILL.md").write_text(body if isinstance(body, str) else str(body))
        log(f"skills seeded: {len(data)} -> {root}")
    except Exception as exc:
        log(f"skill seed note: {exc}")

def ensure_light_security_tools() -> None:
    try:
        subprocess.run(["bash", "-lc", "command -v nmap >/dev/null || apt-get install -y -qq nmap 2>/dev/null || true"], timeout=180, check=False)
        subprocess.run(["bash", "-lc", "command -v file >/dev/null || apt-get install -y -qq file 2>/dev/null || true"], timeout=120, check=False)
    except Exception as exc:
        log(f"security tools note: {exc}")

def main_computer(started: float) -> None:
    """The agent's personal computer: gateway + persistent workspace + relay. No model here."""
    os.makedirs(COMPUTER_WORKSPACE, exist_ok=True)
    notify_workspace("CHECKING_ENVIRONMENT", extra={"role": "computer"})

    try:
        urllib.request.urlopen("https://huggingface.co", timeout=8)
    except Exception as exc:
        fail("INTERNET_UNAVAILABLE", f"Kaggle internet is off ({exc}). Enable Internet in session options.")
        return

    notify_workspace("INSTALLING_DEPS", extra={"role": "computer"})
    try:
        ensure_python_deps()
        try:
            subprocess.run([sys.executable, "-m", "kaggle", "--version"], capture_output=True, timeout=30)
        except Exception:
            log("📦 Installing the kaggle CLI (checkpoint publishing needs it)")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-input", "kaggle"],
                           check=False, timeout=600)
    except Exception as exc:
        fail("BOOT_FAILED", f"dependency install failed: {exc}")
        return

    try:
        restored = restore_workspace()
        if restored:
            env_result = restore_environment()
            if not env_result.get("ok"):
                log(f"Environment recovery incomplete: {env_result.get('error')}")
    except Exception as exc:
        log(f"Persistence restore note: {type(exc).__name__}")

    notify_workspace("STARTING_GATEWAY", extra={"role": "computer"})
    try:
        gateway_proc = start_gateway()
    except (RuntimeError, OSError) as exc:
        fail("SERVICE_DOWN", f"computer gateway startup failed: {type(exc).__name__}")
        return
    try:
        cloudflared_bin = ensure_cloudflared()
    except Exception:
        cloudflared_bin = "cloudflared"

    tunnel_url = ""
    cf_proc = None
    try:
        cf_proc, tunnel_url = launch_cloudflared(cloudflared_bin)
    except Exception:
        tunnel_url = ""
    ready = bool(tunnel_url and _tunnel_answers(tunnel_url))
    if not ready:
        fail("TUNNEL_ERROR", "The computer API is not reachable through the public tunnel; recovering")
        tunnel_url = ""
    else:
        notify_workspace("TUNNEL_ONLINE", tunnel_url=tunnel_url, extra={"role": "computer"})
        notify_workspace("MODEL_READY_AND_WARMED", tunnel_url=tunnel_url,
                         extra={"role": "computer", "boot_seconds": round(time.time() - started, 1)})
        log(f"🖥️  Computer ready in {time.time() - started:.0f}s — workspace {COMPUTER_WORKSPACE}")

    loop_start = time.time()
    _last_heartbeat = 0.0
    _last_checkpoint = time.time()
    _last_tunnel_probe = 0.0
    _tunnel_fails = 0 if ready else TUNNEL_FAILS_BEFORE_RELAUNCH - 1
    _next_recover = 0.0
    _recover_attempts = 0
    while time.time() - loop_start < MAX_RUNTIME_SECONDS:
        time.sleep(15)
        now = time.time()
        # A 502 can mean the *origin* crashed. Fix the service first, and do not
        # touch the tunnel, environment, or workspace unless the public probe fails.
        local_ok = _computer_gateway_ok(f"http://127.0.0.1:{GATEWAY_PORT}")
        if not local_ok:
            notify_workspace("SERVICE_DOWN", extra={"role": "computer"})
            if now >= _next_recover:
                if gateway_proc is not None and gateway_proc.poll() is None:
                    gateway_proc.terminate()
                    try:
                        gateway_proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        gateway_proc.kill()
                try:
                    gateway_proc = start_gateway()
                except (RuntimeError, OSError) as exc:
                    _recover_attempts += 1
                    _next_recover = now + min(300, 5 * (2 ** min(_recover_attempts, 6)))
                    log(f"gateway recovery failed: {type(exc).__name__}; retry at {_next_recover:.0f}")
                    continue
                _recover_attempts = 0
                _next_recover = 0.0
                # A healthy origin behind the existing tunnel is again routable.
                # Do not leave D1 on SERVICE_DOWN with an empty URL until the next
                # 90-second heartbeat; clients must be able to resume immediately.
                if tunnel_url and _tunnel_answers(tunnel_url):
                    notify_workspace("HEARTBEAT_ONLINE", tunnel_url=tunnel_url,
                                     extra={"role": "computer", "service_recovered": True})
                    _last_heartbeat = now
            else:
                continue

        _answer_checkpoint_request()
        if now - _last_tunnel_probe >= 30:
            _last_tunnel_probe = now
            public_ok = bool(cf_proc is None or cf_proc.poll() is None) and _tunnel_answers(tunnel_url)
            if public_ok:
                _tunnel_fails = 0
                _recover_attempts = 0
            else:
                _tunnel_fails += 1
                log(f"public computer API probe failed ({_tunnel_fails}/{TUNNEL_FAILS_BEFORE_RELAUNCH})")
                if _tunnel_fails >= TUNNEL_FAILS_BEFORE_RELAUNCH:
                    notify_workspace("TUNNEL_ERROR", extra={"role": "computer"})
                    tunnel_url = ""            # never heartbeat or route through a stale URL
                    if now >= _next_recover:
                        # One owner (this notebook loop) serializes restarts. Stop the
                        # old process before launching a replacement, and verify upstream.
                        if cf_proc is not None and cf_proc.poll() is None:
                            cf_proc.terminate()
                            try:
                                cf_proc.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                cf_proc.kill()
                        cf_proc = None
                        try:
                            candidate, fresh_url = launch_cloudflared(cloudflared_bin)
                            if not fresh_url or not _tunnel_answers(fresh_url):
                                if candidate is not None and candidate.poll() is None:
                                    candidate.terminate()
                                raise RuntimeError("fresh public endpoint failed computer API probe")
                            cf_proc, tunnel_url = candidate, fresh_url
                            _tunnel_fails = 0
                            _recover_attempts = 0
                            _next_recover = 0.0
                            notify_workspace("HEARTBEAT_ONLINE", tunnel_url=tunnel_url,
                                             extra={"role": "computer", "recovered": True})
                        except (OSError, RuntimeError, ValueError) as exc:
                            _recover_attempts += 1
                            _next_recover = now + min(300, 5 * (2 ** min(_recover_attempts, 6)))
                            log(f"relay recovery failed: {type(exc).__name__}; retry at {_next_recover:.0f}")
        if now - _last_checkpoint >= CHECKPOINT_INTERVAL_S:
            _last_checkpoint = now
            try:
                result = checkpoint_workspace("auto")
                if not result.get("ok"):
                    log(f"Persistence checkpoint FAILED: {result.get('error')}")
            except Exception as exc:
                log(f"Persistence checkpoint note: {type(exc).__name__}")
        if tunnel_url and _tunnel_fails == 0 and now - _last_heartbeat >= 90:
            _last_heartbeat = now
            notify_workspace("HEARTBEAT_ONLINE", tunnel_url=tunnel_url,
                             extra={"role": "computer", "uptime": round(now - loop_start, 1)})
    try:
        last = checkpoint_workspace("session-limit")
        if not last.get("ok"):
            log(f"session limit checkpoint FAILED: {last.get('error')}")
    except Exception as exc:
        log(f"session limit checkpoint FAILED: {type(exc).__name__}")
    notify_workspace("GPU_STOPPED_SAVING_QUOTA", extra={"role": "computer"})


def main() -> None:
    started = time.time()
    boot_state = {"gpu": "", "cache_source": "", "download_skipped": False, "boot_seconds": 0}
    try:
        start_progress_reporter()
    except Exception:
        pass
    notify_workspace("CHECKING_ENVIRONMENT")

    if ROLE == "computer":
        try:
            main_computer(started)
        except Exception as exc:
            fail("BOOT_FAILED", f"computer environment failed: {type(exc).__name__}: {exc}")
        return

    try:
        gpu_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]
        ).decode().strip()
        boot_state["gpu"] = gpu_out
        log(f"✅ GPU detected: {gpu_out}")
    except Exception:
        gpu_out = "CPU / No GPU attached"
        boot_state["gpu"] = gpu_out
        fail("GPU_UNAVAILABLE", "No NVIDIA GPU detected. Set Accelerator → GPU T4 x2 in Kaggle session options.")
        return

    try:
        urllib.request.urlopen("https://huggingface.co", timeout=8)
    except Exception as exc:
        fail("INTERNET_UNAVAILABLE", f"Kaggle internet is off ({exc}). Enable Internet in session options.")
        return

    # ---- installs run in the background while the model is fetched ----
    # These three steps are independent of the model transfer, so overlapping them
    # removes ~40-70 s from every cold start. The result arrives via futures.
    install_pool = concurrent.futures.ThreadPoolExecutor(max_workers=3)
    deps_future = install_pool.submit(ensure_python_deps)
    cloudflared_future = install_pool.submit(ensure_cloudflared)
    ollama_future = install_pool.submit(ensure_ollama_serving)
    notify_workspace("INSTALLING_DEPS", extra={"gpu": gpu_out, "note": "in parallel with the model transfer"})

    # ---- 1/2: find an existing persistent copy (no download) ----
    notify_workspace("CHECKING_CACHE", extra={"gpu": gpu_out})
    gguf = ""
    source = ""
    local_blob = find_local_store_model()
    if local_blob:
        gguf, source = local_blob, "persistent-store"

    if not gguf:
        mount = find_dataset_gguf()
        if mount:
            gguf, source = mount, "dataset"

    # ---- 3: download only when nothing valid exists ----
    if not gguf:
        notify_workspace("CACHE_MISS", extra={"gpu": gpu_out,
                                              "note": "no persistent copy mounted — downloading once"})
        notify_workspace("DOWNLOADING_MODEL", extra={"gpu": gpu_out, "note": "first run only"})
        dest = os.path.join(BLIND_DISK_DIR, GGUF_BASENAME.format(quant=MODEL_QUANT))
        # Prefer permanent Kaggle cache so subsequent boots never redownload from HF.
        notify_workspace("CHECKING_CACHE", extra={"gpu": gpu_out, "note": f"cache={CACHE_DATASET_SLUG}"})
        gguf = download_from_cache_dataset(dest)
        how = "kaggle-dataset" if gguf else ""
        if not gguf:
            notify_workspace("DOWNLOADING_MODEL", extra={"gpu": gpu_out, "note": "cache miss — one-time HF download"})
            try:
                gguf, how = download_model(dest)
            except Exception as exc:
                log(f"⚠️ Hugging Face source failed: {exc}")
                notify_workspace("DOWNLOADING_MODEL", extra={"gpu": gpu_out, "note": f"HF failed: {str(exc)[:160]}"})
                gguf, how = "", ""
        if not gguf:
            log("GGUF sources failed — attempting ollama pull as last resort")
            try:
                ollama_bin_early = ensure_ollama_serving()
                if ollama_pull_model(ollama_bin_early):
                    gguf = "ollama-registry"
                    source = "ollama-pull"
                    how = "ollama-pull"
                else:
                    fail("MODEL_CORRUPT", "no model source reachable — HF GGUF, Kaggle dataset, and ollama pull all failed")
                    return
            except Exception as e:
                fail("MODEL_CORRUPT", f"no model source reachable — {e}")
                return
        expected_sha = MODEL_SHA256.get(MODEL_QUANT, "")
        if expected_sha and how in ("verified", "kaggle-dataset"):
            notify_workspace("VERIFYING_MODEL", extra={"gpu": gpu_out})
            actual = _sha256(gguf)
            if actual != expected_sha:
                # Retry once from a clean slate before declaring the model unusable.
                log(f"⚠️ sha256 mismatch ({actual[:12]}…) — re-downloading from scratch once")
                try:
                    os.remove(gguf)
                except OSError:
                    pass
                gguf, how = download_model(dest)
                if how == "verified" and _sha256(gguf) != expected_sha:
                    try:
                        os.remove(gguf)
                    except OSError:
                        pass
                    fail("MODEL_CORRUPT", "the model download failed its sha256 check twice; "
                                          "re-run the boot to retry (resume is safe)")
                    return
            _write_marker(gguf, expected_sha)
            log("✅ sha256 verified")
        else:
            log(f"⚠️ mirror source used ({how}) — size-checked only, sha256 not comparable")
        source = ("dataset-copy" if how == "kaggle-dataset" else
                  "ollama-pull" if how == "ollama-pull" else "hf-download")
        boot_state["download_skipped"] = how == "kaggle-dataset"
        if gguf != "ollama-registry" and _healthy_file(gguf, MODEL_BYTES.get(MODEL_QUANT, 0)):
            _start_cache_publish(gguf, gpu_out)
        else:
            log("⚠️ No verified GGUF file is available to publish as a model cache")
    else:
        boot_state["download_skipped"] = True
        log(f"⏭️  Download skipped — reusing {source} model")

    notify_workspace("CACHE_HIT" if boot_state["download_skipped"] else "MODEL_DOWNLOADED",
                     extra={"gpu": gpu_out, "source": source})

    # ---- collect the parallel installs (only the missing ones were touched) ----
    try:
        deps_future.result(timeout=420)
        cloudflared_bin = cloudflared_future.result(timeout=420)
        ollama_bin = ollama_future.result(timeout=420)
    except Exception as exc:
        fail("BOOT_FAILED", f"dependency install failed: {exc}")
        return
    finally:
        install_pool.shutdown(wait=False)
    log("✅ Parallel install stage complete")

    # ---- engine should already be serving (it was started during the download) ----
    notify_workspace("STARTING_OLLAMA", extra={"gpu": gpu_out})
    start_ollama_daemon(ollama_bin)   # no-op when already up

    # ---- register + load the model on the GPU ----
    notify_workspace("LOADING_MODEL", extra={"gpu": gpu_out})
    if source == "ollama-pull" or gguf == "ollama-registry":
        log("Model already in ollama store via pull — skip create")
        if not ollama_model_present():
            # ensure pull once more
            ollama_pull_model(ollama_bin)
    else:
        register_model(ollama_bin, gguf)

    boot_state["cache_source"] = source
    boot_state["boot_seconds"] = round(time.time() - started, 1)
    with open("/tmp/boot_state.json", "w") as fh:
        json.dump(boot_state, fh)
    notify_workspace("STARTING_GATEWAY", extra={"gpu": gpu_out})
    start_gateway()

    cf_proc, tunnel_url = launch_cloudflared(cloudflared_bin)
    if not tunnel_url:
        fail("TUNNEL_ERROR", "Could not bring up the Cloudflare tunnel (no URL and no configured hostname)")
        return
    notify_workspace("TUNNEL_ONLINE", tunnel_url=tunnel_url, extra={"gpu": gpu_out})

    notify_workspace("WARMING_GPU", tunnel_url=tunnel_url, extra={"gpu": gpu_out})
    warmup_ok = False
    for attempt in range(3):
        try:
            warm_req = urllib.request.Request(
                f"http://127.0.0.1:{GATEWAY_PORT}/v1/chat/completions",
                data=json.dumps({
                    "model": MODEL_ALIAS,
                    "messages": [{"role": "user", "content": "Ping. Reply with PONG."}],
                    "max_tokens": 24,
                }).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
            )
            urllib.request.urlopen(warm_req, timeout=300)
            warmup_ok = True
            break
        except Exception as exc:
            log(f"⚠️ Warmup attempt {attempt + 1} note: {exc}")
            time.sleep(5)

    boot_state["warmup_ok"] = warmup_ok
    boot_state["boot_seconds"] = round(time.time() - started, 1)
    with open("/tmp/boot_state.json", "w") as fh:
        json.dump(boot_state, fh)
    log(f"🚀 Boot complete in {boot_state['boot_seconds']}s (source={source}, download_skipped={boot_state['download_skipped']})")

    notify_workspace(
        "MODEL_READY_AND_WARMED" if warmup_ok else "MODEL_READY_COLD",
        tunnel_url=tunnel_url,
        extra={"gpu": gpu_out, "source": source, "boot_seconds": boot_state["boot_seconds"]},
    )

    # (the one-time Kaggle cache publish was started right after the download; see _start_cache_publish)

    log(f"⏳ Server loop active for up to {MAX_RUNTIME_SECONDS // 3600}h")
    loop_start = time.time()
    _last_heartbeat = 0.0
    _tunnel_fail_streak = 0
    while time.time() - loop_start < MAX_RUNTIME_SECONDS:
        time.sleep(15)
        # 1) Process dead → always restart cloudflared and publish new URL
        if cf_proc is None or cf_proc.poll() is not None:
            log("⚠️ cloudflared exited; restarting tunnel")
            cf_proc, new_url = launch_cloudflared(cloudflared_bin)
            if new_url:
                tunnel_url = new_url
                _tunnel_fail_streak = 0
                notify_workspace("MODEL_READY_AND_WARMED", tunnel_url=tunnel_url,
                                 extra={"gpu": gpu_out, "source": source, "tunnel_restart": True})
            else:
                _tunnel_fail_streak += 1
                log(f"⚠️ cloudflared restart produced no URL (streak={_tunnel_fail_streak})")
                continue
        # 2) External URL may go stale while process still runs (quick-tunnel edge case).
        #    Probe /health through the public URL; on repeated failure, force restart.
        if tunnel_url:
            try:
                req = urllib.request.Request(
                    f"{tunnel_url.rstrip('/')}/health",
                    headers={"User-Agent": "BlackthornTunnelSelfCheck/1.0",
                             **(INROK_INTERSTITIAL_HEADERS if "inrok.in" in tunnel_url else {})},
                )
                with urllib.request.urlopen(req, timeout=8) as resp:
                    body = resp.read()
                    if resp.status == 200:
                        _tunnel_fail_streak = 0
                    else:
                        _tunnel_fail_streak += 1
            except Exception as _te:
                _tunnel_fail_streak += 1
                log(f"⚠️ public tunnel health fail streak={_tunnel_fail_streak}: {type(_te).__name__}")
            if _tunnel_fail_streak >= 3:
                log("⚠️ public tunnel unhealthy 3× — checking gateway, then restarting tunnel")
                # a dead gateway looks exactly like a dead tunnel from the outside; revive it first
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{GATEWAY_PORT}/health", timeout=5)
                except Exception:
                    log("⚠️ local gateway is down — restarting it")
                    try:
                        subprocess.run(["pkill", "-f", "gateway_server"], capture_output=True, timeout=10)
                        time.sleep(1)
                        start_gateway()
                    except Exception as _ge:
                        log(f"⚠️ gateway restart note: {_ge}")
                try:
                    cf_proc.terminate()
                except Exception:
                    pass
                time.sleep(1)
                try:
                    if cf_proc is not None and cf_proc.poll() is None:
                        cf_proc.kill()
                except Exception:
                    pass
                cf_proc, new_url = launch_cloudflared(cloudflared_bin)
                if new_url:
                    tunnel_url = new_url
                    _tunnel_fail_streak = 0
                    notify_workspace(
                        "MODEL_READY_AND_WARMED",
                        tunnel_url=tunnel_url,
                        extra={"gpu": gpu_out, "source": source, "tunnel_restart": True, "reason": "self-heal"},
                    )
        # 3) Heartbeat to D1 so Render always has a fresh URL + ONLINE status
        now = time.time()
        if now - _last_heartbeat >= 90:
            _last_heartbeat = now
            notify_workspace(
                "HEARTBEAT_ONLINE",
                tunnel_url=tunnel_url,
                extra={"gpu": gpu_out, "source": source},
            )


if __name__ == "__main__":
    # The model and computer loops own their tunnel process and perform bounded
    # recovery. The older generic watchdog races those loops and references a
    # nonexistent PORT/start_tunnel; never run two independent tunnel owners.
    try:
        main()
    except Exception as exc:  # never leave the dashboard stuck on "starting"
        import traceback
        traceback.print_exc()
        fail("BOOT_FAILED", f"{type(exc).__name__}: {exc}")
        raise
