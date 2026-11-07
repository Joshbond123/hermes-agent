"""The two Kaggle kernels, as source code generated at push time.

``install_source``  runs once on a CPU session (no GPU quota spent): it downloads the Q4_K_M weights
                   and leaves them in the kernel's *output*, which Kaggle keeps permanently and
                   mounts read-only into later kernels. That is the persistent storage.
``serve_source``    runs whenever the GPU is switched on: it mounts that output, starts llama.cpp
                   against the local file, opens a tunnel, and reports to Blackthorn. It never
                   downloads weights, so a GPU that is switched off and on again reloads from
                   Kaggle's disk.

Secrets are injected here, at push time, from the server's environment — never from the repository.
The kernels are private, and the only secret they carry is a low-privilege token that can register
and heartbeat this app's GPU.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

# The Q4_K_M file the user asked for. orcarouter's own repo is gated; bartowski's requantisation of
# the same model is not, so it is the default and the gated repo is tried only when a token has
# already accepted its terms.
MIRROR_REPO = "bartowski/orcarouter_Qwen3.8-27B-Uncensored-GGUF"
MIRROR_FILE = "orcarouter_Qwen3.8-27B-Uncensored-Q4_K_M.gguf"
GATED_REPO = "orcarouter/Qwen3.8-27B-Uncensored-GGUF"
GATED_FILE = "Qwen3.8-27B-Uncensored-Q4_K_M.gguf"

FOOTER = """

def _log(msg):
    print(f"[blackthorn {time.strftime('%H:%M:%S')}] {msg}", flush=True)
"""


def _header(cfg: Dict[str, Any], doc: str) -> str:
    return (f'"""Blackthorn Kaggle kernel — generated at push time; do not edit by hand.\n\n{doc}\n"""\n'
            "import json, os, shutil, subprocess, sys, time, urllib.request, urllib.error, zipfile, glob\n\n"
            f"CFG = json.loads({json.dumps(json.dumps(cfg))})\n" + FOOTER)


def install_source(config: Dict[str, Any]) -> str:
    """Download the weights once into the kernel output (persistent, no GPU used)."""
    cfg = {"hf_token": config.get("hf_token", ""), "kaggle_user": config.get("kaggle_user", ""),
           "gated_repo": GATED_REPO, "gated_file": GATED_FILE,
           "mirror_repo": MIRROR_REPO, "mirror_file": MIRROR_FILE}
    return _header(cfg, "Permanent model install: one download, kept in the kernel output for ever.") + '''

OUT = "/kaggle/working/qwen"


def _get(url, dest, token=""):
    """Download with resume, so a dropped connection continues instead of restarting 17 GB."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    if have:
        headers["Range"] = f"bytes={have}-"
        _log(f"resuming at {have/1e9:.2f} GB")
    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as exc:
        if exc.code == 416:                      # already complete
            return dest
        raise
    total = int(resp.headers.get("content-length") or 0) + have
    mode = "ab" if have else "wb"
    done, last = have, 0.0
    with open(dest, mode) as out:
        while True:
            chunk = resp.read(16 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if time.time() - last > 20:
                last = time.time()
                _log(f"{done/1e9:.2f}/{total/1e9:.2f} GB")
    return dest


def download_model():
    os.makedirs(OUT, exist_ok=True)
    token = CFG["hf_token"]
    # 1) the gated original, but only if this token has already accepted its terms
    try:
        dest = os.path.join(OUT, CFG["gated_file"])
        if os.path.exists(dest) and os.path.getsize(dest) > 10**9:
            _log("weights already present in the output; nothing to do")
            return dest
        _get(f"https://huggingface.co/{CFG['gated_repo']}/resolve/main/{CFG['gated_file']}", dest, token)
        _log("downloaded the original orcarouter file")
        return dest
    except Exception as exc:
        _log(f"original repo unusable ({type(exc).__name__}: {str(exc)[:120]}); using the ungated mirror")
    # 2) bartowski's requantisation of the same model, ungated
    dest = os.path.join(OUT, CFG["mirror_file"])
    if not (os.path.exists(dest) and os.path.getsize(dest) > 10**9):
        _get(f"https://huggingface.co/{CFG['mirror_repo']}/resolve/main/{CFG['mirror_file']}", dest, token)
    return dest


def main():
    started = time.time()
    size_before = 0
    path = download_model()
    size = os.path.getsize(path)
    _log(f"weights ready: {path} ({size/1e9:.2f} GB) in {time.time()-started:.0f}s")
    with open("/kaggle/working/qwen-model.json", "w") as fh:
        json.dump({"file": os.path.basename(path), "bytes": size, "bytes_downloaded_now": size - size_before,
                   "finished": time.time()}, fh, indent=2)
    _log("PERSISTED: this output is mounted read-only by the GPU kernel; no future download is needed")


main()
'''


def serve_source(config: Dict[str, Any], *, sources: List[str] = ()) -> str:
    """Load the persistent weights onto the GPU and serve them; stop when told to."""
    cfg = {"backend": config.get("backend", ""), "token": config.get("token", ""),
           "model_id": config.get("model_id", "orcarouter/Qwen3.8-27B-Uncensored"),
           "weights_glob": config.get("weights_glob", "/kaggle/input/**/*Q4_K_M*.gguf"),
           "ctx": int(config.get("ctx", 16384)), "port": int(config.get("port", 8080)),
           "sources": list(sources or config.get("sources") or [])}
    return _header(cfg, "Serve Qwen3.8-27B-Uncensored from the persistent weight cache.") + '''

PORT = CFG["port"]
SERVER = None
TUNNEL = None


def find_weights():
    """The mounted, persistent copy — a download here would mean the cache was lost."""
    hits = [p for p in glob.glob(CFG["weights_glob"], recursive=True) if os.path.getsize(p) > 10**9]
    if not hits:
        raise SystemExit("no weights found in the mounted inputs; run the installer kernel first "
                         f"(looked for {CFG['weights_glob']})")
    hits.sort(key=os.path.getsize, reverse=True)
    return hits[0]


def gpu_report():
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    _log("GPU: " + out.replace("\\n", " | "))
    return len([l for l in out.splitlines() if l.strip()])


def fetch_llama_cpp():
    """The prebuilt CUDA build from the newest ``b*`` release: minutes saved on every switch-on.

    Two details that a naive fetch gets wrong: ``/releases/latest`` is a decoy tag with a single
    asset, so the release list is read instead; and the Linux builds ship as ``.tar.gz`` (a
    ``llama-…-cuda-12.8-x64`` binary archive plus its matching ``cudart-…`` runtime archive).
    """
    bin_dir = "/kaggle/working/llama"
    server = os.path.join(bin_dir, "llama-server")
    if os.path.exists(server):
        return server
    os.makedirs(bin_dir, exist_ok=True)
    releases = json.loads(urllib.request.urlopen(
        "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=8", timeout=60).read())
    import tarfile
    for rel in releases:
        names = [a["name"] for a in rel.get("assets", [])]
        for flavour in ("cuda-12.8-x64", "cuda-13.5-x64"):
            wanted = [n for n in names
                      if f"-bin-ubuntu-{flavour}.tar.gz" in n and (n.startswith("llama-") or n.startswith("cudart-"))]
            if not any(n.startswith("llama-") for n in wanted):
                continue
            for name in wanted:
                url = next(a["browser_download_url"] for a in rel["assets"] if a["name"] == name)
                tmp = os.path.join(bin_dir, name)
                _log(f"fetching {name} ({round(next(a['size'] for a in rel['assets'] if a['name'] == name)/1e6)} MB)")
                urllib.request.urlretrieve(url, tmp)
                with tarfile.open(tmp) as tf:
                    tf.extractall(bin_dir)
                os.remove(tmp)
            found = glob.glob(os.path.join(bin_dir, "**", "llama-server"), recursive=True)
            if found:
                # The build is a binary plus a pile of .so files beside it; copying the binary out of
                # its directory makes the loader fail, so it is run where it was unpacked.
                binary = found[0]
                os.chmod(binary, 0o755)
                _log(f"llama.cpp ready ({rel.get('tag_name')}, {flavour})")
                return binary
            _log(f"{rel.get('tag_name')} {flavour} produced no llama-server; trying another build")
    raise SystemExit("no usable prebuilt llama.cpp CUDA build found")


def lib_path_for(server_bin):
    """Every directory that holds a .so the server needs (the build plus its cudart siblings)."""
    dirs = []
    for root in ("/kaggle/working/llama",):
        for path, _sub, files in os.walk(root):
            if any(f.endswith(".so") or ".so." in f for f in files):
                dirs.append(path)
    dirs.append(os.path.dirname(server_bin))
    seen, out = set(), []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return ":".join(out)


def start_server(gguf, server_bin, n_gpu):
    global SERVER
    cmd = [server_bin, "-m", gguf, "--host", "127.0.0.1", "--port", str(PORT),
           "-ngl", "99", "-c", str(CFG["ctx"]), "--jinja", "--metrics",
           "--flash-attn", "on", "-ctk", "q8_0", "-ctv", "q8_0", "--parallel", "1",
           "--alias", CFG["model_id"]]
    if n_gpu > 1:                       # 17 GB of weights do not fit one 16 GB T4
        cmd += ["--tensor-split", ",".join("1" for _ in range(n_gpu)), "--split-mode", "layer"]
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = lib_path_for(server_bin) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    _log("starting: " + " ".join(cmd))
    log_file = open("/kaggle/working/llama-server.log", "w")
    SERVER = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    deadline = time.time() + 900
    while time.time() < deadline:
        if SERVER.poll() is not None:
            raise SystemExit("llama-server exited; see /kaggle/working/llama-server.log")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5) as r:
                if r.status == 200:
                    _log("server is up")
                    return
        except Exception:
            time.sleep(3)
    raise SystemExit("llama-server did not become ready in 15 minutes")


def start_tunnel():
    """A public https URL for our backend, without an account or a token."""
    global TUNNEL
    path = "/kaggle/working/cloudflared"
    if not os.path.exists(path):
        urllib.request.urlretrieve(
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", path)
        os.chmod(path, 0o755)
    TUNNEL = subprocess.Popen([path, "tunnel", "--url", f"http://127.0.0.1:{PORT}", "--no-autoupdate"],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + 180
    while time.time() < deadline:
        line = TUNNEL.stdout.readline()
        if not line:
            time.sleep(0.5)
            continue
        if "trycloudflare.com" in line:
            for token in line.replace('"', " ").split():
                if token.startswith("https://") and "trycloudflare.com" in token:
                    return token.strip()
    raise SystemExit("the tunnel did not come up")


def post(path, payload):
    req = urllib.request.Request(f"{CFG['backend']}{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {CFG['token']}"})
    with urllib.request.urlopen(req, timeout=45) as resp:
        return json.loads(resp.read() or b"{}")


def selftest():
    """Prove the model answers *and* calls a tool before announcing it as ready."""
    body = {"model": CFG["model_id"], "max_tokens": 128, "stream": False,
            "tools": [{"type": "function", "function": {
                "name": "run_command", "description": "Run a shell command",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                               "required": ["command"]}}}],
            "messages": [{"role": "user", "content": "Call run_command to print the kernel version."}]}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    calls = ((data.get("choices") or [{}])[0].get("message") or {}).get("tool_calls") or []
    _log(f"selftest tool_calls={len(calls)}")
    return bool(calls)


def busy():
    """An inference request is in flight — the kernel must not shut down."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=5) as r:
            text = r.read().decode()
    except Exception:
        return False
    for line in text.splitlines():
        if line.startswith("llamacpp:requests_processing"):
            try:
                return float(line.split()[-1]) > 0
            except ValueError:
                return False
    return False


def main():
    global SERVER
    boot = time.time()
    gguf = find_weights()
    _log(f"weights from the persistent cache: {gguf} ({os.path.getsize(gguf)/1e9:.2f} GB)")
    n_gpu = gpu_report()
    started = time.time()
    bin_path = fetch_llama_cpp()
    start_server(gguf, bin_path, n_gpu)
    load_seconds = time.time() - started
    tool_ok = selftest()
    url = start_tunnel()
    _log(f"public url: {url}")
    post("/api/gpu/register", {"base_url": url, "session": os.environ.get("KAGGLE_KERNEL_RUN_TYPE", ""),
                               "version": "1", "loaded_seconds": round(time.time() - boot, 1),
                               "weights_source": f"persistent cache ({os.path.basename(gguf)})",
                               "tool_call_ok": tool_ok})
    _log("registered with Blackthorn; heartbeating until told to stop")
    while True:
        time.sleep(30)
        if SERVER.poll() is not None:
            _log("llama-server died; ending the session so the GPU is not billed")
            return
        try:
            answer = post("/api/gpu/heartbeat", {"busy": busy(), "base_url": url})
        except Exception as exc:
            # A backend restart must not kill a working GPU; keep serving and retry.
            _log(f"heartbeat failed ({type(exc).__name__}); retrying")
            continue
        if answer.get("stop"):
            _log(f"stop ordered ({answer.get('reason','')}); shutting down the GPU")
            try:
                if TUNNEL:
                    TUNNEL.terminate()
            finally:
                SERVER.terminate()
                try:
                    SERVER.wait(timeout=30)
                except Exception:
                    SERVER.kill()
            _log("session ending — GPU released")
            return


main()
'''
