#!/usr/bin/env python3
"""Serve Qwen3.8-27B-Uncensored (Q4_K_M) over an OpenAI-compatible API on a GPU host.

Why this file exists: Blackthorn already treats the model as a *choice* (``route.model_options``), so
the only thing missing for "Qwen3.8-27B-Uncensored on a GPU" is an endpoint that speaks the same API
as the NVIDIA one. This script produces exactly that with llama.cpp, on whatever GPU host it runs on
— a Kaggle notebook with two T4s, or any single machine with enough VRAM.

Real constraints it encodes (measured, not guessed):

* ``orcarouter/Qwen3.8-27B-Uncensored-GGUF`` is a **gated** repository, so a Hugging Face token with
  access is required (``HF_TOKEN``). Without it the download is a 401, not a mystery.
* The Q4_K_M file is **16.81 GB**. A single 16 GB T4 cannot hold it, so the default is a two-GPU
  tensor split (Kaggle's "GPU T4 x2"); ``--gpus 1`` is refused for that size with a clear reason
  instead of an out-of-memory crash twenty minutes in.
* Kaggle keeps ``/kaggle/working`` between sessions but wipes everything else, so the weights live
  there and an existing file is reused — the download is skipped when it is already on disk.
* Tool calling needs the model's own chat template, so the server runs with ``--jinja``; the self-test
  performs a real tool call rather than only checking that the server answers.

Usage::

    export HF_TOKEN=hf_...                       # gated repo access
    python3 blackthorn/scripts/serve_qwen.py --selftest

It prints the two values Blackthorn needs::

    QWEN_BASE_URL=<public https url>              # or http://127.0.0.1:8080 behind your own proxy
    QWEN_MODEL=orcarouter/Qwen3.8-27B-Uncensored
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = "orcarouter/Qwen3.8-27B-Uncensored-GGUF"
FILE = "Qwen3.8-27B-Uncensored-Q4_K_M.gguf"
MODEL_ID = "orcarouter/Qwen3.8-27B-Uncensored"
SIZE_GB = 16.81
# Weights + a KV cache for 8k tokens + graph buffers. Below this the server will OOM under load.
NEEDED_VRAM_GB = 20.0


def weights_dir() -> Path:
    """Where the weights live, preferring persistent storage when the host has it."""
    for candidate in (os.environ.get("QWEN_MODEL_DIR", ""), "/kaggle/working/qwen-model",
                      str(Path.home() / ".cache" / "blackthorn" / "models")):
        if candidate and (Path(candidate).exists() or candidate.startswith("/kaggle/")):
            return Path(candidate)
    return Path.home() / ".cache" / "blackthorn" / "models"


def gpu_report() -> list[dict]:
    """Name and VRAM of each visible NVIDIA GPU (empty when there is no ``nvidia-smi``)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        name, _, total = line.rpartition(",")
        try:
            gpus.append({"name": name.strip(), "vram_gb": round(float(total) / 1024, 1)})
        except ValueError:
            continue
    return gpus


def build_command(gguf: Path, gpus: list[dict], *, port: int = 8080, ctx: int = 8192,
                  threads: int = 4) -> list[str]:
    """The llama.cpp server command for this machine.

    Pure so it can be tested without a GPU: the split and the layer count are the parts that decide
    whether the model loads at all.
    """
    total_vram = sum(g["vram_gb"] for g in gpus) or 0.0
    if total_vram and total_vram < NEEDED_VRAM_GB:
        raise SystemExit(
            f"This machine has {total_vram:.1f} GB of GPU memory; {SIZE_GB:.1f} GB of weights plus the KV "
            f"cache need about {NEEDED_VRAM_GB:.0f} GB. Use a two-T4 host or a smaller quantisation.")
    binary = shutil.which("llama-server") or shutil.which("server") or "llama-server"
    cmd = [binary, "-m", str(gguf), "--host", "127.0.0.1", "--port", str(port),
           "-ngl", "99",                     # every layer on the GPU: this is a GPU deployment
           "-c", str(ctx), "-t", str(threads),
           "--jinja",                        # the model's own chat template, which is what tools need
           "--metrics"]
    if len(gpus) > 1:
        # Even split across identical cards; llama.cpp needs one weight per GPU it can see.
        cmd += ["--tensor-split", ",".join("1" for _ in gpus)]
        cmd += ["--split-mode", "layer"]
    return cmd


def download(target: Path) -> Path:
    """Fetch the GGUF once; reuse it for ever afterwards (a re-download is gigabytes and minutes)."""
    gguf = target / FILE
    if gguf.exists() and gguf.stat().st_size > 1_000_000_000:
        print(f"[qwen] reusing {gguf} ({gguf.stat().st_size / 1e9:.2f} GB)")
        return gguf
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if not token:
        raise SystemExit(
            f"{REPO} is a gated repository: set HF_TOKEN to a Hugging Face token that has accepted its "
            "terms (https://huggingface.co/" + REPO + ").")
    target.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{REPO}/resolve/main/{FILE}"
    print(f"[qwen] downloading {FILE} ({SIZE_GB} GB) to {gguf}")
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(request, timeout=120) as resp, open(gguf, "wb") as out:
            total, done = int(resp.headers.get("content-length") or 0), 0
            last = 0.0
            while True:
                chunk = resp.read(8 << 20)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if time.time() - last > 10:
                    last = time.time()
                    print(f"[qwen]   {done / 1e9:.1f}/{total / 1e9:.1f} GB", flush=True)
    except urllib.error.HTTPError as exc:
        gguf.unlink(missing_ok=True)
        if exc.code in (401, 403):
            raise SystemExit("Hugging Face refused the download: the token lacks access to this gated "
                             "repository. Accept its terms and retry with that token.") from exc
        raise
    print(f"[qwen] ready: {gguf}")
    return gguf


def wait_ready(port: int, timeout: float = 900.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(3)
    return False


def selftest(port: int, model: str = MODEL_ID) -> int:
    """Prove the model answers *and* can call a tool — the two things Blackthorn needs."""
    base = f"http://127.0.0.1:{port}/v1/chat/completions"
    tool = {"type": "function", "function": {
        "name": "run_command", "description": "Run a shell command on the computer.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}
    body = {"model": model, "stream": False, "max_tokens": 256, "tools": [tool],
            "messages": [{"role": "user", "content": "Use the run_command tool to print the kernel version. "
                                                     "Call the tool; do not answer in prose."}]}
    request = urllib.request.Request(base, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as resp:
        data = json.loads(resp.read())
    message = (data.get("choices") or [{}])[0].get("message") or {}
    calls = message.get("tool_calls") or []
    if not calls:
        print("[qwen] the model answered but did not call the tool:", json.dumps(message)[:400])
        return 1
    call = calls[0].get("function") or {}
    print(f"[qwen] tool call OK: {call.get('name')}({call.get('arguments')})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=int(os.environ.get("QWEN_PORT", "8080")))
    parser.add_argument("--ctx", type=int, default=int(os.environ.get("QWEN_CTX", "8192")))
    parser.add_argument("--threads", type=int, default=int(os.environ.get("QWEN_THREADS", "4")))
    parser.add_argument("--selftest", action="store_true", help="after starting, prove tools work")
    parser.add_argument("--print-command", action="store_true", help="show the command and exit")
    args = parser.parse_args(argv)

    gpus = gpu_report()
    print(f"[qwen] GPUs: {gpus or 'none detected'}")
    gguf = weights_dir() / FILE
    cmd = build_command(gguf, gpus, port=args.port, ctx=args.ctx, threads=args.threads)
    if args.print_command:
        print(" ".join(cmd))
        return 0
    gguf = download(weights_dir())
    print("[qwen] starting:", " ".join(cmd), flush=True)
    server = subprocess.Popen(cmd)
    try:
        if not wait_ready(args.port):
            print("[qwen] the server did not become ready", file=sys.stderr)
            return 1
        print(f"[qwen] serving on http://127.0.0.1:{args.port}")
        print(f"[qwen] configure Blackthorn with:\n  QWEN_BASE_URL=<public url for port {args.port}>\n  QWEN_MODEL={MODEL_ID}")
        if args.selftest:
            return selftest(args.port)
        server.wait()
        return 0
    finally:
        if server.poll() is None:
            server.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
