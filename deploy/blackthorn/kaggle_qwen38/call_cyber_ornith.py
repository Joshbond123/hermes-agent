#!/usr/bin/env python3
"""
Client & CLI tool for calling and managing DuoNeural/Qwen3.8-27B-Uncensored
hosted on Kaggle GPU (joshbond123) via Cloudflare Quick Tunnel + Bearer API Key.
Supports:
  --start   : Turn ON Kaggle GPU & deploy the Cloudflare Tunnel server
  --stop    : Turn OFF Kaggle GPU immediately to save weekly GPU quota
  --status  : Check Kaggle GPU quota, kernel status, and live Tunnel URL
"""

import os
import sys
import json
import argparse
import subprocess
import urllib.request
import urllib.error

DEFAULT_API_KEY = os.environ.get("CYBER_ORNITH_API_KEY", "${MODEL_API_KEY}")
TELEMETRY_TOPIC = "cyber_ornith_kaggle_cyberaiagent_8492"
KAGGLE_USERNAME = "joshbond123"
KAGGLE_SLUG = "qwen38-1-5-9b-obliterated-api-server"
KAGGLE_KERNEL_ID = f"{KAGGLE_USERNAME}/{KAGGLE_SLUG}"
KAGGLE_TOKEN = "${KAGGLE_API_TOKEN}"
BASE_DIR = os.path.expanduser("~/kaggle_cyber_ornith")
CONFIG_FILE = os.path.join(BASE_DIR, "connection_state.json")


def ensure_kaggle_auth():
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    token_path = os.path.expanduser("~/.kaggle/access_token")
    with open(token_path, "w") as f:
        f.write(KAGGLE_TOKEN)
    os.chmod(token_path, 0o600)
    os.environ["KAGGLE_API_TOKEN"] = KAGGLE_TOKEN
    os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME


def stop_gpu():
    """Immediately stop the Kaggle GPU session to save quota."""
    ensure_kaggle_auth()
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiDeleteKernelRequest

    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as kaggle:
        req = ApiDeleteKernelRequest()
        req.user_name = KAGGLE_USERNAME
        req.kernel_slug = KAGGLE_SLUG
        try:
            kaggle.kernels.kernels_api_client.delete_kernel(req)
        except Exception:
            pass

    state = {
        "tunnel_url": "",
        "api_key": DEFAULT_API_KEY,
        "status": "GPU_STOPPED_SAVING_QUOTA",
        "gpu": "OFF (0 GPU Quota Reserved)",
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(state, f, indent=2)
    print("🛑 Kaggle GPU session has been turned OFF. 0 GPU quota is currently reserved.")


def start_gpu():
    """Turn ON the Kaggle GPU session and deploy the Cloudflare Quick Tunnel."""
    ensure_kaggle_auth()
    env = os.environ.copy()
    print("🚀 Turning ON Kaggle GPU (2x Tesla T4) and deploying Qwen3.8-27B-Uncensored...")
    out = subprocess.check_output(
        ["kaggle", "kernels", "push", "-p", BASE_DIR, "--accelerator", "NvidiaTeslaT4"],
        env=env,
        text=True,
    )
    state = {
        "tunnel_url": "",
        "api_key": DEFAULT_API_KEY,
        "status": "BOOTING_KAGGLE_GPU",
        "gpu": "Starting 2x Tesla T4...",
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(state, f, indent=2)
    print(out)


def discover_tunnel_url() -> dict:
    """Discover the latest Cloudflare Tunnel URL reported by the Kaggle notebook."""
    state = {"tunnel_url": None, "status": "UNKNOWN", "api_key": DEFAULT_API_KEY}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                saved = json.load(f)
                state.update(saved)
        except Exception:
            pass

    if state.get("status") == "GPU_STOPPED_SAVING_QUOTA":
        return state

    try:
        req = urllib.request.urlopen(
            f"https://ntfy.sh/{TELEMETRY_TOPIC}/json?poll=1&since=30m", timeout=8
        )
        lines = req.read().decode("utf-8", errors="ignore").strip().splitlines()
        for line in lines:
            if not line.strip():
                continue
            msg = json.loads(line)
            raw = msg.get("message", "")
            try:
                payload = json.loads(raw)
                if isinstance(payload, dict) and "status" in payload:
                    state["status"] = payload.get("status", state["status"])
                    if payload.get("tunnel_url"):
                        state["tunnel_url"] = payload["tunnel_url"].rstrip("/")
                    if payload.get("api_key"):
                        state["api_key"] = payload["api_key"]
                    state["gpu"] = payload.get("gpu", state.get("gpu", ""))
            except Exception:
                pass
    except Exception:
        pass

    return state


def chat_with_cyber_ornith(
    prompt: str,
    tunnel_url: str = None,
    api_key: str = DEFAULT_API_KEY,
    system_prompt: str = None,
    temperature: float = 0.4,
    max_tokens: int = 1024,
) -> dict:
    """Send an OpenAI-compatible chat completion request to the Kaggle tunnel."""
    if not tunnel_url:
        discovered = discover_tunnel_url()
        tunnel_url = discovered.get("tunnel_url")
        if not tunnel_url:
            raise RuntimeError(
                "Kaggle GPU is currently OFF (or still booting). "
                "Run `python3 ~/kaggle_cyber_ornith/call_cyber_ornith.py --start` to turn it back ON."
            )

    tunnel_url = tunnel_url.rstrip("/")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": "Qwen3.8-27B-Uncensored",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    req = urllib.request.Request(
        f"{tunnel_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "CyberOrnithClient/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Qwen3.8-27B-Uncensored Kaggle GPU Controller & Client")
    parser.add_argument("prompt", nargs="?", help="Prompt to send to Qwen3.8-27B-Uncensored")
    parser.add_argument("--url", help="Override Cloudflare Tunnel URL (https://xxx.trycloudflare.com)")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="Bearer API key")
    parser.add_argument("--status", action="store_true", help="Check Kaggle GPU & tunnel status")
    parser.add_argument("--start", "--redeploy", action="store_true", dest="start", help="Turn ON Kaggle GPU & deploy tunnel")
    parser.add_argument("--stop", action="store_true", help="Turn OFF Kaggle GPU immediately to save quota")
    args = parser.parse_args()

    ensure_kaggle_auth()

    if args.stop:
        stop_gpu()
        return

    if args.start:
        start_gpu()
        return

    if args.status or not args.prompt:
        state = discover_tunnel_url()
        print("=== Kaggle GPU & Tunnel Status ===")
        print("Kaggle Account :", KAGGLE_USERNAME)
        print("State          :", state.get("status"))
        print("GPU Status     :", state.get("gpu") or "OFF")
        print("Tunnel URL     :", state.get("tunnel_url") or "OFF (Run with --start to launch)")
        print("API Key        :", state.get("api_key"))
        if not args.prompt:
            return

    resp = chat_with_cyber_ornith(args.prompt, tunnel_url=args.url, api_key=args.api_key)
    content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    print("\n=== Qwen3.8-27B-Uncensored Response ===\n")
    print(content)


if __name__ == "__main__":
    main()
