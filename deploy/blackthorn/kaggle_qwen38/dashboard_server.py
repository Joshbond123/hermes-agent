#!/usr/bin/env python3
"""
Live Control & Chat Dashboard for Qwen3.8-27B-Uncensored on Kaggle GPU.
Binds to 0.0.0.0:8080 and proxies all browser requests via relative paths.
"""

import os
import json
import time
import subprocess
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler

PORT = 8080
BASE_DIR = "/home/user/kaggle_cyber_ornith"
CONFIG_FILE = os.path.join(BASE_DIR, "connection_state.json")
TELEMETRY_TOPIC = "cyber_ornith_kaggle_joshbond123_8492"
KAGGLE_KERNEL_ID = "joshbond123/cyber-ornith-1-5-9b-obliterated-api-server"
DEFAULT_API_KEY = "${MODEL_API_KEY}"

KAGGLE_ENV = os.environ.copy()
KAGGLE_ENV["KAGGLE_API_TOKEN"] = "${KAGGLE_API_TOKEN}"
KAGGLE_ENV["KAGGLE_USERNAME"] = "joshbond123"


def get_connection_state() -> dict:
    state = {
        "tunnel_url": "",
        "api_key": DEFAULT_API_KEY,
        "status": "WAITING_FOR_GPU_SESSION",
        "gpu": "",
        "kaggle_kernel": KAGGLE_KERNEL_ID,
        "kaggle_notebook_url": f"https://www.kaggle.com/code/{KAGGLE_KERNEL_ID}",
        "kaggle_edit_url": f"https://www.kaggle.com/code/{KAGGLE_KERNEL_ID}/edit",
        "kaggle_worker_status": "UNKNOWN",
        "tunnel_healthy": False,
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                state.update(json.load(f))
        except Exception:
            pass

    if state.get('status') == 'GPU_STOPPED_SAVING_QUOTA':
        state['kaggle_worker_status'] = 'OFF (Quota Saved)'
        return state
    # Poll ntfy.sh telemetry channel for live updates from Kaggle
    try:
        req = urllib.request.urlopen(
            f"https://ntfy.sh/{TELEMETRY_TOPIC}/json?poll=1&since=12h", timeout=5
        )
        lines = req.read().decode("utf-8", errors="ignore").strip().splitlines()
        for line in lines:
            if not line.strip():
                continue
            msg = json.loads(line)
            try:
                payload = json.loads(msg.get("message", ""))
                if isinstance(payload, dict) and "status" in payload:
                    state["status"] = payload.get("status", state["status"])
                    if payload.get("tunnel_url"):
                        state["tunnel_url"] = payload["tunnel_url"].rstrip("/")
                    if payload.get("api_key"):
                        state["api_key"] = payload["api_key"]
                    if payload.get("gpu"):
                        state["gpu"] = payload["gpu"]
            except Exception:
                pass
    except Exception:
        pass

    # Check Kaggle worker status via CLI
    try:
        out = subprocess.check_output(
            ["kaggle", "kernels", "status", KAGGLE_KERNEL_ID],
            env=KAGGLE_ENV,
            text=True,
            timeout=8,
        ).strip()
        state["kaggle_worker_status"] = out
    except Exception as e:
        state["kaggle_worker_status"] = f"Status check: {e}"

    # If we have a tunnel_url, ping /health
    if state.get("tunnel_url"):
        try:
            h_req = urllib.request.Request(
                f"{state['tunnel_url'].rstrip('/')}/health",
                headers={"User-Agent": "CyberOrnithDashboard/1.0"},
            )
            with urllib.request.urlopen(h_req, timeout=5) as resp:
                h_data = json.loads(resp.read().decode("utf-8"))
                state["tunnel_healthy"] = h_data.get("status") == "online"
                state["gpu"] = h_data.get("gpu", state.get("gpu", ""))
                state["status"] = "ONLINE_READY"
        except Exception:
            state["tunnel_healthy"] = False

    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(
                {
                    "tunnel_url": state.get("tunnel_url", ""),
                    "api_key": state.get("api_key", DEFAULT_API_KEY),
                    "status": state.get("status", ""),
                    "gpu": state.get("gpu", ""),
                },
                f,
                indent=2,
            )
    except Exception:
        pass

    return state


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Qwen3.8-27B-Uncensored — Kaggle GPU Tunnel Console</title>
<style>
  :root {
    --bg: #0b0f17;
    --panel: #131b2e;
    --panel-alt: #19233c;
    --border: #263554;
    --accent: #10b981;
    --accent-cyan: #06b6d4;
    --warn: #f59e0b;
    --danger: #ef4444;
    --text: #e2e8f0;
    --muted: #94a3b8;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, monospace;
    line-height: 1.5;
  }
  header {
    background: linear-gradient(90deg, #0f172a, #131b2e);
    border-bottom: 1px solid var(--border);
    padding: 16px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .badge {
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .badge-online { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }
  .badge-wait { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; }
  .container {
    max-width: 1240px;
    margin: 0 auto;
    padding: 20px;
    display: grid;
    grid-template-columns: 410px 1fr;
    gap: 20px;
  }
  @media (max-width: 960px) {
    .container { grid-template-columns: 1fr; }
  }
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 18px;
  }
  .card h2 {
    margin: 0 0 12px 0;
    font-size: 15px;
    color: #38bdf8;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  label {
    display: block;
    font-size: 12px;
    color: var(--muted);
    margin-bottom: 4px;
    margin-top: 10px;
  }
  input, textarea, select {
    width: 100%;
    background: #090d16;
    border: 1px solid var(--border);
    color: var(--text);
    padding: 9px 11px;
    border-radius: 8px;
    font-family: monospace;
    font-size: 13px;
  }
  input:focus, textarea:focus {
    outline: none;
    border-color: var(--accent-cyan);
  }
  .btn-row {
    display: flex;
    gap: 8px;
    margin-top: 12px;
    flex-wrap: wrap;
  }
  button, .btn-link {
    background: var(--accent-cyan);
    color: #04131a;
    border: none;
    padding: 9px 14px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 13px;
    cursor: pointer;
    text-decoration: none;
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }
  button.secondary, .btn-link.secondary {
    background: var(--panel-alt);
    color: var(--text);
    border: 1px solid var(--border);
  }
  button.emerald {
    background: var(--accent);
    color: #03150e;
  }
  .steps {
    font-size: 13px;
    color: #cbd5e1;
    padding-left: 18px;
    margin: 8px 0;
  }
  .steps li { margin-bottom: 6px; }
  .code-box {
    background: #070a12;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 12px;
    font-family: monospace;
    font-size: 12px;
    overflow-x: auto;
    white-space: pre-wrap;
    color: #a7f3d0;
    margin-top: 8px;
  }
  .chat-box {
    display: flex;
    flex-direction: column;
    height: 520px;
  }
  .messages {
    flex: 1;
    overflow-y: auto;
    background: #080c14;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px;
    margin-bottom: 12px;
  }
  .msg {
    margin-bottom: 14px;
    padding: 10px 14px;
    border-radius: 8px;
    font-size: 13.5px;
    white-space: pre-wrap;
  }
  .msg-user {
    background: #1e293b;
    border-left: 3px solid #38bdf8;
  }
  .msg-assistant {
    background: #111c2d;
    border-left: 3px solid #10b981;
  }
  .think-block {
    background: rgba(6, 182, 212, 0.08);
    border: 1px dashed #0e7490;
    border-radius: 6px;
    padding: 8px 10px;
    margin-bottom: 8px;
    color: #94a3b8;
    font-size: 12px;
  }
  .preset-chips {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    margin-bottom: 10px;
  }
  .chip {
    background: #172136;
    border: 1px solid #2b3d63;
    color: #93c5fd;
    font-size: 11.5px;
    padding: 5px 10px;
    border-radius: 999px;
    cursor: pointer;
  }
  .chip:hover { background: #1e2d4a; }
  .kv {
    display: flex;
    justify-content: space-between;
    font-size: 12.5px;
    padding: 5px 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
  }
  .kv span:first-child { color: var(--muted); }
</style>
</head>
<body>
<header>
  <div class="brand">
    <div style="font-size:24px;">🛡️</div>
    <div>
      <div style="font-weight:800;font-size:17px;">Qwen3.8-27B-Uncensored — Kaggle GPU Gateway</div>
      <div style="font-size:12px;color:var(--muted);">Connected to Kaggle Account: <b>joshbond123</b> • Cloudflare Quick Tunnel + Bearer API Key</div>
    </div>
  </div>
  <div id="statusBadge" class="badge badge-wait">Checking Kaggle...</div>
</header>

<div class="container">
  <!-- LEFT COLUMN: KAGGLE + TUNNEL CONFIG -->
  <div>
    <div class="card">
      <h2>
        <span>1. Kaggle GPU Deployment</span>
        <button class="secondary" onclick="refreshStatus()" style="padding:4px 10px;font-size:11px;">↻ Refresh</button>
      </h2>
      <div class="kv"><span>Kaggle Account:</span><b>joshbond123</b></div>
      <div class="kv"><span>Kernel Slug:</span><b>cyber-ornith-1-5-9b-obliterated-api-server</b></div>
      <div class="kv"><span>Kaggle Worker Status:</span><b id="kaggleWorker">Checking...</b></div>
      <div class="kv"><span>Telemetry Status:</span><b id="telemetryState">Waiting...</b></div>
      <div class="kv"><span>Detected GPU:</span><b id="gpuInfo">Pending GPU session</b></div>

      <div style="margin-top:12px;padding:10px;background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.4);border-radius:8px;font-size:12.5px;">
        <b style="color:#fbbf24;">⚡ Activate GPU & Internet on Kaggle (One-Time Step):</b>
        <ol class="steps">
          <li>Ensure your Kaggle account is <b>Phone Verified</b> at <a href="https://www.kaggle.com/settings" target="_blank" style="color:#38bdf8;">kaggle.com/settings</a> (required by Kaggle for GPU & Internet).</li>
          <li>Click <b>Open Notebook on Kaggle</b> below.</li>
          <li>In the right sidebar (<b>Session options</b>), set <b>Accelerator → GPU T4 x2</b> and <b>Internet → ON</b>.</li>
          <li>Click <b>Run All</b> — the Cloudflare Tunnel URL will auto-sync here within ~60 seconds!</li>
        </ol>
      </div>

      <div class="btn-row">
        <button class="emerald" onclick="redeployKernel()">🚀 Turn ON GPU & Launch Tunnel</button>
        <button style="background:#ef4444;color:#fff;" onclick="stopGpu()">🛑 Turn OFF GPU (Save Quota)</button>
      </div>
    </div>

    <div class="card">
      <h2>2. Tunnel URL & API Key</h2>
      <label>Cloudflare Quick Tunnel URL (Auto-detected or Paste manually):</label>
      <input id="tunnelUrlInput" type="text" placeholder="https://xxxx-xxxx.trycloudflare.com" />

      <label>Bearer API Key:</label>
      <input id="apiKeyInput" type="text" value="${MODEL_API_KEY}" />

      <div class="btn-row">
        <button onclick="saveConfig()">💾 Save & Test Connection</button>
      </div>
      <div id="saveMsg" style="font-size:12px;margin-top:8px;color:#34d399;"></div>
    </div>

    <div class="card">
      <h2>3. Ready-to-Use API Snippets</h2>
      <label>cURL (OpenAI-Compatible Endpoint):</label>
      <div id="curlSnippet" class="code-box"></div>
      <label>Python (OpenAI SDK):</label>
      <div id="pySnippet" class="code-box"></div>
    </div>
  </div>

  <!-- RIGHT COLUMN: INTERACTIVE CHAT & REASONING CONSOLE -->
  <div>
    <div class="card">
      <h2>
        <span>🧠 Qwen3.8-27B-Uncensored Live Console</span>
        <span style="font-size:11px;color:var(--muted);font-weight:normal;">System-2 &lt;think&gt; Reasoning Enabled</span>
      </h2>

      <div class="preset-chips">
        <span class="chip" onclick="usePreset('Who are you? Describe your architecture, lineage, and specialized security capabilities.')">🪪 Identity & Pedigree</span>
        <span class="chip" onclick="usePreset('Write a compact single-line Linux CLI pipeline to find all SUID binaries, stat their permissions, compute SHA256 hashes, and sort by file size.')">🐧 Linux SUID Audit CLI</span>
        <span class="chip" onclick="usePreset('Audit a PHP login endpoint that concatenates $_POST[\"user\"] into a raw SQL query. Explain the exploit vector and provide a parameterized PDO patch.')">🛡️ SQLi Triage & Patch</span>
        <span class="chip" onclick="usePreset('Write a Python script using struct.unpack to parse the first 34 bytes of an ELF64 binary header and validate the magic bytes.')">🔬 ELF Binary Forensics</span>
      </div>

      <div class="chat-box">
        <div id="messages" class="messages">
          <div class="msg msg-assistant">
            <b>Qwen3.8-27B-Uncensored Console Ready.</b><br/>
            Notebook Version 4 has been pushed directly to your Kaggle account (<code>joshbond123/cyber-ornith-1-5-9b-obliterated-api-server</code>).<br/><br/>
            Once you click <b>Run All</b> with <b>Internet: ON</b> and <b>Accelerator: GPU T4 x2</b> on Kaggle, this console will automatically detect the live <code>*.trycloudflare.com</code> tunnel URL (or you can paste it on the left) and send prompts straight to your Kaggle GPU!
          </div>
        </div>

        <div style="display:flex;gap:8px;">
          <textarea id="promptInput" rows="3" placeholder="Ask Qwen3.8-27B-Uncensored anything (CLI pipelines, vulnerability triage, code synthesis)..."></textarea>
          <button class="emerald" onclick="sendChat()" id="sendBtn" style="padding:0 22px;">Send ➤</button>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
let currentState = {};

function updateSnippets() {
  const url = (document.getElementById('tunnelUrlInput').value || 'https://YOUR-TUNNEL.trycloudflare.com').replace(/\/$/, '');
  const key = document.getElementById('apiKeyInput').value || '${MODEL_API_KEY}';

  document.getElementById('curlSnippet').textContent =
`curl -s ${url}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer ${key}" \\
  -d '{
    "model": "Qwen3.8-27B-Uncensored",
    "messages": [{"role": "user", "content": "Audit SUID binaries on Linux"}],
    "temperature": 0.4
  }'`;

  document.getElementById('pySnippet').textContent =
`from openai import OpenAI

client = OpenAI(
    base_url="${url}/v1",
    api_key="${key}"
)

resp = client.chat.completions.create(
    model="Qwen3.8-27B-Uncensored",
    messages=[{"role": "user", "content": "Audit SUID binaries on Linux"}],
    temperature=0.4
)
print(resp.choices[0].message.content)`;
}

async function refreshStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    currentState = data;

    document.getElementById('kaggleWorker').textContent = data.kaggle_worker_status.replace('joshbond123/cyber-ornith-1-5-9b-obliterated-api-server has status ', '');
    document.getElementById('telemetryState').textContent = data.status || 'Waiting...';
    document.getElementById('gpuInfo').textContent = data.gpu || 'Waiting for Kaggle GPU session';

    const urlInput = document.getElementById('tunnelUrlInput');
    if (data.tunnel_url && !urlInput.value) {
      urlInput.value = data.tunnel_url;
    }
    if (data.api_key && !document.getElementById('apiKeyInput').value) {
      document.getElementById('apiKeyInput').value = data.api_key;
    }

    const badge = document.getElementById('statusBadge');
    if (data.tunnel_healthy) {
      badge.className = 'badge badge-online';
      badge.textContent = '🟢 Tunnel Online & Ready';
    } else if (data.tunnel_url) {
      badge.className = 'badge badge-wait';
      badge.textContent = '🟡 Tunnel URL Detected (Warming Up)';
    } else {
      badge.className = 'badge badge-wait';
      badge.textContent = '⏳ Waiting for Kaggle GPU Run';
    }
    updateSnippets();
  } catch (e) {
    console.error(e);
  }
}

async function saveConfig() {
  const tunnel_url = document.getElementById('tunnelUrlInput').value.trim();
  const api_key = document.getElementById('apiKeyInput').value.trim();
  const saveMsg = document.getElementById('saveMsg');
  saveMsg.textContent = 'Testing & saving connection...';
  const res = await fetch('/api/save-config', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({tunnel_url, api_key})
  });
  const data = await res.json();
  saveMsg.textContent = data.message || 'Saved!';
  updateSnippets();
  refreshStatus();
}

async function stopGpu() {
  const saveMsg = document.getElementById('saveMsg');
  saveMsg.textContent = 'Turning OFF Kaggle GPU session...';
  const res = await fetch('/api/stop-gpu', {method: 'POST'});
  const data = await res.json();
  document.getElementById('tunnelUrlInput').value = '';
  saveMsg.textContent = data.message || 'GPU Stopped!';
  refreshStatus();
}

async function redeployKernel() {
  const saveMsg = document.getElementById('saveMsg');
  saveMsg.textContent = 'Pushing notebook to Kaggle via API...';
  const res = await fetch('/api/redeploy', {method: 'POST'});
  const data = await res.json();
  saveMsg.textContent = data.message || 'Kernel pushed!';
  refreshStatus();
}

function usePreset(text) {
  document.getElementById('promptInput').value = text;
}

function formatAssistantMessage(text) {
  const thinkMatch = text.match(/<think>([\s\S]*?)<\/think>/i);
  if (thinkMatch) {
    const thinkContent = thinkMatch[1].trim();
    const restContent = text.replace(/<think>[\s\S]*?<\/think>/i, '').trim();
    const div = document.createElement('div');
    const thinkDiv = document.createElement('div');
    thinkDiv.className = 'think-block';
    thinkDiv.textContent = '💭 System-2 Reasoning (<think>):\n' + thinkContent;
    const mainDiv = document.createElement('div');
    mainDiv.textContent = restContent;
    div.appendChild(thinkDiv);
    div.appendChild(mainDiv);
    return div;
  }
  const div = document.createElement('div');
  div.textContent = text;
  return div;
}

async function sendChat() {
  const input = document.getElementById('promptInput');
  const prompt = input.value.trim();
  if (!prompt) return;

  const msgs = document.getElementById('messages');
  const userDiv = document.createElement('div');
  userDiv.className = 'msg msg-user';
  userDiv.textContent = 'You: ' + prompt;
  msgs.appendChild(userDiv);
  input.value = '';
  msgs.scrollTop = msgs.scrollHeight;

  const btn = document.getElementById('sendBtn');
  btn.disabled = true;
  btn.textContent = 'Thinking...';

  const tunnel_url = document.getElementById('tunnelUrlInput').value.trim();
  const api_key = document.getElementById('apiKeyInput').value.trim();

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({prompt, tunnel_url, api_key})
    });
    const data = await res.json();
    const botDiv = document.createElement('div');
    botDiv.className = 'msg msg-assistant';
    if (data.error) {
      botDiv.style.borderLeftColor = '#ef4444';
      botDiv.textContent = '⚠️ ' + data.error;
    } else {
      botDiv.appendChild(formatAssistantMessage(data.reply || JSON.stringify(data)));
    }
    msgs.appendChild(botDiv);
  } catch (err) {
    const errDiv = document.createElement('div');
    errDiv.className = 'msg msg-assistant';
    errDiv.style.borderLeftColor = '#ef4444';
    errDiv.textContent = '⚠️ Request error: ' + err.message;
    msgs.appendChild(errDiv);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Send ➤';
    msgs.scrollTop = msgs.scrollHeight;
  }
}

document.getElementById('tunnelUrlInput').addEventListener('input', updateSnippets);
document.getElementById('apiKeyInput').addEventListener('input', updateSnippets);
refreshStatus();
setInterval(refreshStatus, 12000);
</script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/status"):
            state = get_connection_state()
            self._send_json(state)
            return

        body = HTML_PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        try:
            body = json.loads(raw)
        except Exception:
            body = {}

        if self.path.startswith("/api/save-config"):
            tunnel_url = (body.get("tunnel_url") or "").strip().rstrip("/")
            api_key = (body.get("api_key") or DEFAULT_API_KEY).strip()
            saved = {"tunnel_url": tunnel_url, "api_key": api_key, "status": "CONFIG_SAVED"}
            with open(CONFIG_FILE, "w") as f:
                json.dump(saved, f, indent=2)
            msg = "Configuration saved."
            if tunnel_url:
                try:
                    req = urllib.request.Request(
                        f"{tunnel_url}/health",
                        headers={"User-Agent": "CyberOrnithDashboard/1.0"},
                    )
                    with urllib.request.urlopen(req, timeout=6) as resp:
                        h = json.loads(resp.read().decode("utf-8"))
                        msg = f"✅ Connected to {h.get('model')} on {h.get('gpu')}!"
                except Exception as e:
                    msg = f"Saved URL, but /health check returned: {e}"
            self._send_json({"ok": True, "message": msg})
            return

        if self.path.startswith("/api/stop-gpu"):
            try:
                out = subprocess.check_output(
                    ["python3", os.path.join(BASE_DIR, "call_cyber_ornith.py"), "--stop"],
                    env=KAGGLE_ENV,
                    text=True,
                    timeout=20,
                ).strip()
                self._send_json({"ok": True, "message": out})
            except Exception as e:
                self._send_json({"ok": False, "message": f"Stop error: {e}"}, 500)
            return

        if self.path.startswith("/api/redeploy"):
            try:
                out = subprocess.check_output(
                    ["python3", os.path.join(BASE_DIR, "call_cyber_ornith.py"), "--start"],
                    env=KAGGLE_ENV,
                    text=True,
                    timeout=25,
                ).strip()
                self._send_json({"ok": True, "message": out})
            except Exception as e:
                self._send_json({"ok": False, "message": f"Redeploy error: {e}"}, 500)
            return

        if self.path.startswith("/api/chat"):
            state = get_connection_state()
            tunnel_url = (body.get("tunnel_url") or state.get("tunnel_url") or "").strip().rstrip("/")
            api_key = (body.get("api_key") or state.get("api_key") or DEFAULT_API_KEY).strip()
            prompt = body.get("prompt", "")

            if not tunnel_url:
                self._send_json(
                    {
                        "error": (
                            "No active Cloudflare Tunnel URL yet! Please open "
                            "https://www.kaggle.com/code/joshbond123/cyber-ornith-1-5-9b-obliterated-api-server/edit "
                            "on Kaggle, set Accelerator -> GPU T4 x2 and Internet -> ON, and click 'Run All'."
                        )
                    },
                    400,
                )
                return

            payload = {
                "model": "Qwen3.8-27B-Uncensored",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.4,
                "max_tokens": 1024,
                "stream": False,
            }
            try:
                req = urllib.request.Request(
                    f"{tunnel_url}/v1/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                        "User-Agent": "CyberOrnithDashboard/1.0",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=180) as resp:
                    r_data = json.loads(resp.read().decode("utf-8"))
                    reply = (
                        r_data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                    self._send_json({"reply": reply, "raw": r_data})
            except urllib.error.HTTPError as he:
                err_body = he.read().decode("utf-8", errors="ignore")
                self._send_json({"error": f"HTTP {he.code}: {err_body}"}, 502)
            except Exception as e:
                self._send_json({"error": f"Tunnel connection failed: {e}"}, 502)
            return

        self._send_json({"error": "Not found"}, 404)


if __name__ == "__main__":
    print(f"Starting Cyber-Ornith Kaggle Gateway Dashboard on 0.0.0.0:{PORT}...", flush=True)
    server = HTTPServer(("0.0.0.0", PORT), DashboardHandler)
    server.serve_forever()
