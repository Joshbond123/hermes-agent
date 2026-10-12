# Qwen3.8-27B-Uncensored on Kaggle

This is the **current** Blackthorn model: `Qwen3.8-27B-Uncensored`, Q4_K_M GGUF from `ressl/Qwen3.8-27B-uncensored-GGUF` (fallback `orcarouter/Qwen3.8-27B-Uncensored-GGUF`). Do not use a legacy 9B model name in configuration or agent memory.

| Component | Configuration |
| --- | --- |
| Model host | Dedicated Kaggle notebook with 2× Tesla T4; Ollama or llama-server; 8192-token default context |
| Computer host | Separate Kaggle notebook (different account, independent lifecycle), authenticated FastAPI `/computer/*` |
| Tunnel | Inrok named share when configured; named Cloudflare Tunnel or Render relay if configured; disposable Quick Tunnel only as last resort |
| Computer state | Private, versioned Kaggle dataset `<computer-account>/blackthorn-computer-state`; `/kaggle/working/blackthorn_workspace` is temporary between notebook sessions |
| Gateway key | Generated per boot, sent only to D1 and the notebook; never committed to Git |

The 16.8 GB GGUF exceeds one Tesla T4's 15 GB VRAM; the server attempts multi-GPU offloading and reports real startup and warmup status. The computer notebook does not load the model, and turning off an idle model must not stop its workspace.

The controller builds notebook text from `cyber_ornith_server.py` on each push. The computer notebook restores an existing dataset snapshot without overwriting newer local files, restores absent packages from the saved inventory where possible, starts and probes its gateway, then publishes its authenticated public URL to D1. It checkpoints periodically and before a verified explicit stop. Tunnel restarts do not reset files. A Kaggle session has a finite runtime and its local filesystem is not durable until a dataset checkpoint succeeds; large files, excluded caches, and secrets are not included. See the reliability report for test evidence and limits.
