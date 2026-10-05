# Qwen3.8-27B-Uncensored on Kaggle dual T4

Replaces Cyber-Ornith-1.5-9B-OBLITERATED.

| Setting | Value |
| --- | --- |
| Model | `Qwen3.8-27B-Uncensored` (GGUF Q4_K_M ~16.8 GB) |
| Source | `ressl/Qwen3.8-27B-uncensored-GGUF` |
| Hardware | 2× NVIDIA Tesla T4 (~30 GB VRAM) |
| Runtime | Ollama + Cloudflare Quick Tunnel |
| API key | `${MODEL_API_KEY}` |

## Dual-T4 memory strategy
- Q4_K_M weights ~16.8 GB → fits on one T4 with headroom
- Default context 8192 (override with `QWEN38_NUM_CTX`)
- All layers offloaded to GPU (`num_gpu=999`)
- Single parallel slot to avoid KV OOM

## Boot
Push this folder as a Kaggle notebook with NvidiaTeslaT4 accelerator (same flow as before).
The server reports tunnel URL into Cloudflare D1 / connection_state.json.
