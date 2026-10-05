# Blackthorn Phase 1–3 Status (2026-10-05)

## Model
- **Replaced** Cyber-Ornith-1.5-9B-OBLITERATED → **Qwen3.8-27B-Uncensored**
- GGUF: `ressl/Qwen3.8-27B-uncensored-GGUF` Q4_K_M (~16.8 GB)
- Dual T4: num_ctx=8192, full GPU offload, single parallel slot
- Kaggle kernel: https://www.kaggle.com/code/joshbond123/qwen38-27b-uncensored-api-server
- API key: `${MODEL_API_KEY}`

## Overlay (Cloudflare D1)
- Updated overlay SHA deployed to D1 (`blackthorn_overlay_*`)
- Hermes defaults: `HERMES_MODEL=Qwen3.8-27B-Uncensored`, custom provider
- System prompt stored in D1 key `blackthorn_system_prompt`
- Tavily rotating keys in D1 key `tavily_api_keys`
- Agent router: `/api/system-prompt`, `tavily_search` tool, SSE stream at `/api/studio/agent/stream`

## Render
- Redeploys triggered with clearCache to pull new overlay

## Still in progress
- Full ChatGPT-style UI restyle (frontend rebuild)
- Avatar removal in React chat
- Kali tooling bootstrap on Kaggle
- End-to-end model health once Qwen finishes downloading on Kaggle (~16GB first run)
