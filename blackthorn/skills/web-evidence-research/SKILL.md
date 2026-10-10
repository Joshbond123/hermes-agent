---
name: web-evidence-research
description: >-
  Research a claim with Tavily web_search and optional browser fetch; preserve URLs and dates; compare sources.
domain: research
subdomain: web
tags: [tavily, evidence, citations]
version: "1.0"
license: Apache-2.0
---
## Workflow
1. web_search with a precise query.
2. Extract titles, URLs, published dates when present.
3. Optionally browse 1–3 top sources for detail.
4. Compare claims; mark uncertain items.
5. Never invent sources.

