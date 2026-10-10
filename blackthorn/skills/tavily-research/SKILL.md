---
name: tavily-research
description: >-
  Multi-query Tavily research with source comparison and citation hygiene.
domain: research
subdomain: web
tags: [tavily, research]
version: "1.0"
license: Apache-2.0
---
## Workflow
1. Break the question into 1–3 search queries.
2. Call web_search for each; keep URLs and dates.
3. Synthesize with explicit citations; flag conflicts.

