---
name: camoufox-browser-ops
description: >-
  Use the browse tool (Camoufox/Playwright when available) for navigation, DOM text, screenshots, and multi-step page interaction.
domain: browser
subdomain: automation
tags: [camoufox, browser, playwright]
version: "1.0"
license: Apache-2.0
---
## Workflow
1. browse with action=goto and the target URL.
2. action=content or screenshot as needed.
3. On failure, retry once; fall back to computer_fetch_url or web_search.
4. Respect CAPTCHA limits — report when blocked, do not loop.

