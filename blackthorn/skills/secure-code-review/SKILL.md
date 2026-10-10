---
name: secure-code-review
description: >-
  Review source code for common vulnerabilities (injection, authz, secrets, insecure crypto) without running exploits.
domain: cybersecurity
subdomain: application-security
tags: [code-review, appsec, secrets]
version: "1.0"
license: Apache-2.0
---
## Workflow
1. List relevant files; read entry points and auth paths.
2. Check for injection sinks, hardcoded secrets, weak crypto, missing authz.
3. Report findings with file paths, severity, and remediation.
4. Do not run destructive tests.

