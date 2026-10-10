---
name: network-hygiene-check
description: >-
  Check the agent computer network posture (listening ports, outbound DNS, basic firewall state) for hygiene reporting.
domain: cybersecurity
subdomain: hardening
tags: [hygiene, ports, host]
version: "1.0"
license: Apache-2.0
---
## Workflow
1. `ss -tulpn` or `netstat -tulpn` for listening services.
2. `ip a` / `hostname -I` for interfaces.
3. Note unexpected listeners; do not change firewall rules unless asked.

