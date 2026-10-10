---
name: authorized-recon-basics
description: >-
  Authorized reconnaissance playbook for systems you own or have written permission to test.
  Uses only passive checks and explicitly allowed active probes. Never targets third-party systems.
domain: cybersecurity
subdomain: reconnaissance
tags: [recon, authorized, nmap, osint]
version: "1.0"
license: Apache-2.0
---
## When to Use
User asks for recon, asset inventory, or port/service discovery on systems they own or authorize.

## Prerequisites
- Explicit authorization in the user message or system policy
- Computer tools: run_command, list_files, read_file
- Prefer passive checks first

## Workflow
1. Confirm scope: only hostnames/IPs the user authorized.
2. Passiveify connectivity: `ping -c 2 <host>` or `curl -I --max-time 5 <url>`.
3. If nmap is available and scope is authorized: `nmap -Pn -sV --top-ports 100 <host>`.
4. Summarize open services, versions, and risk notes. Do not exploit.
5. Save notes under workspace `security_notes/`.

## Out of scope
Exploitation, credential stuffing, scanning systems without permission.

