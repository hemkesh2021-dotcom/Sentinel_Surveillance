# Security Policy

Sentinel is a self-hosted surveillance system. It processes live camera feeds, stores face embeddings, and exposes a web dashboard — so security reports are taken seriously.

## Supported versions

| Version | Supported |
| ------- | --------- |
| v2.x (current development) | Yes |
| v1.x | Security fixes only, best effort |

## Reporting a vulnerability

**Please do not open a public GitHub issue for security problems.**

Report privately using one of:

1. **GitHub Private Vulnerability Reporting** (preferred):
   go to the repository's **Security** tab → **Report a vulnerability**.
2. **Maintainer contact:** [@hemkesh2021-dotcom](https://github.com/hemkesh2021-dotcom) on GitHub.
   Do not post vulnerability details in public issues or discussions.

Include:

- A description of the issue and its impact
- Steps to reproduce, or a proof-of-concept
- Affected component (dashboard, RTSP ingest, face DB, Telegram alerting, llama-server integration, etc.)
- Your environment (JetPack version, deployment setup)

## What to expect

- Acknowledgement within **72 hours**
- An initial assessment within **7 days**
- You will be kept informed of progress; a fix or mitigation will be coordinated with you before public disclosure
- Credit in the release notes if you'd like it

This is a student-maintained project with no bug bounty. Response times are best effort.

## Scope

In scope:

- Flask dashboard (authentication, session handling, MJPEG stream exposure)
- Face database storage and loading (serialization, file permissions)
- RTSP / camera input handling
- Telegram alerting (token handling, message injection)
- Configuration and secrets handling
- Any code in this repository

Out of scope:

- Vulnerabilities in upstream dependencies (YOLOv8, llama.cpp, DeepFace, Flask) — report those upstream, but let us know if Sentinel's usage makes them exploitable
- Issues requiring physical access to the Jetson
- Social engineering

## Known limitations (as of v2 development)

These are acknowledged and on the hardening roadmap; reports that only restate them will be closed as duplicates:

- Dashboard has no authentication or TLS by default — run it only on a trusted LAN
- Face database uses Python pickle serialization
- Runtime files may be created with permissive permissions

See the project roadmap in the README for hardening status.
