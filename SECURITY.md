# Security Policy

Sentinel is a self-hosted surveillance system. It processes live camera feeds, stores face embeddings, and exposes a web dashboard — so security reports are taken seriously.

## Supported versions

| Version | Security reporting |
| ------- | ------------------ |
| v2.0.0-beta.1 and current `master` | Reports accepted; experimental beta, not production-accepted |
| Earlier v2 alphas | Reproduce against the current beta where possible; historical tags remain available |
| v1.x | Security fixes only, best effort |

The [beta release](https://github.com/hemkesh2021-dotcom/Sentinel_Surveillance/releases/tag/v2.0.0-beta.1)
and [current limitations](https://github.com/hemkesh2021-dotcom/Sentinel_Surveillance/wiki/Status-and-limitations)
describe feature validation separately from this reporting policy.

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

- v2 dashboard/status server: loopback access, request validation, chat and operator-key-protected feature controls
- Sealed enrollment storage, file permissions, consent/revocation and artifact verification
- RTSP input, live preview and optional incident snapshots
- SQLite incidents and delivery outbox; Telegram credentials and notification handling
- Installer downloads, checksums, local file handling and process cleanup
- Configuration and secrets handling
- Legacy v1 Flask dashboard and pickle-loading paths, identified explicitly as v1
- Any other code in this repository

Upstream defects should also be reported to the dependency's maintainers; tell us if Sentinel's usage makes them
exploitable. Physical-access-only issues and social engineering remain outside this policy's scope.

## Current security boundaries

**v2.0.0-beta.1** uses a loopback-only dashboard (`127.0.0.1:18090`) reached through an SSH tunnel.
It does not provide general user login, roles or HTTPS itself. Do not publish the status port to the LAN or Internet.
The operator key authorizes runtime feature changes; it is not authentication for reading the dashboard.
Chat and feature requests enforce their page token and same-origin checks, and feature changes additionally require
the run's operator key.

The v2 gallery is **sealed with a passphrase**, using authenticated encryption; it is not a pickle file.
Its identity directory and files use owner-only permissions. Incidents, run records and optional snapshots are
separate data and should not be assumed to be encrypted by the gallery's encryption.
v2 does not import the legacy `face_db.pkl`.

**Legacy v1** has a separate Flask/dashboard and pickle-gallery implementation. Do not load an untrusted v1 pickle
or apply its networking/setup instructions to v2. The [legacy guide](docs/LEGACY_V1.md) identifies those files.

These boundaries do not claim complete hardening. Report defects in them privately even if a related limitation
is documented. Never include real credentials, face data or footage in a public issue.
