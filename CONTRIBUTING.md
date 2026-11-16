# Contributing to Sentinel

Sentinel v2.0.0-beta.1 is the current experimental beta. Development and documentation live on **`master`**.
The [Wiki](https://github.com/hemkesh2021-dotcom/Sentinel_Surveillance/wiki) and
[beta release notes](docs/RELEASE_NOTES_v2.0.0-beta.1.md) describe the current scope.
This is a small student-maintained project.

## Before you start

- Search existing issues. Discuss a large change before implementing it.
- Use the installation issue form for the tester package, the bug form for runtime/UI defects, and private
  vulnerability reporting for security issues.
- Keep device observations separate from portable tests. Only one reference Jetson has been used for the recorded
  checks; report your hardware and versions rather than assuming compatibility.
- Preserve matching thresholds, gallery data, admission records and existing resource profiles unless the change
  explicitly addresses them. A passing synthetic test does not establish recognition accuracy or device acceptance.

## Development setup

Fork and clone the repository, then branch from current `master`:

```bash
git switch master
git pull --ff-only
git switch -c fix/short-description
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
sentinel config validate config/default.yaml
python -m pytest tests/unit/<affected-test-file>.py
```

Python 3.10 or newer is required. The portable package and tests do not need a camera or a model;
browser tests need a Chromium-family browser. Run tests appropriate to the change.
CI runs the v2 suite on Python 3.10 and 3.12 and the legacy dashboard checks separately.
The [tester installation guide](packaging/beta-install/QUICKSTART.md) is for device setup, not a prerequisite
for documentation or portable development.

Keep application code in `src/sentinel/`; the root `surveillance4_1.py`, `dashboard.py` and their setup guides
are legacy v1. The tester archive has its own pinned contents and revision record. Changes to a packaged path
must identify the affected archive revision; never replace a published asset silently.

## Useful contributions

- Reproducible runtime, dashboard or installer bug reports
- Documentation corrections and clearer setup/recovery steps
- Camera/platform compatibility observations with exact versions
- Focused regression tests for confirmed failures
- Measured performance reports with configuration, method and limits stated

## Pull request checklist

- [ ] One logical change, based on current `master`
- [ ] Relevant tests run; failures and skipped checks explained
- [ ] Device checks, if performed, identify hardware, L4T/JetPack, Python, artifacts and configuration
- [ ] No credentials, control keys, enrollment, face photos, footage or private run folders
- [ ] Documentation updated where behavior or setup changes
- [ ] Resource/accuracy claims are measured and scoped; no admission implied by CI
- [ ] Packaged-file changes identify their installer impact

Reports and commands must be sanitized. Include short relevant error excerpts, not complete run logs or secrets.
Do not start a camera/model workload merely to verify a documentation-only change.

## Code and commits

Keep functions focused, hot paths bounded and device-only imports out of portable import paths.
Use atomic file writes where needed. Explain new dependencies.
Use concise commit subjects such as `fix: preserve Ask scrolling during resize` or
`docs: update beta installation guidance`.

## Licensing and conduct

Sentinel's own code is MIT; see [LICENSE](LICENSE). Contributions use the same terms.
Downloaded software and models keep their own terms, listed in
[THIRD_PARTY_NOTICES.md](packaging/beta-install/THIRD_PARTY_NOTICES.md).
Do not add or redistribute third-party files without checking their terms.

Read [SECURITY.md](SECURITY.md) for private vulnerability reporting and
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community expectations.
