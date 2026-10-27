# Contributing to Sentinel

Thanks for your interest in Sentinel — a self-hosted edge AI surveillance system for the NVIDIA Jetson Orin Nano Super.

This is a small project maintained by students. Contributions are welcome, but please read this first so your time (and ours) is well spent.

## Before you start

- **Search existing issues** before opening a new one.
- **Open an issue before a large PR.** Describe the change and get a go-ahead; this avoids wasted effort on work that doesn't fit the roadmap.
- **Hardware reality:** the maintainer has the only Jetson used for testing. If your change touches TensorRT, CUDA, llama.cpp, or the camera pipeline, say what hardware you tested on. Untested changes to the inference path will be merged only after on-device verification by the maintainer.

## Ways to contribute

- Bug reports (use the Bug report issue template)
- Feature requests (use the Feature request template)
- Documentation fixes — README accuracy, setup steps, BUILD notes
- Camera compatibility reports — which RTSP cameras work or don't
- Performance findings on Jetson (FPS, memory, latency) with the exact command and model used

## Development setup

1. Fork the repo and clone your fork.
2. Create a branch from `main`:
   ```bash
   git checkout -b fix/short-description
   ```
3. Follow the README setup steps (Python environment, model downloads, `llama-server` launch).
4. Make your change.
5. Test it. For Jetson-specific work, include:
   - JetPack / L4T version
   - `llama-server` launch command (including `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` if used)
   - Memory state (`free -h` or `tegrastats`) before and during the run

## Pull request checklist

- [ ] One logical change per PR
- [ ] Branch is up to date with `main`
- [ ] Code runs on Jetson Orin Nano (or clearly marked as untested on device)
- [ ] No hardcoded secrets, tokens, chat IDs, or RTSP credentials
- [ ] README / docs updated if behaviour or setup changed
- [ ] Performance claims are measured, not estimated — state the method

## Code style

- Python: follow PEP 8. Keep functions small; the main loop is latency-sensitive.
- Don't add dependencies without a reason stated in the PR.
- Don't write to `/tmp` or other SD-card-heavy paths in hot loops.
- Prefer atomic file writes (write to temp, then rename).

## Commit messages

Use short imperative subjects:

```
fix: prevent identity bleed across tracks in multi-face recognition
docs: correct unified memory env var in README step 6
feat: add dashboard auth middleware
```

## Licensing

- Sentinel is released under the MIT License (see `LICENSE`).
- By contributing, you agree your contribution is licensed under the same terms.
- Note: YOLOv8 (Ultralytics) is AGPL-3.0 and LFM2-VL has its own license. Do not add code or weights that change the project's licensing position without discussing it in an issue first.

## Reporting security issues

Do **not** open a public issue for vulnerabilities. See [SECURITY.md](SECURITY.md).

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
