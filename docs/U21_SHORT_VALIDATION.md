# Operator inspection and guarded diagnostics — PENDING

Codex session 8, 2026-10-03. This tested entry point supersedes session 7's inline
controller and the historical unguarded Check 8/9 recipes. **No inspection or
hardware workload was executed by Codex.** U20's portable increment is complete;
real-model quality, demo-exception evidence, U18/U19/U21 acceptance and the
30-minute rerun remain PENDING. Claude session 11 adds the opt-in S1 prompt-cache
A/B mode (D38) and the D37 cache-drop preparation step; Claude ran neither.

## Start here: one read-only command

```bash
/home/villain8001/sentinel-surveillance/.venv/bin/python /home/villain8001/sentinel-surveillance/benchmarks/runner/operator_check.py
```

Return the JSON it prints. It also saves a mode-0600 `result.json` in a new
mode-0700 `/tmp/sentinel-operator-*` directory, outside Git. It reads meminfo,
swap counters, process names (never argv/environment), service state, repository
revision and the current boot's latest 1,000 kernel records. It saves only
whitelisted states, numeric readings and OOM/NvMap candidate-line counts, not
kernel messages. Each read-only subprocess has a 3 s deadline and 0.2 s TERM/
KILL cleanup grace; normal operation takes seconds. Permission denial, empty,
malformed, oversized or timed-out logs are **unavailable**, never zero errors.
Candidate lines are not unique events; absence cannot exclude earlier failures.

No CUDA API, model binary, model import, stream, allocation probe, cache eviction
or service mutation runs by default. Listening-port inspection connects only to
loopback 18081 without sending a model/HTTP request. Asset checks use metadata
only on the existing runtime/pretrained models, not clips or enrollment data.
An inspection can succeed while reporting reasons that prohibit workloads.

## Explicit execution, only after reviewing inspection output

One entry point, three **separate** opt-in modes; none automatically runs another.
Workload options are not approval to execute them in this session.

- `--execute-workload check9`: bounded device/managed API smoke, sequentially,
  each at most **256 MiB / 268,435,456 B**, with **32 MiB** chunks, projected
  pressure/headroom checks before each chunk, finally-based release, and a
  **20 s** process deadline. Failure of the first probe prevents the second.
  Missing prerequisites start neither. No allocation-until-failure, cache-fill,
  low-MemFree experiment or llama-server API comparison runs. The bounded smoke
  **does not answer allocation beyond MemFree**; Check 9b/U18 policy/acceptance
  stay PENDING even after it completes. Native/context overhead is outside the
  explicit buffer budget and is monitored separately, not guaranteed bounded.
- `--execute-workload u21`: additionally requires `--check9-report` naming a
  successful private, same-boot/repository-revision Check 9 result and
  `--confirm-u21-prerequisites`. That option is an explicit operator attestation
  of reviewed U18 conservative-load policy, the provisional model/template
  exception, unchanged workload conditions and separate short-run approval.
  It is not inferred from a filename listing or portable test success. Invalid,
  missing, mismatched or incomplete reports prevent the dependent workload.
  Reuses `demo_profile.py` with **synthetic noise only**, no cache eviction,
  sanitized logs, 15 s idle baseline/settling, 30 s warm-up, **120 s steady**,
  face 1 Hz, scene every 4 s, and a **360 s** child/descendant deadline. Existing
  llama readiness/load limits are 60/90 s; HTTP requests are bounded at 2/30 s.
  Expected duration is about 4–5 min, not 30 min. Quality is not evaluated.

- `--execute-workload s1 --s1-arm a|b` (D38, prompt-cache A/B): the same
  prerequisites as U21 (successful private same-boot/revision Check 9, operator
  attestation `--confirm-s1-prerequisites`), the same admission, guard, 360 s
  deadline and owned-group cleanup. It runs `demo_profile.py --scene-only`:
  llama-server plus scene requests only, **no detector, face model, torch or CUDA
  driver in the workload process**, 15 s baseline/settle, 30 s warm-up, **180 s
  steady** (about 45 requests at one per 4 s), no cache eviction, sanitized logs.
  Every request carries its own deterministic synthetic noise image (index *i* is
  byte-identical in both arms; no two requests share an image). Arm **a** keeps
  llama-server's default host-RAM prompt cache (b8932: 8192 MiB); arm **b** adds
  only `--cache-ram 0`. Nothing else differs. Expected about 5 min per arm.
  The result adds a numeric excerpt: steady trend (first/last/least-squares
  slope of pressure, MemFree, Cached and llama-server PSS), the prompt-cache
  numbers llama-server logged (startup limit; entries/MiB per update; duplicate,
  eviction and allocation-failure counts), cumulative per-request scene counters
  (`scene_progress`, emitted after every request so a stop keeps them) and unload
  residues. It compares the arms; it is not scene accuracy, a long-run result or
  approval to adopt `--cache-ram 0` (a separate decision).

`--latest-check9-report` (instead of `--check9-report PATH`) selects the newest
`/tmp/sentinel-operator-*/result.json` whose mode is `check9`; it must still pass
every same-boot/revision/success/privacy check. If the newest Check 9 failed or
was refused, the workload refuses: an older success is never used instead.

After the initial output is returned, identify actual remaining conditions and
provide the appropriate concrete next command; do not guess a report path now.

## Conditions and stops

Run as the normal maintainer user, awake on one boot, headless from plain SSH,
without desktop/dev tools, v1, model servers or competing media/GPU work. Both
Ollama **service state and process presence** are reported; inactive is not
disabled, and a service can be absent while a manually started worker exists.
Unclassified Python processes prevent execution because v1/ML work cannot be
ruled out. The sole narrow exception is the root-owned MainPID of the installed
`nvidia-pva-allowd.service`, with exact cgroup membership, active/running systemd
metadata, no unit drop-ins, the expected launch path and package ownership plus
matching installed checksums for its launcher/unit. Identity checks fail closed;
other Python processes, including additional processes in that cgroup, still
refuse execution. Fixed known-service labels/PIDs are reported separately; its
memory stays in whole-device pressure without subtraction. This exception is
**not proof of GPU/PVA idleness**. Read-only identity subprocesses each have a
3 s deadline. Systemd launch metadata is captured only to check the executable
path; arguments are never displayed/saved. Missing process/service inspection, assets, revision or
kernel access, root execution, or an occupied port causes refusal. Existing
services/processes are never stopped, disabled or killed by this workflow.
Resolve conditions with the maintainer rather than adding automatic shutdowns.

### Headless preparation checklist (operator only)

1. Save desktop work and use a plain SSH session that survives desktop logout.
   Close your own dev-tool sessions after this Codex turn; keep the host awake.
2. Record `timeout --kill-after=1s 3s systemctl is-active display-manager.service`.
   Only if it was active and you choose to interrupt the desktop, authenticate
   with `timeout --kill-after=2s 30s sudo -v`, then run
   `timeout --kill-after=2s 30s sudo -n systemctl stop display-manager.service`.
   Do not disable it. On timeout/unknown state, stop preparation and inspect,
   since timing out the systemctl client does not cancel an outstanding job.
3. Leave the NVIDIA PVA service running. Do not automatically stop Ollama, v1,
   other services or unrelated processes; resolve remaining refusals explicitly.
   Run the default inspection command and return its JSON.
   **Cache drop (D37, standing for measurement runs only):** before a Check 9,
   U21 or S1 measurement run, the operator may run
   `sync && sudo sysctl -w vm.drop_caches=1` and must then pass
   `--operator-dropped-caches`, so the result records
   `preparation.drop_caches: operator_declared`. The runner never drops caches,
   calls sudo or verifies the declaration; this is never part of the Sentinel
   runtime (D-1 must not depend on it; U18 stays open).
4. Restore the desktop **only if it was active before your explicit stop**:
   `timeout --kill-after=2s 30s sudo -n systemctl start display-manager.service`.
   Reauthenticate with the bounded `sudo -v` command if necessary; never change
   enablement. Check its state again. No workload follows inspection automatically.

The admission snapshot requires **MemTotal - MemAvailable <2,000,000,000 B**,
**MemFree >=3,500,000,000 B**, **MemAvailable >=4,000,000,000 B**, and valid swap
counters. It is checked again before GPU work and each probe. These are
conservative diagnostic refusal conditions, not relaxed beta gates.

The guard samples every **0.2 s**, streaming numeric readings to `guard.jsonl`
and retaining only scalar extrema/counters. Stop on the first observed:

- pressure **>=4,800,000,000 B** (decimal 4.8 GB);
- MemFree **<1,073,741,824 B** (1 GiB), or MemAvailable **<2,147,483,648 B** (2 GiB);
- changed global swap-page counters, missing/invalid telemetry, non-finite or
  backwards clock, a sample gap >0.5 s, interruption or execution timeout.

Pressure means **whole-device MemTotal - MemAvailable**, not process allocator
bytes, RSS/PSS or tegrastats. Linux meminfo `kB` is KiB, multiplied by 1024;
samples store integer bytes and sample-start `time.monotonic()` seconds within
one boot (Linux excludes suspend). The benchmark must stay awake. The profiler
also samples torch's process-only current/lifetime peaks at most 1 Hz and
tegrastats at 1 Hz; these views must not be added together or used to prove
whole-device usage/unload. Missing readings remain unavailable, not zero.

**4.8 GB is a sampled stop threshold, not a guaranteed cap.** The nominal 600 MB
distance to the 5.4 GB ceiling accommodates guard overhead, transients and
shutdown, but neither sampling nor that margin bounds instantaneous allocations.
Guard footprint is included in absolute pressure and the child's idle baseline;
overhead/observer effects remain unmeasured. A spike causes immediate stop, not
debounce. Global swap activity may belong to another service. A stopped run is
inconclusive, not proof of a linear leak, plateau or component attribution.

## Cleanup, privacy and evidence

Each directly orchestrated subprocess starts in a new owned session/process
group; the reused profiler's children inherit its group. `finally` sends
TERM, then KILL if necessary, **only to that group**; workload cleanup waits
at most 5 s per signal and read-only cleanup 0.2 s per signal. Descendants are
included even if their parent exits first. Lingering groups are reported as
cleanup failure and prevent dependent phases; uninterruptible kernel tasks
cannot be guaranteed to disappear. A 5 s post-exit sample is not U19's longer
reclamation check. No `pkill`, service stop/disable, reboot or drop-caches exists
in this workflow's code; the D37 cache drop is an operator command outside it.

The reused profiler's sanitized mode drops raw model/worker output and retains
only fixed GPU-placement markers, numeric buffers and bounded known telemetry.
No raw descriptions, environment, credentials, private media or face templates
are copied. Never delete existing review images. Clips remain outside Git;
the first pilot never opens them. Neither the current filenames nor an
empty-room filename proves file equivalence, consent or full-clip review.

Return the printed sanitized result; additional numeric profiler artifacts stay
under that run's private temporary directory. Record conditions and limitations
when treating operator output as **USER-SUPPLIED MEASUREMENT**. Bounded Check 9
success and a short U21 completion do not grant hardware acceptance, approve the
demo exception, validate scene accuracy or authorize the 30-minute rerun.
