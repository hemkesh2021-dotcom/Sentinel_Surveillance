# U21 guarded short validation — PENDING

Prepared by Codex, 2026-10-03. **Not executed or approved for execution.** This is
a diagnostic proposal, not U19/U21 hardware acceptance or the 30-minute rerun.
The reported +0.12 decimal GB/min is an observed trend in one historical run,
not an established linear leak. No allocator policy, cadence, threshold or
dependency is changed by this document.

## Decisions and prerequisites

- Obtain a separate maintainer go-ahead for this short model-loading run. Settle
  Check 9/U18's safe load/admission policy first; a failed precheck is a stop,
  not permission to allocate through it or drop system caches.
- Confirm the provisional b8932 demo exception's model/template/schema evidence
  in U20. The shorter prompt has only portable tests, not model acceptance.
- Use the already-installed runtime and cached model assets. Nothing installs,
  downloads, rebuilds or accesses the camera, enrollment or private recordings.
- This first pilot deliberately uses **synthetic noise, not two_people_doorway_60s**.
  Its memory/scene labels cannot be compared as an equal-input replay or quality
  gate. An approved labelled replay and equal-duration repeats come later.
- Default headless/dev-tools preconditions remain. Existing v1 or llama-server
  causes refusal, not automatic shutdown. Keep Ollama and every other existing
  service unchanged; record their operator-confirmed state. If services or
  headroom prevent the check, stop and ask for a separate maintenance decision.
- Run as the normal maintainer user from this repository, never root. Port
  18081 must be free. Do not add `--allow-desktop`/`--allow-dev-tools` or silently
  change conditions to get a pass. No service state has been freshly verified.

## Metric, bounds and interpretation

Whole-device pressure is **MemTotal - MemAvailable**, in integer bytes; Linux
meminfo `kB` is explicitly multiplied by 1024. Report absolute readings and
baseline-subtracted readings, not a sum of allocator/RSS/PSS/tegrastats values.
The profile records meminfo at 0.2 s, process PSS/swap counters at 1 s, torch
allocator current/lifetime peaks at most 1 Hz, and tegrastats at 1 s. Monotonic
sample-start seconds are comparable within one boot; UTC labels the session.
Torch statistics exclude llama-server, TensorRT/native allocations outside the
torch allocator and other processes; they cannot measure the whole device or
prove unload. Lifetime peaks are not reset at the steady boundary.

The additional controller samples meminfo and swap-page counters every **0.2 s**,
retaining only scalar extrema/baseline counters and streaming a restricted log.
Its idle precheck requires pressure <2,000,000,000 B, MemFree >=3,500,000,000 B
and MemAvailable >=4,000,000,000 B. Stop on the first sample of any of:

- pressure **>=4,800,000,000 B** (decimal 4.8 GB);
- MemFree **<1,073,741,824 B** (1 GiB), or MemAvailable **<2,147,483,648 B** (2 GiB);
- any increase in global swap-in/out page counters from the pilot baseline;
- missing/invalid telemetry, a sampling gap >0.5 s, interruption or timeout.

These are proposed conservative diagnostic thresholds, **not revised beta
gates**. Stopping at 4.8 GB leaves 600 MB below the existing 5.4 GB ceiling for
transients and shutdown. The guard's own Python/logging footprint is included
in absolute pressure, and the child's baseline includes the controller. Compare
observed sample gaps and matched guarded/unguarded overhead only in a later,
separately approved safe experiment; no negligible-overhead claim is made.
There is no debounce through a spike and no allocation-to-failure probe.
Sampling and the reserve cannot guarantee an unseen instantaneous peak stays
below 5.4 GB. A threshold hit, even during load, means **aborted/inconclusive**,
not a plateau, a leak diagnosis or hardware failure attribution. Global swap
changes could come from an existing service, not the workload.

The child uses a 15 s baseline, 15 s settling phases, 30 s warm-up and **120 s
steady phase**, face 1 Hz and scene every 4 s. Typical duration is about 4–5 min;
the absolute child/descendant budget is **360 s**, plus at most 10 s cleanup and
a 5 s post-exit observation. llama readiness/load have explicit 60/90 s bounds;
each health HTTP request is 2 s, each scene HTTP request 30 s (the D16 8 s deadline
is currently counted, not cancellation-enforced). Metadata subprocesses use the
existing 60 s timeout but all descendants also share the controller's 360 s
deadline. No 30-minute run starts. `--no-evict` avoids changing model page-cache
state deliberately; this is **not an equal cold-cache repeat** of Check 8.

## Operator command — loads models and allocates GPU memory; PENDING

Review the prerequisites first. This copy-pasteable proposal creates a private
temporary output directory, launches only the existing profiler in a new owned
session/process group, and terminates/kills **that group only** in `finally`,
including any surviving llama-server/workload/tegrastats descendants. It never
uses `pkill`, stops a service, logs environment variables/command lines or opens
private media. A missing asset/dependency or profiler refusal ends the attempt.

```bash
cd /home/villain8001/sentinel-surveillance || exit 1
/usr/bin/python3 - <<'PY'
import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

os.umask(0o077)
output = Path(tempfile.mkdtemp(prefix="sentinel-u21-short-"))
process = None
interrupted = False
reason = "not_started"
samples = 0
peak = None
minimum_free = None
baseline = None

def request_stop(signum, frame):
    global interrupted
    interrupted = True

for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(signum, request_stop)

def snapshot():
    started = time.monotonic()
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        fields = value.split()
        if key in ("MemTotal", "MemAvailable", "MemFree"):
            if len(fields) != 2 or fields[1] != "kB" or not fields[0].isdigit():
                raise RuntimeError("meminfo_unavailable")
            memory[key] = int(fields[0]) * 1024
    if set(memory) != {"MemTotal", "MemAvailable", "MemFree"}:
        raise RuntimeError("meminfo_unavailable")
    if not (0 <= memory["MemAvailable"] <= memory["MemTotal"] and 0 <= memory["MemFree"] <= memory["MemTotal"]):
        raise RuntimeError("meminfo_invalid")
    swap = {}
    for line in Path("/proc/vmstat").read_text().splitlines():
        key, value = line.split()
        if key in ("pswpin", "pswpout"):
            swap[key] = int(value)
    if len(swap) != 2:
        raise RuntimeError("swap_counters_unavailable")
    return {
        "t_mono": started, "pressure_bytes": memory["MemTotal"] - memory["MemAvailable"],
        "free_bytes": memory["MemFree"], "available_bytes": memory["MemAvailable"], **swap,
    }

def signal_owned_group(signum):
    if process is not None:
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass

def group_exists():
    if process is None:
        return False
    process.poll()
    try:
        os.killpg(process.pid, 0)
        return True
    except ProcessLookupError:
        return False

try:
    baseline = snapshot()
    if (baseline["pressure_bytes"] >= 2_000_000_000
            or baseline["free_bytes"] < 3_500_000_000
            or baseline["available_bytes"] < 4_000_000_000):
        reason = "initial_headroom_refused"
    elif interrupted:
        reason = "interrupted"
    else:
        deadline = time.monotonic() + 360.0
        previous = baseline["t_mono"]
        with (output / "operator.log").open("w") as child_log, (output / "guard.jsonl").open("w") as guard_log:
            process = subprocess.Popen([
                "/usr/bin/python3", "benchmarks/runner/demo_profile.py", "--out", str(output),
                "--no-evict", "--baseline-s", "15", "--settle-s", "15", "--warmup-s", "30",
                "--steady-s", "120", "--face-hz", "1", "--scene-interval-s", "4",
                "--llama-timeout-s", "60", "--load-timeout-s", "90", "--min-free-gb", "3.5",
            ], stdout=child_log, stderr=subprocess.STDOUT, start_new_session=True)
            while True:
                reading = snapshot()
                guard_log.write(json.dumps(reading) + "\n")
                guard_log.flush()
                samples += 1
                peak = max(peak or 0, reading["pressure_bytes"])
                minimum_free = min(minimum_free if minimum_free is not None else reading["free_bytes"], reading["free_bytes"])
                if interrupted:
                    reason = "interrupted"
                elif reading["t_mono"] >= deadline:
                    reason = "timeout"
                elif reading["t_mono"] - previous > 0.5:
                    reason = "sampling_gap"
                elif reading["pressure_bytes"] >= 4_800_000_000:
                    reason = "pressure_stop"
                elif reading["free_bytes"] < 1_073_741_824 or reading["available_bytes"] < 2_147_483_648:
                    reason = "free_or_available_stop"
                elif reading["pswpin"] != baseline["pswpin"] or reading["pswpout"] != baseline["pswpout"]:
                    reason = "swap_counter_change"
                elif process.poll() is not None:
                    reason = "child_finished" if process.returncode == 0 else "child_refused_or_failed"
                else:
                    previous = reading["t_mono"]
                    time.sleep(max(0.0, reading["t_mono"] + 0.2 - time.monotonic()))
                    continue
                break
except Exception as error:
    reason = "controller_error_" + type(error).__name__
finally:
    signal_owned_group(signal.SIGTERM)
    cleanup_deadline = time.monotonic() + 5.0
    while group_exists() and time.monotonic() < cleanup_deadline:
        time.sleep(0.1)
    signal_owned_group(signal.SIGKILL)
    cleanup_deadline = time.monotonic() + 5.0
    while group_exists() and time.monotonic() < cleanup_deadline:
        time.sleep(0.1)
    leftovers = group_exists()

time.sleep(5.0)
try:
    post_exit = snapshot()
except Exception as error:
    post_exit = {"unavailable": type(error).__name__}
result = {
    "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "clock": "time.monotonic", "memory_units": "bytes", "guard_interval_s": 0.2,
    "initial_baseline": baseline, "child_returncode": process.poll() if process is not None else None,
    "status": reason, "owned_group_remaining": leftovers, "samples": samples,
    "sampled_peak_pressure_bytes": peak, "sampled_min_free_bytes": minimum_free,
    "post_exit": post_exit, "output": str(output), "hardware_acceptance": "PENDING",
}
(output / "guard-summary.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
raise SystemExit(0 if reason == "child_finished" and not leftovers else 1)
PY
```

There is no promise that a kernel-stuck process dies promptly; a remaining owned
group is an explicit cleanup failure. Do not repeat, kill unrelated processes
or use a system reset automatically. Return the diagnostic and request help.
SIGKILL may prevent the child's final summary/unload record; that is inconclusive
and must not be repaired into a successful run.

## Return to the maintainer/developer

Return `guard-summary.json`, and the new run's sanitized numeric summary/scene
counts if it completed: phase timings, absolute/baseline memory, six new meminfo
fields, process RSS/PSS, allocator current/lifetime peaks and availability,
sampling gaps, swap-page deltas, valid/invalid/rejection and character-boundary
counts. State exact repository/build/model/template, synthetic input, headless
state, power/thermal conditions and unchanged service state. Never post raw
logs, environment, credentials, images, embeddings or model descriptions.
Label supplied output **USER-SUPPLIED MEASUREMENT**, with these conditions and
limitations. A 5 s post-exit reading is not U19's longer reclamation check.
If the guard stops before steady, diagnose that condition first; do not raise
its threshold to fit the run. Only a separately approved, equal-input longer
run with attributed growth and unload evidence can advance hardware acceptance.
