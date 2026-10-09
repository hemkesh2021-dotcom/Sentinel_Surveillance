# Alpha flow: inspect, stop, acknowledge, restart, inspect

This is the Review 1 alpha's incident flow on the device (D67, D68 in
`docs/IMPLEMENTATION_STATUS.md`). It covers the guide's journey steps "inspect
an incident and evidence → acknowledge it → see the notification outcome →
restart … and retain the incident" in their alpha form.

> **Monitoring is interrupted during this flow.** From the moment `sentinel
> run` stops until the restarted runtime reports `"run": "starting"` and its
> camera is streaming again, nothing is watched. During that gap there is no
> detection, no incident, no alert and no status page. The gap includes the
> runtime's startup and model loading. Its length on the device has not been
> measured. Do not use this flow when the space must be watched continuously.

## Why monitoring has to stop

The status page is read only and bound to 127.0.0.1 (D45); it has no
acknowledge button. Acknowledging is a write to the incident database, and
that database has exactly one writer at a time. The writer lock is a
kernel `flock` on `sentinel.db.lock`, taken without waiting:

- while `sentinel run` owns the database, `sentinel incident ack` is refused
  (`database_busy`) and changes nothing;
- while `sentinel incident ack` owns it, a `sentinel run` that starts is refused
  at startup (`database: another process is writing to sentinel.db; only one
  Sentinel runtime or operator command may own it`), before any model loads;
- the kernel releases the lock when its owner closes the database or exits,
  even after a crash, so no stale lock is left behind.

This is a lock on the database file, not a check of process names.

## What acknowledging does, and does not do

- It moves an **open** incident to **acknowledged**, as one transaction. The
  acknowledgment is the incident's transition record: the UTC time, `operator:<actor>` and the
  note. It is stored with the database's durable settings (`synchronous=FULL`).
- It is **not** resolving. The incident stays unresolved, and a later
  observation of the same rule and zone can still join it (guide ch. 9: acknowledged means seen, not ended).
- It changes no evidence, severity or rule state. It sends no alert and withdraws none. An alert that
  was still waiting before the acknowledgment is delivered as before (delivery
  is at least once).
- **Repeating it changes nothing.** The command reports `already_acknowledged`
  with the first acknowledgment's time, actor and note.
- Refusals change nothing, and each is a fixed label with exit status 1:
  - `invalid_incident_id`: not an incident ID's form; the database is not opened;
  - `no_database`: no `sentinel.db` in `--data-dir`; nothing is created;
  - `database_busy`: monitoring is still running;
  - `unknown_incident`: no such incident in that database;
  - `not_acknowledgeable`: the incident is resolved or dismissed.

## The steps

Use the same `--data-dir` in every step. The commands below use the runtime's
interpreter, as the operator packages do; in a development checkout,
`.venv/bin/sentinel` is the same command.

1. **Inspect, while monitoring runs.** Forward the page from the Mac with
   `ssh -L 18090:127.0.0.1:18090 <device>`, open `http://127.0.0.1:18090/` and
   follow the incident's link. Note its ID (`inc-…`), its status (`open`), its
   rule observations and its alert rows. The acknowledgment line reads
   `not acknowledged`.
2. **Stop monitoring.** Press Ctrl-C in the terminal running `sentinel run`,
   or send it SIGTERM. Under an operator guard, the guard stops it at its
   duration. Wait until it has exited. **Monitoring is now interrupted.**
3. **Acknowledge.** From the repository directory:

   ```bash
   PYTHONPATH=src ~/onvif_env/bin/python -m sentinel.cli incident ack --data-dir "$DATA" inc-…
   ```

   Optional: `--actor NAME` (default `local`) and `--note TEXT`. It prints one JSON line, with
   `"outcome": "acknowledged"` and the `acknowledged_utc` time, and exits 0.
4. **Restart monitoring** with the same `sentinel run` command and
   `--data-dir`. Monitoring resumes only after the runtime is ready and its
   camera is streaming.
5. **Inspect again.** Open the same incident page. Its status is
   `acknowledged`, and its acknowledgment line gives the time, the actor and
   the note. The incident, its evidence and its alert rows were kept across the
   restart. If the person is still in the zone, the new observation may join
   this acknowledged incident, which then stays acknowledged with no new
   alert. Or it may open a new, linked incident with its own alert: that
   depends on the rule's merge window (120 s by default) and on whether the
   device rebooted.

## Limits

- This is the alpha's stand-in for the guide's authenticated acknowledge
  action (V2-17). There are no users, roles or sessions; the actor is whatever
  name the operator types.
- There is no undo: an acknowledged incident can be resolved or dismissed, but
  not set back to open. No command for those transitions exists yet.
- The incident page appears only while `sentinel run` is running; nothing
  serves it from a stopped runtime's database.
