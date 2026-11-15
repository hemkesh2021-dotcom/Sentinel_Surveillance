# Sentinel implementation handoff

## Context and sources

The user is implementing Sentinel v2 beta with Claude Code. This folder originally
contains planning artifacts, not application source. Apply this handoff in the
actual Sentinel_Surveillance repository; inspect its current HEAD and local
changes before editing. Do not assume a historical audited commit is current.

Read SENTINEL_V2_BETA_IMPLEMENTATION_GUIDE.md for requirements and acceptance
criteria. Start with chapters 2–6, 20–22, 25 and 32, then read chapters relevant
to the selected package. Read SENTINEL_AUDIT_REVIEW_2026-09-23.md for corrections
and regression cases. These files should accompany this file in the repository
root. Reconcile conflicts explicitly; do not silently substitute an older plan.
The audit's proposed v1 snippets are not a validated patch set. The referenced
72-issue master plan was not verified; use the guide's V2-01…V2-56 backlog.

## Implementation rules

- Preserve the working prototype and local changes. Develop v2 on a separate
  branch; migrate incrementally through tested interfaces.
- Create only modules needed for the current slice. Start with C1 rather than
  scaffolding all 56 packages or replacing models immediately.
- Keep portable development/replay checks separate from Jetson integration.
  Hardware adapters must not require CUDA/TensorRT merely to import portable code.
- Record actual Jetson software and camera capabilities before changing runtime
  dependencies. Preserve the validated JetPack family and use platform-specific
  dependency profiles. Never infer GPU compatibility from a successful Mac install.
- Carry frame/source time, boot ID and stream epoch through observations and
  evidence. Use an injected monotonic clock for age within a boot. Reject stale
  or mismatched results for current decisions; late results may annotate history.
- Associate each face with at most one person and vice versa. Ambiguous ownership
  stays unresolved. Handle no face and empty enrollment without labeling a person
  a face-confirmed stranger. Keep identity separate from access rules.
- Publish empty, fresh, stale and offline states explicitly. Preserve scene
  checks when there are no people. A VLM alone must not establish a critical fire
  alert; expire confirmation across gaps, outages and epoch changes.
- Keep queues, buffers and retained image payloads bounded. Use SQLite incident
  and outbox transactions with restart-safe delivery, rather than an in-memory
  queue as the durability mechanism. Validate provider results and sanitize errors.
- Detection, identity and scene understanding remain product requirements.
  Measure optimization while preserving their quality and coverage. Whole-device
  memory targets are decimal 5,000,000,000 bytes steady / 5,400,000,000 bytes peak;
  they are acceptance targets, not measurements already achieved.
- PIR/LiDAR, semantic-model experiments and additional cameras remain deferred
  as specified in the guide. Do not expand beta scope by implication.
- Keep credentials, tokens, face data and private footage out of source control.

## Continuity and evidence

Maintain docs/IMPLEMENTATION_STATUS.md with current branch/base commit, selected
package, changes, exact verification commands/results, unresolved decisions,
hardware checks pending and the next concrete task. Use existing equivalent
project records if present rather than duplicating them.

For each slice, add meaningful behavioral regressions and run the relevant
checks. Record failures and checks not run honestly. Mac replay success does not
establish camera, GPU, memory, throughput or beta readiness. Keep progress tied
to package acceptance criteria; revise effort estimates for actual availability.
