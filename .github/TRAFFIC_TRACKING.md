# Sentinel repository traffic archive

Workflow: **Track Repository Traffic**, installed on the default branch `master`.
Schedule: **23:45 UTC / 05:15 IST daily**, with manual dispatch available.
GitHub can delay or drop scheduled runs; this is not an exact-time guarantee.

Report: [traffic-log/TRAFFIC_REPORT.md](https://github.com/hemkesh2021-dotcom/Sentinel_Surveillance/blob/traffic-log/TRAFFIC_REPORT.md).

## One-time credential setup

1. Create a fine-grained personal access token for **only**
   `hemkesh2021-dotcom/Sentinel_Surveillance`.
2. Set **Administration: Read-only**. Metadata access is automatic. No Contents
   write permission is needed on this token. Choose an expiry and renew before it.
3. Store it as the repository Actions secret **TRAFFIC_TOKEN** in
   [Actions secrets](https://github.com/hemkesh2021-dotcom/Sentinel_Surveillance/settings/secrets/actions).
   Do not put the value in source, issues, logs or chats.
4. Open [the workflow](https://github.com/hemkesh2021-dotcom/Sentinel_Surveillance/actions/workflows/traffic-tracker.yml),
   select **Run workflow**, and use **master**. Verify a green run and an updated report.

The dedicated token reads the four Traffic API endpoints. The short-lived
`GITHUB_TOKEN` writes archive files using job-level Contents write permission.
The workflow never clones, fetches or checks out the repository. Python and gh
come from the GitHub-hosted runner; no dependencies or third-party actions are installed.
Source and regression tests are fetched through the Contents API at `GITHUB_SHA`.

## Data and interpretation

- `traffic-log` is an independent data-only branch with no inherited source or CI.
- Every successful run archives all four unmodified API response bodies plus a
  collection timestamp in a uniquely named `snapshots/YYYY/MM/` JSON file.
- Daily history and Markdown report update with the snapshot in one commit.
  Concurrency is serialized, and ref updates reject non-fast-forward races.
- Overlapping days replace previous observations, including GitHub corrections.
  Missing dates remain unknown. Recent buckets can be partial; no lifetime
  coverage is claimed. Days older than GitHub's retained window cannot be
  recovered unless previously archived elsewhere.
- Views and clones are events. Monthly sums of daily uniques are visitor-days
  and cloner-days, not distinct people over the month. The latest rolling-window
  uniques are preserved separately as returned by GitHub.
- Referrers and paths are top-10 rolling-window results. Historical peaks are
  labelled separately; an absent referrer is not falsely shown as current or zero.
- Report and data are public because this repository is public. Only aggregate
  repository traffic is collected. GitHub traffic does not identify visitors,
  distinguish all human/bot/developer activity or establish product adoption.

## Maintenance

Check Actions failures and the report's collection timestamp and latest bucket.
401 usually means an expired/revoked credential; 403 can mean missing permissions
or repository access. Renew the dedicated token and replace TRAFFIC_TOKEN.
GitHub may disable schedules in a public repository after 60 days without activity;
re-enable in Actions if needed. A gap beyond GitHub's retention window is permanent.
Do not infer exact execution time, complete bot exclusion, or exact global uniques.

Run local regression tests with `python3 .github/scripts/test_traffic_tracker.py`.
They cover overlapping windows, downward corrections, missing data, stale
referrers, table escaping, schema refusal and atomic publication structure.

References: [Traffic API](https://docs.github.com/en/rest/metrics/traffic),
[scheduled workflows](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule).
