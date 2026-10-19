"""Archive GitHub traffic without a repository checkout. Python standard library only."""

import base64
import datetime as dt
import html
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

BRANCH = "traffic-log"
HISTORY = "traffic_history.json"
ENDPOINTS = {
    "views": "traffic/views?per=day",
    "clones": "traffic/clones?per=day",
    "referrers": "traffic/popular/referrers",
    "paths": "traffic/popular/paths",
}


def dumps(value):
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


class API:
    def __init__(self, repo, token):
        self.base = f"https://api.github.com/repos/{repo}/"
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "sentinel-traffic-tracker",
        }

    def __call__(self, path, method="GET", data=None):
        body = json.dumps(data).encode() if data is not None else None
        for attempt in range(3):
            request = urllib.request.Request(
                self.base + path, data=body, headers=self.headers, method=method
            )
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    raw = response.read()
                    return json.loads(raw) if raw else None
            except urllib.error.HTTPError as error:
                if method == "GET" and error.code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError):
                if method == "GET" and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise


def validate_snapshot(snapshot):
    for metric in ("views", "clones"):
        response = snapshot[metric]
        for record in [response] + response[metric]:
            for key in ("count", "uniques"):
                if type(record[key]) is not int or record[key] < 0:
                    raise ValueError(f"Invalid {metric} {key}")
            if record["uniques"] > record["count"]:
                raise ValueError(f"Invalid {metric} unique count")
        days = [row["timestamp"][:10] for row in response[metric]]
        if len(set(days)) != len(days):
            raise ValueError(f"Duplicate {metric} dates")
        for day in days:
            dt.date.fromisoformat(day)
    for metric, label in (("referrers", "referrer"), ("paths", "path")):
        for row in snapshot[metric]:
            if not isinstance(row[label], str):
                raise ValueError(f"Invalid {metric} label")
            for key in ("count", "uniques"):
                if type(row[key]) is not int or row[key] < 0:
                    raise ValueError(f"Invalid {metric} count")


def merge(history, snapshot):
    validate_snapshot(snapshot)
    if history is None:
        history = {"schema_version": 1, "days": {}, "referrers_peak": {}}
    if history.get("schema_version") != 1:
        raise ValueError("Unknown history format; refusing to overwrite it")
    observed = snapshot["collected_at"]
    for metric in ("views", "clones"):
        for row in snapshot[metric][metric]:
            day = row["timestamp"][:10]
            entry = history["days"].setdefault(day, {})
            previous = entry.get(metric, {})
            # Latest observation wins, including downward corrections by GitHub.
            # Missing dates/metrics remain unknown; never manufacture zeroes.
            if observed >= previous.get("observed_at", ""):
                entry[metric] = {"count": row["count"], "uniques": row["uniques"], "observed_at": observed}
    for row in snapshot["referrers"]:
        entry = history["referrers_peak"].setdefault(row["referrer"], {})
        entry["peak_14d_views"] = max(entry.get("peak_14d_views", 0), row["count"])
        entry["peak_14d_uniques"] = max(entry.get("peak_14d_uniques", 0), row["uniques"])
        entry["last_seen"] = max(entry.get("last_seen", ""), observed)
    if observed >= history.get("collected_at", ""):
        history["collected_at"] = observed
        history["latest"] = {key: snapshot[key] for key in ENDPOINTS}
    return history


def cell(value):
    return html.escape(str(value)).replace("|", "&#124;").replace("\n", " ").replace("\r", " ").replace("`", "&#96;")


def report(repo, history):
    days = history["days"]
    latest = history["latest"]
    months = {}
    for day, entry in sorted(days.items()):
        month = months.setdefault(day[:7], {m: {"count": 0, "uniques": 0, "days": 0} for m in ("views", "clones")})
        for metric in ("views", "clones"):
            if metric in entry:
                month[metric]["count"] += entry[metric]["count"]
                month[metric]["uniques"] += entry[metric]["uniques"]
                month[metric]["days"] += 1
    totals = {m: {k: sum(month[m][k] for month in months.values()) for k in ("count", "uniques", "days")} for m in ("views", "clones")}
    first, last = (min(days), max(days)) if days else ("unknown", "unknown")
    lines = [
        f"# Repository traffic — {repo}", "",
        f"Last successful collection: **{history['collected_at']}**. Daily dates are UTC.", "",
        f"Observed daily coverage: **{first} → {last}**. Missing dates are unknown, not zero.", "",
        "This report measures repository interest, not product usage or validated adoption. Clones may include automation and development activity. The collector uses API calls only, with no git clone/fetch/checkout.", "",
        "## Latest GitHub rolling-window response", "",
        "GitHub supplies approximately the last 14 days. These window-level unique counts are GitHub's own deduplicated figures. Referrers and paths are limited to the top 10.", "",
        "| Metric | Events | Unique visitors/cloners in API window | Latest daily bucket |",
        "| --- | ---: | ---: | --- |",
    ]
    for metric in ("views", "clones"):
        data = latest[metric]
        newest = max((row["timestamp"][:10] for row in data[metric]), default="none")
        lines.append(f"| {metric.title()} | {data['count']} | {data['uniques']} | {newest} |")
    lines += ["", "## Totals across recorded days", "",
        "These cover only observed dates, not the repository's lifetime. Daily unique counts are summed visitor-days/cloner-days: the same person can count on multiple days. Monthly and lifetime distinct people cannot be recovered by summing daily uniques.", "",
        "| Metric | Recorded events | Sum of daily uniques | Observed dates |",
        "| --- | ---: | ---: | ---: |"]
    for metric in ("views", "clones"):
        value = totals[metric]
        lines.append(f"| {metric.title()} | {value['count']} | {value['uniques']} | {value['days']} |")
    lines += ["", "## Monthly summary", "", "Months may be partial. Separate coverage counts expose gaps in either metric.", "",
        "| Month | View dates | Views | Visitor-days | Clone dates | Clones | Cloner-days |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for month, value in sorted(months.items(), reverse=True):
        v, c = value["views"], value["clones"]
        lines.append(f"| {month} | {v['days']} | {v['count']} | {v['uniques']} | {c['days']} | {c['count']} | {c['uniques']} |")
    lines += ["", "## Referrers in latest response", "", "Absence from the top 10 does not establish zero traffic.", "",
        "| Referrer | Rolling-window views | Rolling-window uniques |", "| --- | ---: | ---: |"]
    for row in latest["referrers"]:
        lines.append(f"| {cell(row['referrer'])} | {row['count']} | {row['uniques']} |")
    if not latest["referrers"]:
        lines.append("| None returned | — | — |")
    lines += ["", "## Historically observed referrer peaks", "", "Peaks refer to overlapping rolling windows; they are not cumulative totals or current counts.", "",
        "| Referrer | Peak window views | Peak window uniques | Last observed UTC |", "| --- | ---: | ---: | --- |"]
    for name, row in sorted(history["referrers_peak"].items()):
        lines.append(f"| {cell(name)} | {row['peak_14d_views']} | {row['peak_14d_uniques']} | {row['last_seen']} |")
    lines += ["", "## Popular paths in latest response", "", "| Path | Views | Uniques |", "| --- | ---: | ---: |"]
    for row in latest["paths"]:
        lines.append(f"| {cell(row['path'])} | {row['count']} | {row['uniques']} |")
    lines += ["", "## Daily breakdown", "", "Recent dates may be partial or revised by GitHub. A later response replaces the stored daily count; every raw response remains archived.", "",
        "| UTC date | Views | Daily unique visitors | Clones | Daily unique cloners |", "| --- | ---: | ---: | ---: | ---: |"]
    for day, entry in sorted(days.items(), reverse=True):
        values = [entry.get(m, {}).get(k, "—") for m in ("views", "clones") for k in ("count", "uniques")]
        lines.append(f"| {day} | " + " | ".join(map(str, values)) + " |")
    lines += ["", "## Archive and operation", "",
        "Raw responses: `snapshots/YYYY/MM/`. Machine-readable merged daily history: `traffic_history.json`. All report files update together in one commit on `traffic-log`.", "",
        "Schedule: 23:45 UTC / 05:15 IST daily, subject to GitHub scheduling delays. Check Actions for failures. Expired/revoked TRAFFIC_TOKEN credentials require renewal; public-repository schedules may be disabled after 60 days without activity.", "",
        "This branch is public, like the repository. It contains aggregate traffic only. No visitor identities, enrollment data, camera footage or credentials are collected.", ""]
    return "\n".join(lines)


def load_history(api):
    try:
        head = api(f"git/ref/heads/{BRANCH}")["object"]["sha"]
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None, None, None
        raise
    tree = api(f"git/commits/{head}")["tree"]["sha"]
    entries = api(f"git/trees/{tree}")["tree"]
    item = next((row for row in entries if row["path"] == HISTORY), None)
    if item is None:
        raise ValueError("Existing traffic-log has no recognized history; refusing to overwrite it")
    blob = api(f"git/blobs/{item['sha']}")
    history = json.loads(base64.b64decode(blob["content"]))
    return head, tree, history


def publish(api, repo, snapshot, run_key):
    head, tree, history = load_history(api)
    history = merge(history, snapshot)
    stamp = snapshot["collected_at"]
    path = f"snapshots/{stamp[:4]}/{stamp[5:7]}/{stamp.replace(':', '').replace('-', '')}-{run_key}.json"
    files = {HISTORY: dumps(history), "TRAFFIC_REPORT.md": report(repo, history), path: dumps(snapshot)}
    payload = {"tree": [{"path": path, "mode": "100644", "type": "blob", "content": content} for path, content in files.items()]}
    if tree:
        payload["base_tree"] = tree
    new_tree = api("git/trees", "POST", payload)["sha"]
    commit = api("git/commits", "POST", {
        "message": f"chore(traffic): archive {stamp}", "tree": new_tree,
        "parents": [head] if head else [],
    })["sha"]
    if head:
        # Non-fast-forward writes fail instead of discarding concurrent updates.
        api(f"git/refs/heads/{BRANCH}", "PATCH", {"sha": commit, "force": False})
    else:
        # Start an independent data-only branch, without inherited workflows.
        api("git/refs", "POST", {"ref": f"refs/heads/{BRANCH}", "sha": commit})
    return commit, files["TRAFFIC_REPORT.md"]


def main():
    repo = os.environ["GITHUB_REPOSITORY"]
    reader = os.environ.get("TRAFFIC_TOKEN", "").strip()
    writer = os.environ.get("GITHUB_TOKEN", "").strip()
    if not reader or not writer:
        raise SystemExit("TRAFFIC_TOKEN (Administration: read) and GITHUB_TOKEN (Contents: write) are required. See .github/TRAFFIC_TRACKING.md.")
    read_api, write_api = API(repo, reader), API(repo, writer)
    snapshot = {"repository": repo, "collected_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")}
    # Fetch all endpoints successfully before making any repository write.
    snapshot.update({name: read_api(path) for name, path in ENDPOINTS.items()})
    run_key = os.environ.get("GITHUB_RUN_ID", "manual") + "-" + os.environ.get("GITHUB_RUN_ATTEMPT", "1")
    commit, markdown = publish(write_api, repo, snapshot, run_key)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as stream:
            stream.write(f"[Open traffic report](https://github.com/{repo}/blob/{BRANCH}/TRAFFIC_REPORT.md)\n\n" + markdown)
    print(f"Archived traffic successfully: {commit}")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as error:
        raise SystemExit(f"GitHub API returned HTTP {error.code}; check token access, expiry and branch rules. No token or response body is logged.") from None
