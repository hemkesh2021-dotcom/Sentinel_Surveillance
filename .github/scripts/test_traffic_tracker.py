import copy
import json
import unittest
from unittest.mock import patch

from traffic_tracker import merge, publish, report


def snapshot(day="2026-10-14", count=54, observed="2026-10-15T00:00:00.000000Z"):
    return {"collected_at": observed, "views": {"count": count, "uniques": 2, "views": [{"timestamp": day + "T00:00:00Z", "count": count, "uniques": 2}]},
            "clones": {"count": count, "uniques": 2, "clones": [{"timestamp": day + "T00:00:00Z", "count": count, "uniques": 2}]},
            "referrers": [{"referrer": "Google", "count": 5, "uniques": 2}], "paths": []}


class TrafficTests(unittest.TestCase):
    def test_overlap_replaces_not_adds_and_preserves_old_days(self):
        history = merge(None, snapshot())
        history = merge(history, snapshot(count=60, observed="2026-10-16T00:00:00.000000Z"))
        self.assertEqual(history["days"]["2026-10-14"]["clones"]["count"], 60)
        history = merge(history, snapshot("2026-11-01", 3, "2026-11-02T00:00:00.000000Z"))
        self.assertEqual(len(history["days"]), 2)
        self.assertIn("| 2026-10 | 1 | 60 | 2 | 1 | 60 | 2 |", report("a/b", history))

    def test_downward_corrections_and_old_snapshot(self):
        history = merge(None, snapshot())
        history = merge(history, snapshot(count=50, observed="2026-10-16T00:00:00.000000Z"))
        history = merge(history, snapshot(count=70))
        self.assertEqual(history["days"]["2026-10-14"]["views"]["count"], 50)
        self.assertEqual(history["latest"]["views"]["count"], 50)

    def test_missing_is_not_zero_and_window_uniques_not_summed(self):
        data = snapshot()
        data["views"]["views"] = []
        history = merge(None, data)
        self.assertNotIn("views", history["days"]["2026-10-14"])
        self.assertIn("| 2026-10-14 | — | — | 54 | 2 |", report("a/b", history))
        self.assertEqual(history["latest"]["clones"]["uniques"], 2)

    def test_old_referrer_is_not_current(self):
        history = merge(None, snapshot())
        newer = snapshot(observed="2026-10-16T00:00:00.000000Z")
        newer["referrers"] = []
        history = merge(history, newer)
        markdown = report("a/b", history)
        current = markdown.split("## Referrers in latest response")[1].split("## Historically")[0]
        self.assertNotIn("Google", current)
        self.assertIn("Google", history["referrers_peak"])

    def test_escape_external_table_text(self):
        data = snapshot()
        data["referrers"][0]["referrer"] = "<script>|x\ny"
        markdown = report("a/b", merge(None, data))
        self.assertNotIn("<script>", markdown)
        self.assertIn("&lt;script&gt;&#124;x y", markdown)

    def test_invalid_data_refused_before_history_mutation(self):
        history = merge(None, snapshot())
        before = copy.deepcopy(history)
        data = snapshot()
        data["clones"]["clones"][0]["count"] = -1
        with self.assertRaises(ValueError):
            merge(history, data)
        self.assertEqual(history, before)

    def test_unknown_schema_refused(self):
        with self.assertRaises(ValueError):
            merge({"days": {}}, snapshot())

    def test_one_commit_and_non_force_update(self):
        calls = []
        def api(path, method="GET", data=None):
            calls.append((path, method, data))
            return {"sha": "new-tree" if path == "git/trees" else "new-commit"}
        with patch("traffic_tracker.load_history", return_value=("old-head", "old-tree", None)):
            publish(api, "a/b", snapshot(), "123-1")
        self.assertEqual([c[0] for c in calls], ["git/trees", "git/commits", "git/refs/heads/traffic-log"])
        files = calls[0][2]["tree"]
        self.assertEqual(len(files), 3)
        self.assertEqual(calls[1][2]["parents"], ["old-head"])
        self.assertEqual(calls[2][2], {"sha": "new-commit", "force": False})
        raw = next(x for x in files if x["path"].startswith("snapshots/"))
        self.assertEqual(json.loads(raw["content"]), snapshot())

    def test_first_publish_is_orphan_data_branch(self):
        calls = []
        def api(path, method="GET", data=None):
            calls.append((path, data))
            return {"sha": "created"}
        with patch("traffic_tracker.load_history", return_value=(None, None, None)):
            publish(api, "a/b", snapshot(), "123-1")
        self.assertNotIn("base_tree", calls[0][1])
        self.assertEqual(calls[1][1]["parents"], [])
        self.assertEqual(calls[2], ("git/refs", {"ref": "refs/heads/traffic-log", "sha": "created"}))


if __name__ == "__main__":
    unittest.main()
