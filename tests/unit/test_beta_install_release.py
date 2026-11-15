"""The tester package as a public pre-release asset (session 116, part 48): its licence, its notices, its entry page,
the installation issue form and the version it names.

Documents and the builder's output only: nothing is installed, downloaded, built or launched here. A pass says the
files say what they must and name what the manifests pin; it says nothing about any licence's meaning.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from beta_install_fixture import REPO, build_archive

PACKAGE = REPO / "packaging/beta-install"
ISSUES = "https://github.com/hemkesh2021-dotcom/Sentinel_Surveillance/issues/new?template=installation-issue.md"


def flat(path: Path) -> str:
    return " ".join(path.read_text().split())


def pins() -> dict[str, str]:
    found: dict[str, str] = {}
    for name in ("requirements-pypi.txt", "requirements-jetson.txt"):
        for match in re.finditer(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)", (PACKAGE / "manifest" / name).read_text(), re.M):
            found[match.group(1)] = match.group(2)
    return found


def test_the_archive_carries_the_repositorys_licence_and_says_what_it_does_not_cover() -> None:
    name, members = build_archive.package_members(build_archive.DEFAULT_REVISION, worktree=True)
    assert members["LICENSE"] == (REPO / "LICENSE").read_bytes()
    assert b"Permission is hereby granted, free of charge" in members["LICENSE"]
    for document in ("README.md", "QUICKSTART.md", "THIRD_PARTY_NOTICES.md"):
        assert document in members, document
    assert not (PACKAGE / "LICENSE").exists()  # one licence file, the repository's; the builder puts it in
    version = members["VERSION"].decode()
    assert "status=experimental tester pre-release; validates and admits nothing; not accepted for production\n" in version
    assert "license=MIT for the files in this package (LICENSE, sha256 " in version
    # nothing third-party is inside: the archive is text the project wrote, plus the licence
    assert all(b"\0" not in content for content in members.values())
    assert not [path for path in members if re.search(r"\.(whl|gguf|pt|onnx|engine|h5|so|deb|tar|gz|zip)$", path)]
    # no file still calls the package private or forbids passing it on
    for path, content in members.items():
        if path.startswith("app/") or path == "SHA256SUMS":
            continue
        text = " ".join(content.decode().split())
        for gone in ("not for redistribution", "for the maintainer's testers only", "Private, for"):
            assert gone not in text, (path, gone)


def test_the_notices_name_every_pinned_package_and_every_download_with_its_source() -> None:
    notices = (PACKAGE / "THIRD_PARTY_NOTICES.md").read_text()
    rows = dict(re.findall(r"^\| ([A-Za-z0-9_.-]+) \| ([0-9][^ |]*) \| [^|]+ \|$", notices, re.M))
    pinned = pins()
    assert len(pinned) == 95 and rows == pinned  # every pin with its version, and no package that is not pinned
    agpl = {name for name in rows if re.search(rf"^\| {re.escape(name)} \| [^|]+ \| AGPL-3\.0 \|$", notices, re.M)}
    assert agpl == {"ultralytics", "ultralytics-thop"}
    assets = json.loads((PACKAGE / "manifest/assets.json").read_text())["assets"]
    for asset in assets:
        if asset["kind"] == "download":
            assert Path(asset["path"]).name in notices, asset["id"]
            assert asset["revision"] in notices, asset["id"]
        elif "commit" in asset:
            assert asset["commit"] in notices and asset["repo"] in notices, asset["id"]
    said = " ".join(notices.split())
    for statement in ("**None of what is listed here is inside this package.**",
                      "It grants no permission, changes no licence and is not legal advice",
                      "GNU Affero General Public License version 3", "https://www.ultralytics.com/license",
                      "LFM Open License v1.0", "10 million US dollars", "This package does not redistribute the files.",
                      "NVIDIA's software licence agreements", "The package contains none of them and downloads none of them",
                      "License types will be inherited when you intend to utilize those models.",
                      "https://pypi.jetson-ai-lab.io/jp6/cu126", "https://github.com/ggml-org/llama.cpp"):
        assert statement in said, statement
    # the manifest's own licence words agree with the notices
    by_id = {asset["id"]: asset for asset in assets}
    assert by_id["detector_weights"]["license"].startswith("AGPL-3.0")
    assert by_id["scene_model"]["license"].startswith("LFM Open License v1.0")
    assert by_id["llama_server"]["license"] == "MIT" and by_id["face_detector"]["license"].startswith("MIT")


def test_the_entry_page_says_what_ran_what_did_not_and_what_a_tester_supplies() -> None:
    page = flat(PACKAGE / "README.md")
    for statement in (
            "**Start with [`QUICKSTART.md`](QUICKSTART.md).**",
            "**Experimental, and it validates nothing.**",
            "**The default check and launch refuse on your device.**",
            "Revision 5 of this package completed a camera-free build rehearsal on the reference device",
            "were checked **offline only**",
            "**Never run:** a complete launch from this package; the complete check on real builds; anything on another Jetson.",
            "**Stock JetPack is unverified.**", "NOT VERIFIED", "`--accept-unverified-platform`",
            "**Device resource validation is pending.**",
            "Your enrollment: photos of yourself, sealed with your own passphrase",
            "**yes, only after you type START**", "**yes: reads your photos and writes your sealed enrollment**",
            "ssh -N -o ExitOnForwardFailure=yes -L 18090:127.0.0.1:18090 <user>@<jetson>",
            "http://127.0.0.1:18090/dashboard", "Ctrl-C once, or `bash stop.sh`",
            "[`LICENSE`](LICENSE)", "[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)", "AGPL-3.0",
            "**The issue tracker is public.**", ISSUES):
        assert statement in page, statement
    for claim in ("validated on", "works on any", "supported on stock", "has been admitted"):
        assert claim not in page, claim
    guide = flat(PACKAGE / "QUICKSTART.md")
    assert "## 15. Licences" in guide and ISSUES in guide and "**The tracker is public.**" in guide
    assert "An experimental tester pre-release (Sentinel v2.0.0-alpha.3, application version `2.0.0a3`)" in guide
    assert "https://github.com/hemkesh2021-dotcom/Sentinel_Surveillance/releases" in guide


def test_the_issue_form_asks_for_versions_and_errors_and_for_nothing_private() -> None:
    form = (REPO / ".github/ISSUE_TEMPLATE/installation-issue.md").read_text()
    head, _, body = form.partition("\n---\n")
    assert head.startswith("---\nname: Installation issue (tester package)\n") and "\nabout: " in head
    assert ISSUES.endswith("template=installation-issue.md")
    for asked in ("**Package** (the first line of `VERSION`", "**Platform** (the `ok`, `NOT VERIFIED` and `FAIL` lines of `bash preflight.sh`",
                  "**Step**", "**What happened** (the exact `STOP:` line", "`--accept-unverified-platform`"):
        assert asked in body, asked
    warning = " ".join(body[body.index("<!--"):body.index("-->")].split())
    for kept_out in ("This tracker is PUBLIC.", "the camera's address, user name or password", "the gallery passphrase",
                     "the feature-control key", "photos, footage", "your enrollment", "whole log files or run folders"):
        assert kept_out in warning, kept_out
    # outside that warning it requests no secret, picture or whole log
    asks = body[:body.index("<!--")] + body[body.index("-->"):]
    for never in ("passphrase", "password", "control key", "photo", "footage", "attach", "upload", "run.err", "RTSP", "URL"):
        assert never.lower() not in asks.lower(), never
    assert len(form.splitlines()) <= 40  # concise


def test_one_version_everywhere() -> None:
    import sentinel

    project = re.search(r'^version = "([^"]+)"$', (REPO / "pyproject.toml").read_text(), re.M).group(1)
    _, members = build_archive.package_members(build_archive.DEFAULT_REVISION, worktree=True)
    assert sentinel.__version__ == project == "2.0.0a3"
    assert f"application_version={project}\n" in members["VERSION"].decode()
    readme = (REPO / "README.md").read_text()
    assert "git clone --branch v2.0.0-alpha.3 " in readme and "# sentinel 2.0.0a3" in readme
    assert "docs/RELEASE_NOTES_v2.0.0-alpha.3.md" in readme and "packaging/beta-install/QUICKSTART.md" in readme
    assert "see [`LICENSE`](LICENSE)" in readme
