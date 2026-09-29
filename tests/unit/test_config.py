from __future__ import annotations

from pathlib import Path

import pytest

from sentinel.cli import main
from sentinel.config import ConfigError, load_config
from sentinel.media.clock import NS_PER_SECOND

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "sentinel.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def problems_for(tmp_path: Path, text: str) -> list[str]:
    with pytest.raises(ConfigError) as caught:
        load_config(write_config(tmp_path, text))
    return caught.value.problems


def test_shipped_default_config_is_valid_and_uses_the_guide_targets() -> None:
    config = load_config(DEFAULT_CONFIG)
    assert config.camera.id == "cam-1"
    assert config.freshness.stale_after_ns == 2 * NS_PER_SECOND
    assert config.freshness.offline_after_ns == 10 * NS_PER_SECOND
    assert config.freshness.track_expiry_ns == 1 * NS_PER_SECOND


def test_every_problem_is_reported_with_its_location(tmp_path: Path) -> None:
    problems = problems_for(
        tmp_path,
        """\
config_version: 1
camera:
  id: 7
freshness:
  stale_afer_s: 2.0
  offline_after_s: -1
  track_expiry_s: .inf
""",
    )
    assert sorted(problems) == sorted(
        [
            "camera.id: Input should be a valid string (got 7)",
            "freshness.offline_after_s: Input should be greater than 0 (got -1)",
            "freshness.track_expiry_s: Input should be a finite number (got inf)",
            "freshness.stale_afer_s: unknown setting (did you mean 'stale_after_s'?)",
        ]
    )


def test_offline_threshold_must_exceed_stale_threshold(tmp_path: Path) -> None:
    problems = problems_for(
        tmp_path,
        "config_version: 1\ncamera: {id: cam-1}\nfreshness: {stale_after_s: 5, offline_after_s: 2}\n",
    )
    assert problems == ["freshness.offline_after_s: must be greater than stale_after_s (2.0 <= 5.0)"]


def test_duplicate_keys_are_rejected_instead_of_silently_overriding(tmp_path: Path) -> None:
    problems = problems_for(
        tmp_path,
        """\
config_version: 1
camera: {id: cam-1}
freshness:
  stale_after_s: 2.0
  stale_after_s: 20.0
""",
    )
    assert problems == ["YAML error at line 5, column 3: found duplicate key 'stale_after_s'"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "config_version: 2\ncamera: {id: cam-1}\n",
            "config_version: unsupported version 2; this build reads version 1",
        ),
        ("config_version: 1\n", "camera: required setting is missing"),
        # YAML 1.1 reads a bare `on` as a boolean
        ("config_version: 1\ncamera: {id: on}\n", "camera.id: Input should be a valid string (got True)"),
        (
            'config_version: 1\ncamera: {id: cam-1}\nfreshness: {stale_after_s: "2"}\n',
            "freshness.stale_after_s: Input should be a valid number (got '2')",
        ),
        ("- config_version: 1\n", "top level must be a mapping of settings, not list"),
        ("config_version: 1\ncamera: {id: cam-1\n", "YAML error at line 3, column 1:"),
    ],
)
def test_invalid_documents_fail_with_a_specific_message(
    tmp_path: Path, text: str, expected: str
) -> None:
    problems = problems_for(tmp_path, text)
    assert len(problems) == 1
    assert problems[0].startswith(expected)


def test_errors_never_echo_url_credentials(tmp_path: Path) -> None:
    problems = problems_for(
        tmp_path, 'config_version: 1\ncamera:\n  id: "rtsp://admin:hunter2@192.0.2.10/live"\n'
    )
    assert len(problems) == 1
    assert "hunter2" not in problems[0]
    assert "rtsp://<redacted>@192.0.2.10/live" in problems[0]


def test_cli_validate_exit_status_and_messages(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["config", "validate", str(DEFAULT_CONFIG)]) == 0
    assert "valid Sentinel configuration" in capsys.readouterr().out

    bad = write_config(tmp_path, "config_version: 1\ncamera: {id: cam-1}\nfreshness: {offline_after_s: 1}\n")
    assert main(["config", "validate", str(bad)]) == 1
    assert (
        "freshness.offline_after_s: must be greater than stale_after_s (1.0 <= 2.0)"
        in capsys.readouterr().err
    )

    assert main(["config", "validate", str(tmp_path / "missing.yaml")]) == 1
    assert "cannot read file (No such file or directory)" in capsys.readouterr().err
