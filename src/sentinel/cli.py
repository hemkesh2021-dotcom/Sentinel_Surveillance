"""``sentinel`` command line (guide chapter 18). Only implemented commands exist."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from . import __version__
from .adapters import resolve
from .config import ConfigError, load_config


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sentinel")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    config_parser = commands.add_parser("config", help="configuration tools")
    config_commands = config_parser.add_subparsers(dest="config_command", required=True)
    validate = config_commands.add_parser("validate", help="validate a configuration file")
    validate.add_argument("path", help="YAML configuration file")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.path)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(
        f"{args.path}: valid Sentinel configuration "
        f"(version {config.config_version}, camera {config.camera.id})"
    )
    # Optional adapters never block startup; show why any cannot be used.
    statuses = resolve(config.adapters)
    if not statuses:
        print("  no optional adapters configured: core monitoring only")
    for status in statuses:
        print(f"  adapter {status.manifest.adapter_id}: {status.state.value} ({status.reason})")
    for zone in config.zones:
        if zone.schedule is None:
            when = "always active"
        else:
            windows = ", ".join(f"{w.start}-{w.end}" for w in zone.schedule.windows)
            when = f"{windows} {zone.schedule.timezone}"
        state = "enabled" if zone.enabled else "disabled"
        print(f"  zone {zone.zone_id}: {zone.rule}, {state}, {zone.severity}, {when}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
