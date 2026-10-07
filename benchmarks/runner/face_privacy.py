#!/usr/bin/env python3
"""Privacy counts for the V2-25 device check. Search terms arrive on stdin, never as arguments; only counts are printed.

    printf 'passphrase=%s\\0name=%s\\0name=%s\\0filename=%s\\0' ... | face_privacy.py f1 F1_DIR --audit AUDIT
    printf 'name=%s\\0name=%s\\0' ... | face_privacy.py f1 F1_DIR --audit AUDIT --keys name
    printf 'passphrase=%s\\0name=%s\\0name=%s\\0' ... | face_privacy.py f2 F2_DIR
    printf 'url=%s\\0' "$SENTINEL_RTSP_URL" | face_privacy.py camera F2_DIR
    printf 'url=%s\\0' "$SENTINEL_RTSP_URL" | face_privacy.py top DIR

stdin holds NUL-terminated ``kind=value`` terms: ``passphrase``, ``name`` (the first and the family name; an empty
one is ignored), ``filename`` (one per photo copy) and ``url`` (the camera URL; only its userinfo is searched).
benchmarks/runner/face_counts.sh writes them with bash's printf builtin, so no process ever has a term in its
command line. The output is ``<file> <key>=<n>`` per file, which face_check reads; every count must be 0.

- ``passphrase``, ``filenames``, ``userinfo``: exact substrings in every byte of the file, as grep -cF searched, and
  in every JSON string whose encoding differs from its text (escaped or non-ASCII characters).
- ``name``, three or more letters and digits: case-insensitive substrings likewise (Unicode NFKC, casefolded).
- ``name``, fewer (an initial, a two-letter name): whole tokens (runs of letters and digits; an apostrophe inside a
  word joins it) in the file's content as face_evidence reads it. Schema text cannot match; content always can.
- ``nonconforming`` (with ``name``): keys, values, lines or files that do not fit their profile (face_evidence).

Fail closed: a missing or empty required term, an unknown or repeated single term, a name without a letter, a URL
without userinfo, undecodable stdin or an unreadable directory exit 2 with no counts on stdout. ``--explain`` prints
where each hit is (file, location, kind of location) instead of the counts: never a term or the text around it.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import face_evidence

TOKEN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")
MIN_SUBSTRING_CHARS = 3  # names with fewer letters and digits count only as whole tokens in content
KINDS = ("passphrase", "name", "filename", "url")
SINGLE = ("passphrase", "url")
TOP_SKIP = frozenset({"SHA256SUMS", "secret-counts-top.txt"})


class Refused(Exception):
    pass


@dataclass
class Terms:
    passphrase: str | None = None
    names: list[str] = field(default_factory=list)
    filenames: list[str] = field(default_factory=list)
    userinfo: str | None = None


def norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def userinfo_of(url: str) -> str:
    netloc = urlsplit(url).netloc
    userinfo, at, _ = netloc.rpartition("@")
    if not at or not userinfo:
        raise Refused("the camera URL has no userinfo to count")
    return userinfo


def read_terms(data: bytes, keys: tuple[str, ...]) -> Terms:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise Refused("stdin is not UTF-8") from None
    if text and not text.endswith("\0"):
        raise Refused("stdin must be NUL-terminated kind=value terms")
    terms, seen = Terms(), Counter()
    for record in text.split("\0")[:-1]:
        kind, sep, value = record.partition("=")
        if not sep or kind not in KINDS:
            raise Refused("an unknown kind of term")
        seen[kind] += 1
        if kind in SINGLE and seen[kind] > 1:
            raise Refused(f"more than one {kind}")
        if kind == "passphrase":
            terms.passphrase = value
        elif kind == "name":
            if value.strip():
                if not any(ch.isalpha() for ch in value):
                    raise Refused("a name without a letter")
                terms.names.append(value.strip())
        elif kind == "filename":
            if value:
                terms.filenames.append(value)
        else:
            terms.userinfo = userinfo_of(value) if value else None
    need = {"passphrase": ("passphrase", terms.passphrase), "name": ("name", terms.names),
            "filenames": ("filename", terms.filenames), "userinfo": ("url", terms.userinfo)}
    for key in keys:
        kind, value = need[key]
        if not value:
            raise Refused(f"no {kind} given")
    unused = {k for k in KINDS if seen[k]} - {need[k][0] for k in keys}
    if unused:
        raise Refused("terms given that this count does not use")
    return terms


def short(name: str) -> list[str] | None:
    """The name's tokens when it is short (fewer than MIN_SUBSTRING_CHARS letters and digits), else None."""
    tokens = TOKEN.findall(norm(name))
    return tokens if sum(ch.isalnum() for ch in "".join(tokens)) < MIN_SUBSTRING_CHARS else None


@dataclass
class Hit:
    key: str
    where: str
    kind: str  # bytes, decoded string, content


def _line_of(data: bytes, offset: int) -> int:
    return data.count(b"\n", 0, offset) + 1


def count_file(path: Path, part: str | None, keys: tuple[str, ...], terms: Terms,
               name: str | None = None) -> tuple[dict[str, int], list[Hit]]:
    data = path.read_bytes()
    reading = face_evidence.read(path, part, name) if part is not None else face_evidence.Reading()
    if part is None and path.suffix in (".json", ".jsonl"):
        reading.decoded = face_evidence.decoded_strings(data.decode("utf-8", errors="replace"), path.suffix == ".jsonl")
    counts: dict[str, int] = {}
    hits: list[Hit] = []

    def exact(key: str, needles: list[str], folded: bool = False) -> None:
        haystack = norm(data.decode("utf-8", errors="replace")) if folded else data
        found = 0
        for needle in needles:
            target = norm(needle) if folded else needle.encode("utf-8")
            start = haystack.find(target)
            while start != -1:
                found += 1
                where = (haystack.count("\n", 0, start) + 1) if folded else _line_of(data, start)
                hits.append(Hit(key, f"line {where}", "bytes"))
                start = haystack.find(target, start + 1)
            for text in reading.decoded:
                n = (norm(text) if folded else text).count(norm(needle) if folded else needle)
                found += n
                hits.extend(Hit(key, "a JSON string", "decoded string") for _ in range(n))
        counts[key] = found

    for key in keys:
        if key == "passphrase":
            exact(key, [terms.passphrase])
        elif key == "filenames":
            exact(key, terms.filenames)
        elif key == "userinfo":
            exact(key, [terms.userinfo])
        elif key == "name":
            longs = [n for n in terms.names if short(n) is None]
            exact("name", longs, folded=True)
            wanted = Counter(t for n in terms.names for t in (short(n) or ()))
            if wanted:
                for where, text in reading.content:
                    for token in TOKEN.findall(norm(text)):
                        if token in wanted:
                            counts["name"] += 1
                            hits.append(Hit("name", where, "content"))
            counts["nonconforming"] = len(reading.nonconforming)
            hits.extend(Hit("nonconforming", where, "structure") for where in reading.nonconforming)
    return counts, hits


def _regular(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise Refused(f"{directory} is not a directory")
    return sorted(p for p in directory.iterdir() if p.is_file())


def files_for(command: str, directory: Path, audit: Path | None) -> list[tuple[str, Path]]:
    if command in ("f1", "f2", "camera"):
        files = [(p.name, p) for p in _regular(directory) if p.name not in face_evidence.NOT_COUNTED]
        if command == "f1":
            if audit is None or not audit.is_file():
                raise Refused("the audit log is missing")
            files.append(("audit.jsonl", audit))
        return files
    files = [(p.name, p) for p in _regular(directory) if p.suffix in (".txt", ".json", ".yaml")]
    if (directory / "f1").is_dir():
        files += [(f"f1/{p.name}", p) for p in _regular(directory / "f1")]
    return [(label, p) for label, p in files if label not in TOP_SKIP]


KEYS = {"f1": ("passphrase", "name", "filenames"), "f2": ("passphrase", "name"), "camera": ("userinfo",),
        "top": ("userinfo",)}
PART = {"f1": "f1", "f2": "f2", "camera": None, "top": None}


def run(command: str, directory: Path, stdin: bytes, *, audit: Path | None = None, keys: tuple[str, ...] | None = None,
        explain: bool = False) -> list[str]:
    keys = keys or KEYS[command]
    if not set(keys) <= set(KEYS[command]):
        raise Refused(f"{command} counts only {', '.join(KEYS[command])}")
    terms = read_terms(stdin, keys)
    out = []
    for label, path in files_for(command, directory, audit):
        try:
            counts, hits = count_file(path, PART[command], keys, terms, path.name)
        except OSError as exc:
            raise Refused(f"{label} could not be read ({type(exc).__name__})") from None
        if explain:
            out += [f"{label} {h.key}: {h.where} ({h.kind})" for h in hits]
        else:
            out += [f"{label} {key}={counts[key]}" for key in (*keys, *(("nonconforming",) if "name" in keys else ()))]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("f1", "f2", "camera", "top"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--audit", type=Path, help="f1: the identity directory's audit.jsonl")
    parser.add_argument("--keys", type=lambda s: tuple(s.split(",")), help="a subset of the command's counts")
    parser.add_argument("--explain", action="store_true", help="where each hit is, instead of the counts")
    args = parser.parse_args(argv)
    try:
        lines = run(args.command, args.directory, sys.stdin.buffer.read(), audit=args.audit, keys=args.keys,
                    explain=args.explain)
    except Refused as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
