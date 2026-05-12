#!/usr/bin/env python3
"""Ensure a Git release tag vX.Y.Z matches project.version in pyproject.toml.

PyPI sdist uses pyproject.toml as the single source of truth; this script
refuses a release when the published tag and that version disagree.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"


def read_pyproject_version() -> str:
    text = PYPROJECT.read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not m:
        print("ERROR: could not parse version from pyproject.toml", file=sys.stderr)
        sys.exit(2)
    return m.group(1).strip()


def normalize_tag(tag: str) -> str:
    tag = tag.strip()
    if tag.startswith("refs/tags/"):
        tag = tag[len("refs/tags/") :]
    if not tag.startswith("v"):
        print(f"ERROR: tag must start with 'v', got {tag!r}", file=sys.stderr)
        sys.exit(2)
    return tag[1:]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify v-tag matches pyproject.toml version (PyPI/git sync guard)."
    )
    parser.add_argument(
        "tag",
        nargs="?",
        default=os.environ.get("GITHUB_REF_NAME", ""),
        help="e.g. v0.2.1 (default: GITHUB_REF_NAME)",
    )
    args = parser.parse_args()
    tag = (args.tag or "").strip()
    if not tag:
        print("ERROR: pass tag (e.g. v0.2.1) or set GITHUB_REF_NAME", file=sys.stderr)
        sys.exit(2)

    tag_version = normalize_tag(tag)
    py_version = read_pyproject_version()
    if tag_version != py_version:
        print(
            f"ERROR: Git tag {tag!r} implies version {tag_version!r} but "
            f"pyproject.toml has {py_version!r}.",
            file=sys.stderr,
        )
        print(
            f"       Align them: pyproject.toml version should be {tag_version!r} with tag {tag!r}, "
            f"or use Git tag v{py_version!r} with the current pyproject.toml.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"OK: tag {tag} matches pyproject.toml version {py_version}")


if __name__ == "__main__":
    main()
