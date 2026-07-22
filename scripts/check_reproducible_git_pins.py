#!/usr/bin/env python3
"""Fail closed if TY's vendored Git dependencies are not immutable.

Both the generated vendored manifests and their Cargo.toml.orig sources must
name the same full commit. Cargo.lock must bind that selector to the identical
resolved commit, rather than merely recording whatever a branch resolved to.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9/3.10
    try:
        import tomli as tomllib
    except ModuleNotFoundError as exc:  # pragma: no cover - environment error
        raise SystemExit("ERROR: Python 3.11+ or the 'tomli' package is required") from exc


FULL_REV = re.compile(r"[0-9a-f]{40}\Z")


@dataclass(frozen=True)
class GitPin:
    name: str
    version: str
    git: str
    rev: str
    manifests: tuple[str, ...]

    @property
    def lock_source_base(self) -> str:
        return f"git+{self.git}?rev={self.rev}"

    @property
    def lock_source(self) -> str:
        return f"{self.lock_source_base}#{self.rev}"


PINS = (
    GitPin(
        name="proptest",
        version="1.0.0",
        git="https://github.com/input-output-hk/proptest",
        rev="270ea99959dc4c4e5076047d79a9654f5f0ac74a",
        manifests=(
            "crates/tla-unarray/Cargo.toml",
            "crates/tla-unarray/Cargo.toml.orig",
        ),
    ),
    GitPin(
        name="test-helper",
        version="0.0.0",
        git="https://github.com/taiki-e/test-helper.git",
        rev="e2e8e3763cdd76b13eb28097877a12ee6f86a08e",
        manifests=(
            "crates/tla-portable-atomic/Cargo.toml",
            "crates/tla-portable-atomic/Cargo.toml.orig",
        ),
    ),
)


def _load_toml(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("rb") as handle:
            value = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"{path}: cannot read TOML: {exc}") from exc
    if not isinstance(value, Mapping):  # pragma: no cover - TOML roots are tables
        raise ValueError(f"{path}: TOML root is not a table")
    return value


def _dependency(data: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    for table_name in ("dependencies", "dev-dependencies", "build-dependencies"):
        table = data.get(table_name)
        if isinstance(table, Mapping) and name in table:
            value = table[name]
            return value if isinstance(value, Mapping) else None
    return None


def _check_manifest(root: Path, relative: str, pin: GitPin) -> list[str]:
    path = root / relative
    try:
        data = _load_toml(path)
    except ValueError as exc:
        return [str(exc)]
    dep = _dependency(data, pin.name)
    if dep is None:
        return [f"{relative}: missing table dependency {pin.name!r}"]

    errors: list[str] = []
    git = dep.get("git")
    rev = dep.get("rev")
    moving = sorted(key for key in ("branch", "tag") if key in dep)
    if git != pin.git:
        errors.append(f"{relative}: {pin.name} git must be {pin.git!r}, found {git!r}")
    if moving:
        errors.append(
            f"{relative}: {pin.name} uses moving selector(s): {', '.join(moving)}"
        )
    if not isinstance(rev, str):
        errors.append(f"{relative}: {pin.name} is missing a rev selector")
    elif not FULL_REV.fullmatch(rev):
        errors.append(f"{relative}: {pin.name} rev is not a full lowercase 40-hex commit: {rev!r}")
    elif rev != pin.rev:
        errors.append(f"{relative}: {pin.name} rev must be {pin.rev}, found {rev}")
    return errors


def _check_lock(root: Path, pin: GitPin, lock: Mapping[str, Any]) -> list[str]:
    packages = lock.get("package")
    if not isinstance(packages, list):
        return ["Cargo.lock: missing package array"]
    candidates = [
        item
        for item in packages
        if isinstance(item, Mapping)
        and item.get("name") == pin.name
        and item.get("version") == pin.version
        and isinstance(item.get("source"), str)
        and str(item["source"]).startswith(f"git+{pin.git}")
    ]
    if len(candidates) != 1:
        return [
            f"Cargo.lock: expected exactly one Git {pin.name} {pin.version} entry, "
            f"found {len(candidates)}"
        ]

    errors: list[str] = []
    source = candidates[0].get("source")
    if source != pin.lock_source:
        errors.append(
            f"Cargo.lock: {pin.name} source must be {pin.lock_source!r}, found {source!r}"
        )

    metadata = lock.get("metadata")
    checksum_key = f"checksum {pin.name} {pin.version} ({pin.lock_source_base})"
    if not isinstance(metadata, Mapping) or metadata.get(checksum_key) != "<none>":
        errors.append(f"Cargo.lock: missing exact metadata key {checksum_key!r}")
    return errors


def check_repository(root: Path) -> list[str]:
    errors: list[str] = []
    try:
        lock = _load_toml(root / "Cargo.lock")
    except ValueError as exc:
        return [str(exc)]
    for pin in PINS:
        for manifest in pin.manifests:
            errors.extend(_check_manifest(root, manifest, pin))
        errors.extend(_check_lock(root, pin, lock))
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) > 1:
        print(f"usage: {Path(sys.argv[0]).name} [REPOSITORY_ROOT]", file=sys.stderr)
        return 2
    root = Path(args[0]).resolve() if args else Path(__file__).resolve().parent.parent
    errors = check_repository(root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"reproducible_git_pins: clean ({len(PINS)} exact Git commits)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
