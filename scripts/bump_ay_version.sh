#!/usr/bin/env bash
# Copyright 2026 Andrew Yates.
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

# bump_ay_version.sh - Update ay dependency pin to latest HEAD
#
# Usage:
#   ./scripts/bump_ay_version.sh           # bump to latest HEAD
#   ./scripts/bump_ay_version.sh <commit>  # bump to specific commit

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AY_REPO="https://github.com/alabsystems/ay"
CARGO_TOML="$REPO_ROOT/Cargo.toml"

# Get target commit
if [[ $# -ge 1 ]]; then
    NEW_REV="$1"
    echo "Bumping ay to specified commit: $NEW_REV"
else
    echo "Fetching latest ay HEAD..."
    NEW_REV=$(git ls-remote "$AY_REPO" HEAD | cut -f1)
    if [[ -z "$NEW_REV" ]]; then
        echo "ERROR: Failed to fetch ay HEAD" >&2
        exit 1
    fi
    echo "Latest ay commit: $NEW_REV"
fi

if [[ ! "$NEW_REV" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: expected ay rev to be a 40-hex git commit, got: $NEW_REV" >&2
    exit 2
fi

if [[ ! -f "$CARGO_TOML" ]]; then
    echo "ERROR: missing $CARGO_TOML" >&2
    exit 1
fi

echo "Updating Cargo.toml (workspace.dependencies)..."
IFS=$'\t' read -r CURRENT_REV ACTION AY_DEPS_CSV < <(
    python3 - "$CARGO_TOML" "$NEW_REV" <<'PY'
import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.11+ in normal use
    import tomli as tomllib

cargo_toml = Path(sys.argv[1])
new_rev = sys.argv[2]
canonical_repo = "https://github.com/alabsystems/ay"
allowed_repo_ids = set([
    "github.com/alabsystems/ay",
])

text = cargo_toml.read_text(encoding="utf-8")
data = tomllib.loads(text)
workspace_deps = data.get("workspace", dict()).get("dependencies", dict())
if not isinstance(workspace_deps, dict):
    raise SystemExit("ERROR: workspace.dependencies missing from Cargo.toml")

def normalize_repo_identity(url: str) -> str | None:
    base = re.split(r"[?#]", url, maxsplit=1)[0]
    patterns = (
        r"^https://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$",
        r"^ssh://git@github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$",
        r"^git@github\.com:([^/]+)/([^/]+?)(?:\.git)?/?$",
    )
    for pattern in patterns:
        match = re.match(pattern, base)
        if match:
            owner, repo = match.groups()
            return f"github.com/{owner}/{repo}"
    return None

def is_ay_dep(name: str) -> bool:
    return name == "ay" or name.startswith("ay-")

deps: dict[str, dict[str, str]] = dict()
bad_specs = []
bad_urls = []
invalid_revs = []
needs_url_canonicalization = False
for name, spec in workspace_deps.items():
    if not is_ay_dep(name):
        continue
    if not isinstance(spec, dict):
        bad_specs.append(name)
        continue
    git_url = spec.get("git")
    rev = spec.get("rev")
    if not isinstance(git_url, str):
        bad_specs.append(name)
        continue
    repo_id = normalize_repo_identity(git_url)
    if repo_id not in allowed_repo_ids:
        bad_urls.append(f"{name}={git_url!r}")
    if git_url != canonical_repo:
        needs_url_canonicalization = True
    if not isinstance(rev, str) or re.fullmatch(r"[0-9a-f]{40}", rev) is None:
        invalid_revs.append(f"{name}={rev!r}")
        continue
    deps[name] = dict(git=git_url, rev=rev)

if bad_specs:
    raise SystemExit(f"ERROR: invalid ay dependency table(s): {', '.join(sorted(bad_specs))}")
if bad_urls:
    raise SystemExit(f"ERROR: unsupported ay git url(s): {', '.join(sorted(bad_urls))}")
if invalid_revs:
    raise SystemExit(f"ERROR: invalid ay rev(s): {', '.join(sorted(invalid_revs))}")
if not deps:
    raise SystemExit("ERROR: missing workspace ay deps in Cargo.toml")

dep_names = ",".join(sorted(deps))
revs = {spec["rev"] for spec in deps.values()}
if len(revs) != 1:
    detail = ", ".join(f"{name}={spec['rev'][:8]}" for name, spec in sorted(deps.items()))
    raise SystemExit(f"ERROR: mismatched ay revs in Cargo.toml: {detail}")

(current_rev,) = tuple(revs)

if current_rev == new_rev and not needs_url_canonicalization:
    print(f"{current_rev}\tnoop\t{dep_names}")
    raise SystemExit(0)

def rewrite_dep_line(source: str, dep_name: str) -> str:
    line_pat = re.compile(
        rf'^(?P<prefix>{re.escape(dep_name)}\s*=\s*\{{)(?P<body>[^}}]*)(?P<suffix>\}}\s*(?:#.*)?)$',
        re.MULTILINE,
    )

    def repl(match: re.Match[str]) -> str:
        body = match.group("body")
        if re.search(r'\bgit\s*=\s*"[^"]+"', body) is None:
            raise SystemExit(f"ERROR: missing git url for {dep_name} in Cargo.toml")
        if re.search(r'\brev\s*=\s*"[0-9a-f]{40}"', body) is None:
            raise SystemExit(f"ERROR: missing git rev for {dep_name} in Cargo.toml")
        body = re.sub(r'(\bgit\s*=\s*")[^"]+(")', rf'\g<1>{canonical_repo}\2', body, count=1)
        body = re.sub(r'(\brev\s*=\s*")[0-9a-f]{40}(")', rf'\g<1>{new_rev}\2', body, count=1)
        return f"{match.group('prefix')}{body}{match.group('suffix')}"

    updated_text, count = line_pat.subn(repl, source, count=1)
    if count != 1:
        raise SystemExit(f"ERROR: unable to rewrite {dep_name} in Cargo.toml")
    return updated_text

updated = text
for dep_name in sorted(deps):
    updated = rewrite_dep_line(updated, dep_name)
cargo_toml.write_text(updated, encoding="utf-8")
print(f"{current_rev}\tupdated\t{dep_names}")
PY
)

echo "Current ay pin: $CURRENT_REV"

if [[ "${ACTION:-}" == "noop" ]]; then
    echo "Already at target revision. No changes needed."
    exit 0
fi

# Refresh Cargo.lock
echo "Refreshing Cargo.lock..."
cd "$REPO_ROOT"
rm -f "$REPO_ROOT/Cargo.lock"
IFS=',' read -r -a AY_DEPS <<< "${AY_DEPS_CSV:-}"
for dep in "${AY_DEPS[@]}"; do
    cargo update -p "$dep" --precise "$NEW_REV"
done

echo ""
echo "Done! ay bumped: $CURRENT_REV -> $NEW_REV"
echo ""
echo "Next steps:"
echo "  1. cargo check -p tla-ay"
echo "  2. cargo test -p tla-ay"
echo "  3. scripts/determinism_gate.sh   # TwoPhase x3 run-to-run gate (deterministic ay inprocessing)"
echo "  4. Run ay-related regression tests"
echo "  5. Commit with a message referencing the ay bump"
