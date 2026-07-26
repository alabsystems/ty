#!/usr/bin/env python3
"""Reject ignored or permanently disabled coverage in TY-owned source.

The repository intentionally carries many near-verbatim third-party forks.
Their upstream target-, feature-, and Miri-specific ignores are not TY coverage
claims and remain outside this gate. ``FIRST_PARTY_ROOTS`` is the explicit
authority boundary for project-native Rust and Markdown documentation.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


FIRST_PARTY_ROOTS = (
    "crates/pnml-tools",
    "crates/tla-aiger",
    "crates/tla-ay",
    "crates/tla-backend",
    "crates/tla-bdd",
    "crates/tla-btor2",
    "crates/tla-cert",
    "crates/tla-check",
    "crates/tla-cli",
    "crates/tla-codegen",
    "crates/tla-core",
    "crates/tla-dd",
    "crates/tla-dialect",
    "crates/tla-eval",
    "crates/tla-gpu",
    "crates/tla-hw-evidence",
    "crates/tla-ir",
    "crates/tla-jit-abi",
    "crates/tla-lsp",
    "crates/tla-mc-core",
    "crates/tla-mdd",
    "crates/tla-petri",
    "crates/tla-resource",
    "crates/tla-runtime",
    "crates/tla-tir",
    "crates/tla-trust-cg",
    "crates/tla-value",
    "crates/tla-zenon",
    "fuzz",
)

_DIRECT_IGNORE = re.compile(r"^ignore(?:\s*=|\s*$)")
_EMPTY_CFG_ANY = re.compile(r"^cfg\s*\(\s*any\s*\(\s*\)\s*\)\s*$", re.S)
_CONDITIONAL_ATTRIBUTE = re.compile(r"^(?:cfg_attr|rustversion::attr)\s*\(")
_IGNORE_TOKEN = re.compile(r"\bignore(?:-[A-Za-z0-9_-]+)?\b")
_QUOTED_LITERAL = re.compile(r'''"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*' ''', re.S | re.X)
_DOC_STRING_ASSIGNMENT = re.compile(
    r'''\bdoc\s*=\s*"(?P<value>(?:\\.|[^"\\])*)"''',
    re.S,
)
_DOC_FENCE_IGNORE = re.compile(r"```[^\r\n`]*\bignore(?:-[A-Za-z0-9_-]+)?\b")
_RUSTDOC_IGNORE_FENCE = re.compile(
    r"""(?mx)
    ^[ \t]*//[/!][ \t]*
    ```(?=[^`\r\n]*\bignore(?P<target>-[A-Za-z0-9_-]+)?\b)
    [^`\r\n]*[ \t\r]*$
    """
)
_MARKDOWN_IGNORE_FENCE = re.compile(
    r"""(?mx)
    ^[ \t]*```(?=[^`\r\n]*\bignore(?:-[A-Za-z0-9_-]+)?\b)
    [^`\r\n]*[ \t\r]*$
    """
)
_DOCTEST_DISABLED = re.compile(r"(?m)^[ \t]*doctest[ \t]*=[ \t]*false[ \t]*(?:#.*)?$")


@dataclass(frozen=True, order=True)
class Finding:
    path: str
    line: int
    kind: str


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _bracket_delta(line: str) -> int:
    """Count attribute brackets outside ordinary quoted strings and comments."""
    delta = 0
    quote = None
    escaped = False
    index = 0
    while index < len(line):
        char = line[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            index += 1
            continue
        if char in {'"', "'"}:
            quote = char
        elif char == "/" and index + 1 < len(line) and line[index + 1] == "/":
            break
        elif char == "[":
            delta += 1
        elif char == "]":
            delta -= 1
        index += 1
    return delta


def _attribute_bodies(text: str) -> Iterable[tuple[int, str]]:
    """Yield ``(line, body)`` for source attributes in one linear pass."""
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        stripped = lines[index].lstrip()
        prefix_length = 3 if stripped.startswith("#![") else 2
        if not (stripped.startswith("#![") or stripped.startswith("#[")):
            index += 1
            continue

        start_line = index + 1
        chunks = [stripped[prefix_length:]]
        depth = _bracket_delta(stripped)
        index += 1
        while depth > 0 and index < len(lines):
            chunks.append(lines[index])
            depth += _bracket_delta(lines[index])
            index += 1
        if depth != 0:
            continue
        body = "\n".join(chunks)
        closing = body.rfind("]")
        if closing >= 0:
            body = body[:closing]
        yield start_line, body.strip()


def _rust_findings(relative: str, text: str) -> list[Finding]:
    findings = [
        Finding(
            relative,
            _line_number(text, match.start()),
            (
                "target_specific_ignored_rustdoc"
                if match.group("target")
                else "ignored_rustdoc"
            ),
        )
        for match in _RUSTDOC_IGNORE_FENCE.finditer(text)
    ]
    for line, body in _attribute_bodies(text):
        kind = None
        if _DIRECT_IGNORE.match(body):
            kind = "ignored_test"
        elif _EMPTY_CFG_ANY.match(body):
            kind = "permanently_disabled_cfg"
        elif _CONDITIONAL_ATTRIBUTE.match(body):
            doc_values = (
                match.group("value") for match in _DOC_STRING_ASSIGNMENT.finditer(body)
            )
            if any(_DOC_FENCE_IGNORE.search(value) for value in doc_values):
                kind = "conditional_rustdoc_ignore"
            elif _IGNORE_TOKEN.search(_QUOTED_LITERAL.sub("", body)):
                kind = "conditional_test_ignore"
        if kind is not None:
            findings.append(Finding(relative, line, kind))
    return findings


def _markdown_findings(relative: str, text: str) -> list[Finding]:
    return [
        Finding(relative, _line_number(text, match.start()), "ignored_markdown_example")
        for match in _MARKDOWN_IGNORE_FENCE.finditer(text)
    ]


def _manifest_findings(relative: str, text: str) -> list[Finding]:
    return [
        Finding(relative, _line_number(text, match.start()), "doctests_disabled")
        for match in _DOCTEST_DISABLED.finditer(text)
    ]


def scan_paths(root: Path, relative_roots: Iterable[str]) -> list[Finding]:
    findings: list[Finding] = []
    for relative_root in relative_roots:
        path = root / relative_root
        if not path.exists():
            raise ValueError(f"missing first-party source root: {relative_root}")
        for source in sorted(path.rglob("*")):
            if not source.is_file():
                continue
            relative = source.relative_to(root).as_posix()
            if source.suffix == ".rs":
                findings.extend(
                    _rust_findings(relative, source.read_text(encoding="utf-8"))
                )
            elif source.suffix == ".md":
                findings.extend(
                    _markdown_findings(relative, source.read_text(encoding="utf-8"))
                )
            elif source.name == "Cargo.toml":
                findings.extend(
                    _manifest_findings(relative, source.read_text(encoding="utf-8"))
                )
    return sorted(findings)


def check_repository(root: Path) -> list[Finding]:
    return scan_paths(root, FIRST_PARTY_ROOTS)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) > 1:
        print(f"usage: {Path(sys.argv[0]).name} [REPOSITORY_ROOT]", file=sys.stderr)
        return 2
    root = Path(args[0]).resolve() if args else Path(__file__).resolve().parent.parent
    try:
        findings = check_repository(root)
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"ERROR: first-party zero-ignore scan failed: {exc}", file=sys.stderr)
        return 2
    if findings:
        for finding in findings:
            print(
                f"ERROR: {finding.path}:{finding.line}: {finding.kind}",
                file=sys.stderr,
            )
        return 1
    print(
        "first_party_zero_ignores: clean "
        f"({len(FIRST_PARTY_ROOTS)} project-native roots)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
