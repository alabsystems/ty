#!/usr/bin/env python3
"""Regression tests for check_first_party_zero_ignores.py."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("check_first_party_zero_ignores.py")
SPEC = importlib.util.spec_from_file_location("check_first_party_zero_ignores", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
zero_ignores = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = zero_ignores
SPEC.loader.exec_module(zero_ignores)


class FirstPartyZeroIgnoresTests(unittest.TestCase):
    def scan(self, rust: str = "", markdown: str = "", manifest: str = ""):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        source_root = root / "crates" / "tla-check"
        source_root.mkdir(parents=True)
        if rust:
            (source_root / "lib.rs").write_text(rust, encoding="utf-8")
        if markdown:
            (source_root / "README.md").write_text(markdown, encoding="utf-8")
        if manifest:
            (source_root / "Cargo.toml").write_text(manifest, encoding="utf-8")
        findings = zero_ignores.scan_paths(root, ("crates/tla-check",))
        return temporary, findings

    def test_compiling_and_prose_examples_are_clean(self) -> None:
        temporary, findings = self.scan(
            rust="//! ```rust,no_run\n//! fn main() {}\n//! ```\n//! ```text\n//! sketch\n//! ```\n"
        )
        with temporary:
            self.assertEqual(findings, [])

    def test_direct_and_conditional_test_ignores_fail(self) -> None:
        temporary, findings = self.scan(
            rust=(
                "#[test]\n#[ignore = \"manual\"]\nfn direct() {}\n"
                "#[test]\n#[cfg_attr(\n    miri,\n    ignore = \"slow\"\n)]\n"
                "fn conditional() {}\n"
            )
        )
        with temporary:
            self.assertEqual(
                [finding.kind for finding in findings],
                ["ignored_test", "conditional_test_ignore"],
            )

    def test_empty_cfg_any_fails_without_rejecting_real_cfg_any(self) -> None:
        temporary, findings = self.scan(
            rust=(
                "#[cfg(any())]\nfn permanently_disabled() {}\n"
                "#[cfg(any(\n    target_os = \"linux\",\n    test,\n))]\n"
                "fn conditionally_enabled() {}\n"
            )
        )
        with temporary:
            self.assertEqual(
                [finding.kind for finding in findings],
                ["permanently_disabled_cfg"],
            )

    def test_literal_and_injected_rustdoc_ignores_fail(self) -> None:
        temporary, findings = self.scan(
            rust=(
                "//! ```rust,ignore\n//! hidden setup\n//! ```\n"
                "#![cfg_attr(not(feature = \"full\"), doc = \"```rust,ignore\")]\n"
            )
        )
        with temporary:
            self.assertEqual(
                [finding.kind for finding in findings],
                ["ignored_rustdoc", "conditional_rustdoc_ignore"],
            )

    def test_markdown_ignores_and_disabled_doctests_fail(self) -> None:
        temporary, findings = self.scan(
            markdown="```rust,ignore\nexample\n```\n",
            manifest="[lib]\ndoctest = false\n",
        )
        with temporary:
            self.assertEqual(
                [finding.kind for finding in findings],
                ["doctests_disabled", "ignored_markdown_example"],
            )

    def test_comments_and_non_ignore_words_do_not_fail(self) -> None:
        temporary, findings = self.scan(
            rust=(
                "// #[ignore]\n"
                "const NOTE: &str = \"cfg_attr(miri, ignore)\";\n"
                "#[cfg_attr(miri, allow(dead_code))]\n"
                "fn active() {}\n"
            )
        )
        with temporary:
            self.assertEqual(findings, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
