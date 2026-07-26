#!/usr/bin/env python3
"""Regression tests for check_reproducible_git_pins.py."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("check_reproducible_git_pins.py")
SPEC = importlib.util.spec_from_file_location("check_reproducible_git_pins", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
pins = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pins
SPEC.loader.exec_module(pins)


def _manifest(name: str, git: str, selector: str) -> str:
    return f'[dev-dependencies.{name}]\ngit = "{git}"\n{selector}\n'


def _valid_tree(root: Path) -> None:
    package_blocks = []
    for pin in pins.PINS:
        selector = f'rev = "{pin.rev}"'
        for relative in pin.manifests:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_manifest(pin.name, pin.git, selector), encoding="utf-8")
        package_blocks.append(
            "[[package]]\n"
            f'name = "{pin.name}"\n'
            f'version = "{pin.version}"\n'
            f'source = "{pin.lock_source}"\n'
        )
    lock = "version = 3\n\n" + "\n".join(package_blocks)
    (root / "Cargo.lock").write_text(lock, encoding="utf-8")


class ReproducibleGitPinsTests(unittest.TestCase):
    def make_tree(self) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        _valid_tree(root)
        return temporary, root

    def test_exact_manifest_and_lock_pins_pass(self) -> None:
        temporary, root = self.make_tree()
        with temporary:
            self.assertEqual(pins.check_repository(root), [])

    def test_missing_revisions_fail_for_each_dependency(self) -> None:
        for pin in pins.PINS:
            with self.subTest(dependency=pin.name):
                temporary, root = self.make_tree()
                with temporary:
                    (root / pin.manifests[0]).write_text(
                        _manifest(pin.name, pin.git, ""), encoding="utf-8"
                    )
                    errors = pins.check_repository(root)
                    self.assertTrue(any("missing a rev selector" in e for e in errors), errors)

    def test_moving_selectors_fail_for_each_dependency(self) -> None:
        for pin in pins.PINS:
            with self.subTest(dependency=pin.name):
                temporary, root = self.make_tree()
                with temporary:
                    selector = f'branch = "main"\nrev = "{pin.rev}"'
                    (root / pin.manifests[0]).write_text(
                        _manifest(pin.name, pin.git, selector), encoding="utf-8"
                    )
                    errors = pins.check_repository(root)
                    self.assertTrue(any("moving selector" in e for e in errors), errors)

    def test_short_revisions_fail_for_each_dependency(self) -> None:
        for pin in pins.PINS:
            with self.subTest(dependency=pin.name):
                temporary, root = self.make_tree()
                with temporary:
                    (root / pin.manifests[0]).write_text(
                        _manifest(pin.name, pin.git, f'rev = "{pin.rev[:7]}"'),
                        encoding="utf-8",
                    )
                    errors = pins.check_repository(root)
                    self.assertTrue(any("not a full lowercase 40-hex" in e for e in errors), errors)

    def test_source_manifests_are_also_guarded(self) -> None:
        temporary, root = self.make_tree()
        with temporary:
            pin = pins.PINS[0]
            (root / pin.manifests[1]).write_text(
                _manifest(pin.name, pin.git, f'rev = "{pin.rev[:7]}"'), encoding="utf-8"
            )
            errors = pins.check_repository(root)
            self.assertTrue(any(pin.manifests[1] in e for e in errors), errors)

    def test_lock_query_and_resolved_commit_are_bound_for_each_dependency(self) -> None:
        for pin in pins.PINS:
            with self.subTest(dependency=pin.name):
                temporary, root = self.make_tree()
                with temporary:
                    lock = root / "Cargo.lock"
                    text = lock.read_text(encoding="utf-8")
                    text = text.replace(pin.lock_source, f"{pin.lock_source_base}#{'0' * 40}", 1)
                    lock.write_text(text, encoding="utf-8")
                    errors = pins.check_repository(root)
                    self.assertTrue(any("source must be" in e for e in errors), errors)


if __name__ == "__main__":
    unittest.main(verbosity=2)
