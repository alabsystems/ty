// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0
//
// mcc-keyword-guard: allow-spaced-mention
// (regression-fence tests in the library module construct the round-1
// spaced literals at runtime so an auto-fixer cannot rewrite them into
// tautologies.)

//! Thin shim around [`tla_petri::mccctl_cmd::validate`].
//!
//! All CLI parsing, stdout parsing, and validation logic (including the
//! spaced-keyword regression fence) live in the library module so
//! `ty-mccctl spec-validate` can invoke the same entry point
//! in-process without a subprocess hop.

use std::process::ExitCode;

fn main() -> ExitCode {
    tla_petri::mccctl_cmd::validate::run()
}
