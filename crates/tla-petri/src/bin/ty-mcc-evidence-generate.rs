// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0
//
// mcc-keyword-guard: allow-spaced-mention
// (this binary emits no MCC stdout; comment present so a future spaced-
// literal regression test in the library module cannot be silently
// rewritten.)

//! Thin shim around [`tla_petri::mccctl_cmd::evidence_generate`].
//!
//! All CLI parsing, JSONL generation logic, and unit tests live in the
//! library module so `ty-mccctl evidence-generate` can invoke the same
//! entry point in-process without a subprocess hop.

use std::process::ExitCode;

fn main() -> ExitCode {
    tla_petri::mccctl_cmd::evidence_generate::run()
}
