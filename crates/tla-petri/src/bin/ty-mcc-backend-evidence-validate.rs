// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0
//
// mcc-keyword-guard: allow-spaced-mention
// (regression-fence tests live in the library module and construct
// spaced literals at runtime so an auto-fixer cannot rewrite them.)

//! Thin shim around [`tla_petri::mccctl_cmd::backend_evidence_validate`].
//!
//! All CLI parsing, evidence-row validation, and tests live in the
//! library module so `ty-mccctl validate` can invoke the same entry
//! point in-process without a subprocess hop.

use std::process::ExitCode;

fn main() -> ExitCode {
    tla_petri::mccctl_cmd::backend_evidence_validate::run()
}
