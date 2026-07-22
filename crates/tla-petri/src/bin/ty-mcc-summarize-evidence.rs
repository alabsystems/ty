// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0
//
// mcc-keyword-guard: allow-spaced-mention
// (the regression-fence tests live in the library module and construct
// legacy spaced literals at runtime so an auto-fixer cannot rewrite
// them; production sources only emit the canonical underscored forms via
// `tla_petri::mcc_keywords`.)

//! Thin shim around [`tla_petri::mccctl_cmd::summarize_evidence`].
//!
//! All CLI parsing, JSONL aggregation, and tests live in the library
//! module so `ty-mccctl summarize-evidence` can invoke the same entry
//! point in-process without a subprocess hop.

use std::process::ExitCode;

fn main() -> ExitCode {
    tla_petri::mccctl_cmd::summarize_evidence::run()
}
