// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Thin shim around [`tla_petri::mccctl_cmd::drift_guard`].
//!
//! All logic, CLI parsing, and tests live in the library module so
//! `ty-mccctl drift-guard` can invoke the same entry point in-process
//! without a subprocess hop.

use std::process::ExitCode;

fn main() -> ExitCode {
    tla_petri::mccctl_cmd::drift_guard::run()
}
