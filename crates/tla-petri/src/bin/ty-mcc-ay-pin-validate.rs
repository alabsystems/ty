// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0
//
// mcc-keyword-guard: allow-spaced-mention
// (this binary emits no MCC stdout; comment present so the keyword guard is
// satisfied for any future tests that build spaced literals at runtime.)

//! Thin shim around [`tla_petri::mccctl_cmd::ay_pin_validate`].
//!
//! All CLI parsing and validation logic lives in the library module so
//! `ty-mccctl ay-pin-validate` can invoke the same entry point
//! in-process without a subprocess hop.

use std::process::ExitCode;

fn main() -> ExitCode {
    tla_petri::mccctl_cmd::ay_pin_validate::run()
}
