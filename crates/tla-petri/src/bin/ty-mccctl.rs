// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Operator CLI for TY MCC smoke, benchmark, and submission workflows.

fn main() -> anyhow::Result<()> {
    tla_petri::mccctl::main_entry()
}
