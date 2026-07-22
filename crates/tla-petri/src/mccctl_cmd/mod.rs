// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Library modules backing every `ty-mcc-*` operator helper binary.
//!
//! The 9 historical helper binaries (backend-evidence-validate, drift-guard,
//! evidence-generate, history, smoke, summarize-evidence, sweep, validate,
//! ay-pin-validate) each had a clap CLI defined inside `src/bin/<name>.rs`.
//! That meant the in-tree binary was the ONLY consumer of the entry point —
//! `ty-mccctl` and tla-cli integration tests could not call those entry
//! points without a subprocess hop.
//!
//! Moving each binary's `Cli` struct and `run()` function into a library
//! module here gives us a single compiler-enforced surface: `ty-mccctl`
//! adds subcommands that delegate to these `run_from(...)` functions
//! in-process, BenchKit and pre-commit hooks keep their old binary names
//! via thin 3-line shims in `src/bin/`, and the tla-cli tests can drive
//! the same logic without rebuilding multiple Cargo binaries.
//!
//! Each sub-module exports two entry points:
//!
//! * `pub fn run() -> ExitCode` — parses from `std::env::args_os()`. This
//!   is the entry point used by the corresponding `src/bin/<name>.rs`
//!   shim binary.
//! * `pub fn run_from<I, T>(args: I) -> ExitCode` — parses from a caller-
//!   supplied iterator. This is what `ty-mccctl <subcommand>` uses to
//!   forward its tail arguments without shelling out.

pub mod ay_pin_validate;
pub mod backend_evidence_validate;
pub mod dhat_summary;
pub mod drift_guard;
pub mod evidence_generate;
pub mod fetch;
pub mod history;
pub mod smoke;
pub mod summarize_evidence;
pub mod sweep;
pub mod symmetry_bench;
pub mod test_results_compare;
pub mod validate;
