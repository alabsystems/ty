// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `tla` — legacy-named alias of the `ty` command-line binary.
//!
//! This target compiles the exact same program as `ty`: it `include!`s
//! `src/main.rs` verbatim, so both binaries share one entry point, argument
//! schema, and subcommand set. The separate file exists only to avoid Cargo's
//! "file found in multiple build targets" warning when two `[[bin]]` entries
//! would otherwise share one source path. Prefer the `ty` name in new usage.
//!
//! # Architecture
//!
//! The entry point (`main`) is deliberately thin: it selects a global allocator
//! at compile time (`mimalloc` by default, or `dhat` under the `dhat-heap`
//! feature), then runs `async_main` on a freshly spawned thread with a **64 MiB
//! stack** — the enlarged stack is required because deeply nested or recursive
//! TLA+ expressions can overflow the default thread stack during evaluation.
//! `async_main` parses the command line into the `Cli` struct and dispatches
//! each `Command` variant to its `cmd_*` module. Each subcommand lives in its
//! own private module (`cmd_check`, `cmd_simulate`, `trace_cmd`, …) and owns its
//! arguments, reporting, and exit-code policy.
//!
//! Heavy lifting is delegated to the workspace library crates: `tla-core`
//! (parsing, lowering, name resolution), `tla-check` (explicit-state model
//! checking — the TLC replacement — plus BMC/PDR and the interactive
//! exploration server), `tla-backend` (interpreter vs. native trust-cg engine
//! selection), and `tla-codegen`/`tla-petri`/`tla-lsp` for the corresponding
//! subcommands. Output formatting (human, JSON, JSONL, ITF, TLC-tool) lives in
//! the `check_report` and `cli_schema` modules.
//!
//! # Exit codes
//!
//! Subcommands follow the conventional shell contract: `0` on success (e.g. a
//! verified spec), `1` on a checking failure or violation (e.g. an invariant
//! counterexample), and `2` for usage errors reported by `clap`. Some commands
//! reserve additional codes — notably `ty check` exits `3` with a JSON
//! `backend_unavailable` sentinel when a requested native backend cannot run.
include!("../main.rs");
