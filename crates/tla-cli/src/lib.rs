// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Command-line interface for TY, a modern TLA+ verification toolchain.
//!
//! `tla-cli` is the home of the `ty` binary — the primary user-facing entry
//! point to the TY ecosystem, offering subcommands for model checking, parsing,
//! formatting, simulation, code generation, and related tooling. The `tla`
//! binary is a legacy alias for the same executable.
//!
//! # This library target is intentionally empty
//!
//! All of the CLI's logic lives in the **binary** target (`src/main.rs` and the
//! `cmd_*` modules it declares), not in this library. The crate exposes no
//! reusable API: the `cmd_*` modules are private to the binary and are not
//! re-exported here. This `lib.rs` exists only so the crate has a library target
//! for documentation and metadata purposes; consume TY's functionality through
//! the workspace library crates (`tla-core`, `tla-check`, `tla-backend`, …)
//! rather than this crate. See the binary crate documentation in `main.rs` for
//! the runtime architecture, process setup, and exit-code contract.
//!
//! # Commands
//!
//! The binary supports over 150 subcommands, curated into eight visible
//! groups (see `catalog.rs`, which also enforces the grouping with a
//! partition test):
//!
//! - **Author** — `init`, `parse`, `fmt`, `lint`, `typecheck`, `refactor`, `lsp`
//! - **Check** — `check` (the TLC replacement), `watch`, `test`, `simulate`, `explore`
//! - **Diagnose** — `explain`, `trace`, `graph`, `repair`, `minimize`
//! - **Prove & certify** — `prove`, `certify`, `refine`, `recheck`, `selfcheck`
//! - **Hardware & Petri nets** — `aiger`, `btor2`, `petri`, `mcc`
//! - **Export** — `codegen`, `vmt`, `convert`
//! - **Benchmark** — `bench`, `profile`
//! - **Toolchain** — `commands` (the full catalog), `completions`, `corpus`,
//!   installers, `cache`
//!
//! The remaining specialist/diagnostic commands are hidden from `--help` but
//! stay callable; `ty commands` lists everything, grouped.
//!
//! # Crate relationships
//!
//! `tla-cli` delegates to `tla-check` for model checking, `tla-core` for
//! parsing/lowering/resolution, `tla-backend` for engine selection, and
//! `tla-codegen`/`tla-petri`/`tla-lsp` for the corresponding subcommands.
//! Output formatting (human, JSON, JSONL, ITF, TLC-tool) is handled in the
//! binary's `check_report` and `cli_schema` modules.
