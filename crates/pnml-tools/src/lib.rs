// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Compatibility facade for the legacy `pnml-tools` crate name.
//!
//! `ty`'s Petri-net / Model-Checking-Contest (MCC) engine — PNML parsing,
//! state-space exploration, and MCC `FORMULA` output — now lives canonically
//! in the [`tla_petri`] crate and is surfaced in the unified `ty` CLI via
//! `ty petri` and `ty mcc`. This crate exists purely as a stable alias under
//! the historical `pnml-tools` package name so that external scripts, the MCC
//! competition harness, and downstream code that referenced `pnml_tools::*`
//! keep compiling and resolving to the exact same implementation.
//!
//! # What this crate provides
//!
//! - A single glob re-export, [`pub use tla_petri::*`](tla_petri), so every
//!   public path under `pnml_tools::` (e.g. [`model::load_model_dir`],
//!   [`examination::Examination`], [`cli::run_cli`],
//!   [`simplification_report`]) is the corresponding item in `tla_petri`.
//!   There is no separately-maintained code here; consult the [`tla_petri`]
//!   crate documentation for the API reference, supported MCC examinations,
//!   and exploration techniques.
//! - Two binaries:
//!   - `pnml-tools` — the legacy MCC entry point. It reads a model directory
//!     (or the `BK_INPUT` environment variable) and an examination name (or
//!     `BK_EXAMINATION`), then dispatches to [`cli::run_cli`] in
//!     [`cli::PetriCommandMode::Mcc`]. The real work runs on a worker thread
//!     with a 64 MiB stack to tolerate deeply recursive CTL/LTL evaluation.
//!   - `pnml-simplify-report` — emits a JSON report of the impact of formula
//!     simplification for one model/examination, via
//!     [`model::collect_simplification_report_for_model`].
//!
//! # Where to look instead
//!
//! New code should depend on [`tla_petri`] directly and drive the engine
//! through the `ty` CLI. This facade is retained only for backwards
//! compatibility and adds no behaviour of its own.

#![deny(missing_docs)]

pub use tla_petri::*;
