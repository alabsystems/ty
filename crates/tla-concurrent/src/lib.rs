// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Thread concurrency verification for TY.
//!
//! This crate translates a [`ConcurrentModel`] IR (produced by tRust's MIR
//! extraction) into a TLA+ `Module` AST, feeds it through the TY model
//! checker, and returns structured verification results with source-mapped
//! counterexamples.
//!
//! # Architecture
//!
//! ```text
//! Rust source → tRust MIR extraction → ConcurrentModel (JSON)
//!     → tla-concurrent → TLA+ Module AST → tla-check → CheckResult
//!     → source-mapped counterexample (JSON)
//! ```
//!
//! # What this crate provides
//!
//! - The IR contract between extractor and checker: [`ConcurrentModel`] and its
//!   constituents ([`Process`], [`SharedVar`], [`Transition`],
//!   [`SyncPrimitive`], [`Property`], [`Assumptions`]).
//! - [`generate`]: translation of the IR into a programmatic TLA+ `Module`.
//! - [`check_concurrent_model`]: the end-to-end entry point — validate,
//!   generate, model-check, and source-map (requires the `check` feature).
//! - [`source_map`]: mapping abstract TLA+ counterexamples back to concrete
//!   Rust source spans for human-readable diagnostics.
//!
//! Every result carries the [`Assumptions`] under which it was obtained, so the
//! output is always "verified under these assumptions," never "verified."
//!
//! # Feature flags
//!
//! - `check` (default): enables [`check_concurrent_model`] and the
//!   `tla-check`-backed model-checking path. Without it, the crate is reduced
//!   to the IR types, TLA+ generation, and source-mapping helpers.

#![deny(missing_docs)]

pub mod assumptions;
pub mod generate;
pub mod model;
pub mod property;
pub mod source_map;
pub mod sync_kind;
pub mod transition;

#[cfg(feature = "check")]
mod check;
#[cfg(feature = "check")]
mod output;
#[cfg(feature = "check")]
mod trace_mapper;

pub use assumptions::{
    Assumptions, DataAbstraction, DynDispatchResolution, FairnessMode, MemoryModel, PanicStrategy,
    Reduction,
};
#[cfg(feature = "check")]
pub use check::{check_concurrent_model, CheckOptions, ConcurrentCheckResult, ConcurrentError};
pub use model::{
    AccessKind, AccessSite, ConcurrentModel, GuardMode, HeldGuard, LocalVar, Process, ProcessId,
    ProcessKind, SharedVar, StateId, SyncId,
};
#[cfg(feature = "check")]
pub use output::{CheckerStats, VerificationReport, VerificationStatus};
pub use property::Property;
pub use source_map::{MappedStep, MappedTrace, SourceMap, SourceMapEntry, SourceSpan};
pub use sync_kind::{ChannelKind, SyncKind, SyncPrimitive};
pub use transition::{AtomicOp, CasInfo, MemoryOrdering, PanicGuard, Transition, TransitionKind};
