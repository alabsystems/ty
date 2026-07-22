// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Re-exports from `tla-jit-abi` for the pure-data liveness predicate types.
//!
//! These types (`LivenessPredInfo`, `LivenessPredKind`, `ScalarCompOp`,
//! `LivenessCompileStats`, and the compiled-function pointer aliases) now live
//! in the `tla-jit-abi` leaf crate so trust-codegen and `tla-check` share a single
//! source of truth without creating a cargo cycle. Part of #4267.
//!
//! Existing `crate::runtime_abi::liveness_types::*` imports throughout
//! `tla-trust_cg` continue to resolve unchanged.

pub use tla_jit_abi::liveness_types::{
    CompiledAcceptanceCheckFn, CompiledActionPredBatchFn, CompiledStatePredBatchFn,
    LivenessCompileStats, LivenessPredInfo, LivenessPredKind, ScalarCompOp,
};
