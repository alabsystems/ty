// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Runtime ABI types shared between the trust-codegen codegen pipeline and the model
//! checker. The active native runtime surface lives directly inside
//! `tla-trust_cg`, with pure call-boundary data re-exported from `tla-jit-abi`.
//!
//! Surface:
//!
//! - **ABI** — `JitCallOut`, `JitStatus`, `JitRuntimeErrorKind`, function
//!   pointer types, `extern "C"` runtime helpers for compound operations,
//!   compound scratch buffer
//! - **Flat state** — `FlatState`, `FlatStateSchema`, `VarEncoding`
//! - **Compound layout** — `CompoundLayout`, `StateLayout`, `VarLayout`
//! - **Fingerprint set** — `AtomicFpSet`, `ResizableAtomicFpSet`
//! - **Fingerprint functions** — seeded compiled-domain fingerprinting plus
//!   non-admissible legacy ABI compatibility
//! - **Liveness predicate descriptors** — `LivenessPredInfo`, `LivenessPredKind`,
//!   `ScalarCompOp`, `LivenessCompileStats`, compiled function-pointer types

pub mod abi;
pub mod atomic_fp_set;
pub mod compound_layout;
pub mod compound_read;
pub mod error;
pub mod fingerprint;
pub mod flat_state;
pub mod liveness_types;
pub mod tla_ops;

pub use abi::{
    clear_compound_scratch, compound_scratch_guard, jit_compound_count, jit_func_apply_i64,
    jit_func_set_membership_check, jit_pow_i64, jit_record_get_i64, jit_record_new_scalar,
    jit_seq_append, jit_seq_get_i64, jit_seq_tail, jit_set_contains_i64, jit_set_diff_i64,
    jit_set_union_i64, read_compound_scratch, CompoundScratchGuard, JitCallOut, JitExprFn, JitFn0,
    JitInvariantFn, JitNextStateFn, JitRuntimeErrorKind, JitStatus, COMPOUND_SCRATCH_BASE,
};
pub use atomic_fp_set::{
    atomic_fp_set_probe, resizable_fp_set_probe, single_thread_fp_set_probe, AtomicFpSet,
    CumulativeProbeStats, InsertResult, ProbeStats, ResizableAtomicFpSet, SingleThreadFpSet,
};
pub use compound_layout::{
    deserialize_value, infer_layout, infer_var_layout, serialize_value, CompoundLayout,
    StateLayout, VarLayout, TAG_BOOL, TAG_FUNC, TAG_INT, TAG_RECORD, TAG_SEQ, TAG_SET, TAG_STRING,
    TAG_TUPLE,
};
pub use error::JitRuntimeError;
pub use fingerprint::{
    admissible_flat_state_fingerprint_64, jit_xxh3_fingerprint_64, ty_compiled_fp_u64,
    LEGACY_JIT_XXH3_FINGERPRINT_64_ADMISSIBLE,
};
pub use flat_state::{
    flat_to_state, flat_to_state_from_slice, state_to_flat, state_to_flat_reuse, FlatState,
    FlatStateSchema, VarEncoding,
};
pub use liveness_types::{
    CompiledAcceptanceCheckFn, CompiledActionPredBatchFn, CompiledStatePredBatchFn,
    LivenessCompileStats, LivenessPredInfo, LivenessPredKind, ScalarCompOp,
};
