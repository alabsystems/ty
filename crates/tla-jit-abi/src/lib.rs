// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Stable ABI types for TY JIT/AOT compilation.
//!
//! This crate is a **near-leaf crate** — its only tla-* dependency is
//! `tla-core` (used for `NameId` in compound-layout descriptors; `tla-core`
//! does not depend back on `tla-jit-abi`, so no cycle is introduced). It
//! exists to hold native calling-convention types and pure-data layout
//! descriptors shared by `tla-ir`, `tla-check`, and `tla-trust_cg` without
//! creating cargo cycles.
//!
//! Part of #4265 and #4267: keep the stable ABI leaf crate while trust-codegen owns
//! active native execution.
//!
//! # Types
//!
//! - [`JitStatus`] — Success/error/fallback indicator returned by compiled code.
//! - [`JitRuntimeErrorKind`] — Enumeration of runtime errors that compiled
//!   code can raise (division by zero, overflow, type mismatch, ...).
//! - [`JitCallOut`] — `#[repr(C)]` output struct written by compiled functions
//!   via an out-pointer, avoiding reliance on the Rust ABI for return values.
//! - [`JitRuntimeError`] — `thiserror` wrapper suitable for `Result` types in
//!   the compilation pipeline (backend-agnostic subset of per-backend error
//!   enums).
//! - Function pointer aliases ([`JitExprFn`], [`JitFn0`], [`JitInvariantFn`],
//!   [`JitNextStateFn`]) describing the `extern "C"` shapes that compiled code
//!   exposes to the runtime.
//!
//! # Overflow Strategy
//!
//! trust-codegen compiled code uses **i64-only with overflow rejection** (Part of #746):
//!
//! | Implementation | Strategy     | Behavior on Overflow           |
//! |----------------|--------------|--------------------------------|
//! | TLC (baseline) | 32-bit int   | Runtime error                  |
//! | tla-check      | i64 + BigInt | Silent promotion to BigInt     |
//! | trust-codegen native   | i64 only     | `ArithmeticOverflow` runtime error |

#![deny(missing_docs)]

use thiserror::Error;

pub mod bfs_descriptors;
pub mod bfs_output;
pub mod cache_stats;
pub mod compound_runtime;
pub mod fingerprint;
pub mod helpers;
pub mod homotopy;
pub mod layout;
pub mod liveness_types;
pub mod next_state_loop;
pub mod successor_kernel;
pub mod tier_types;

pub use bfs_descriptors::{BfsStepSpec, CompiledActionFn, CompiledInvariantFn};
pub use bfs_output::{
    BfsBatchResult, BfsStepError, BfsStepOutput, FlatBfsStepOutput, FlatBfsStepOutputRef,
};
pub use cache_stats::{CacheBuildStats, CompileStats};
pub use compound_runtime::{
    clear_compound_scratch, compound_scratch_guard, deserialize_value, infer_layout,
    infer_var_layout, read_compound_scratch, serialize_value, with_compound_scratch,
    with_compound_scratch_mut, CompoundScratchGuard, COMPOUND_SCRATCH_BASE,
};
pub use fingerprint::{ty_compiled_fp_u64, FLAT_COMPILED_DOMAIN_SEED};
pub use helpers::{
    binding_key_for_bindings, binding_key_for_values, bindings_to_jit_i64, classify_value,
    set_bitmask_element_to_value, set_bitmask_elements_to_value, value_is_finite_binding_literal,
    value_to_jit_i64, values_are_finite_binding_literals,
};
pub use homotopy::ActionHomotopy;
pub use layout::{
    CompoundLayout, ScalarSlotKind, SetBitmaskElement, StateLayout, VarLayout, TAG_BOOL, TAG_FUNC,
    TAG_INT, TAG_RECORD, TAG_SEQ, TAG_SET, TAG_STRING, TAG_TUPLE,
};
pub use liveness_types::{
    CompiledAcceptanceCheckFn, CompiledActionPredBatchFn, CompiledStatePredBatchFn,
    LivenessCompileStats, LivenessPredInfo, LivenessPredKind, ScalarCompOp,
};
pub use next_state_loop::{NextStateLoopFn, NextStateLoopSink, NextStateLoopSupport};
pub use successor_kernel::{
    GenericKernelDescriptor, GenericKernelShape, KernelAbiValue, KernelArtifactAdoptionError,
    KernelArtifactAdoptionEvidence, KernelArtifactChecksum, KernelArtifactChecksums,
    KernelArtifactKind, KernelEntry, KernelMarkingLayout, KernelRegisterVectorLayout,
    KernelStateDomain, KernelStateSlot, KernelStateSlotKind, KernelSymbolSignature,
    PredicateKernelDescriptor, SuccessorKernelDescriptor, SuccessorKernelFn, SuccessorKernelOut,
    SuccessorKernelShape, SuccessorKernelStatus, SuccessorKernelUnsupportedReason,
    WholeProgramKernel, ANALYTICAL_KERNEL_ARTIFACT_KIND, AY_SYMBOLIC_KERNEL_ARTIFACT_KIND,
    FINGERPRINT_KERNEL_ARTIFACT_KIND, KERNEL_ABI_VALUE_I32, KERNEL_ABI_VALUE_PTR,
    KERNEL_ABI_VALUE_VOID, KERNEL_ARTIFACT_CONTRACT_SCHEMA,
    KERNEL_ARTIFACT_CONTRACT_SCHEMA_VERSION, KERNEL_SYMBOL_ABI_EXTERN_C,
    NATIVE_HELPER_KERNEL_ARTIFACT_KIND, PREDICATE_KERNEL_ARTIFACT_KIND,
    REPLAY_KERNEL_ARTIFACT_KIND, SUCCESSOR_KERNEL_ARTIFACT_KIND, TY_KERNEL_ARTIFACT_CONSUMER,
    TY_PREDICATE_KERNEL_EVIDENCE_METADATA, TY_SUCCESSOR_KERNEL_EVIDENCE_METADATA,
    WHOLE_PROGRAM_KERNEL_SCHEMA, WHOLE_PROGRAM_KERNEL_SCHEMA_VERSION,
};
pub use tier_types::{
    ActionDescriptor, ActionProfile, CompilationTier, InvariantDescriptor, JitActionResult,
    NextStateDispatchCounters, SpecType, SpecializationPlan, SpecializedVarInfo, TierConfig,
    TierPromotion, DEFAULT_TIER1_THRESHOLD, DEFAULT_TIER2_THRESHOLD,
};

/// Status indicator for JIT function calls.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum JitStatus {
    /// Function completed successfully; value field is valid.
    Ok = 0,
    /// Runtime error occurred; err_kind and span fields are valid.
    RuntimeError = 1,
    /// JIT code hit a compound-value opcode (FuncApply, RecordGet, etc.)
    /// that cannot be evaluated natively. The caller must fall back to
    /// the bytecode VM or tree-walker for this state.
    /// Part of #3798.
    FallbackNeeded = 2,

    /// JIT code evaluated some scalar conjuncts of an invariant conjunction
    /// and they all passed, but could not evaluate the remaining compound
    /// conjuncts. The `conjuncts_passed` field indicates how many top-level
    /// conjuncts were verified by JIT. The caller should tree-walk only the
    /// compound conjuncts, skipping the already-verified scalar ones.
    ///
    /// This is returned when an invariant is a conjunction (And chain / short-
    /// circuit JumpFalse pattern) and the JIT can compile some conjuncts but
    /// not others. Unlike `FallbackNeeded` which requires full re-evaluation,
    /// `PartialPass` means the JIT-checked conjuncts definitively passed.
    PartialPass = 3,
}

/// Runtime error kinds that can occur during JIT'd expression evaluation.
///
/// These map to `EvalError` variants in `tla-check`:
/// - `DivisionByZero` → `EvalError::DivisionByZero`
/// - `ModulusNotPositive` → `EvalError::ModulusNotPositive`
/// - `TypeMismatch` → `EvalError::TypeError`
/// - `ArithmeticOverflow` → `EvalError::ArithmeticOverflow` (to be added)
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum JitRuntimeErrorKind {
    /// Division by zero (x / 0 or x \div 0).
    /// TLC: EC.TLC_MODULE_DIVISION_BY_ZERO
    DivisionByZero = 1,

    /// Modulus with non-positive divisor (x % y where y <= 0).
    /// TLC: EC.TLC_MODULE_ARGUMENT_ERROR for "%"
    ModulusNotPositive = 2,

    /// Type mismatch (e.g., integer operation on boolean).
    /// TLC: EC.TLC_MODULE_ARGUMENT_ERROR
    TypeMismatch = 3,

    /// Arithmetic overflow (i64 bounds exceeded).
    /// TLC: EC.TLC_MODULE_OVERFLOW
    /// Part of #746 - matches TLC's overflow-as-error semantics.
    ArithmeticOverflow = 4,
}

impl JitRuntimeErrorKind {
    /// Returns a human-readable name for the error kind.
    pub fn name(&self) -> &'static str {
        match self {
            JitRuntimeErrorKind::DivisionByZero => "division by zero",
            JitRuntimeErrorKind::ModulusNotPositive => "modulus with non-positive divisor",
            JitRuntimeErrorKind::TypeMismatch => "type mismatch",
            JitRuntimeErrorKind::ArithmeticOverflow => "arithmetic overflow",
        }
    }
}

/// Output struct for JIT'd functions.
///
/// JIT'd functions write their result to this struct via an out-pointer.
/// This avoids relying on Rust ABI for the return value.
///
/// # Memory Layout
///
/// Uses `#[repr(C)]` for stable, portable layout across the JIT boundary.
/// Field offsets should be computed using `std::mem::offset_of!` when
/// generating trust-codegen store instructions.
///
/// # Usage
///
/// ```text
/// let mut out = JitCallOut::default();
/// unsafe { jit_fn(&mut out); }
/// match out.status {
///     JitStatus::Ok => println!("Result: {}", out.value),
///     JitStatus::RuntimeError => println!("Error: {:?}", out.err_kind),
/// }
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JitCallOut {
    /// Status indicator (ok vs runtime error vs partial pass).
    pub status: JitStatus,
    /// Result value (valid when status == Ok).
    pub value: i64,
    /// Runtime error kind (valid when status == RuntimeError).
    pub err_kind: JitRuntimeErrorKind,
    /// Span start byte offset (valid when status == RuntimeError).
    pub err_span_start: u32,
    /// Span end byte offset (valid when status == RuntimeError).
    pub err_span_end: u32,
    /// File ID for span (valid when status == RuntimeError).
    pub err_file_id: u32,
    /// Number of top-level conjuncts verified by JIT (valid when status == PartialPass).
    /// The caller can skip these conjuncts when falling back to the interpreter.
    pub conjuncts_passed: u32,
}

impl Default for JitCallOut {
    fn default() -> Self {
        JitCallOut {
            status: JitStatus::Ok,
            value: 0,
            err_kind: JitRuntimeErrorKind::DivisionByZero, // placeholder
            err_span_start: 0,
            err_span_end: 0,
            err_file_id: 0,
            conjuncts_passed: 0,
        }
    }
}

impl JitCallOut {
    /// Create a successful result.
    pub fn ok(value: i64) -> Self {
        JitCallOut {
            status: JitStatus::Ok,
            value,
            ..Default::default()
        }
    }

    /// Create a runtime error result.
    pub fn error(kind: JitRuntimeErrorKind, file_id: u32, span_start: u32, span_end: u32) -> Self {
        JitCallOut {
            status: JitStatus::RuntimeError,
            value: 0,
            err_kind: kind,
            err_span_start: span_start,
            err_span_end: span_end,
            err_file_id: file_id,
            conjuncts_passed: 0,
        }
    }

    /// Check if this result is successful.
    pub fn is_ok(&self) -> bool {
        self.status == JitStatus::Ok
    }

    /// Check if this result is an error.
    pub fn is_err(&self) -> bool {
        self.status == JitStatus::RuntimeError
    }

    /// Check if fallback to interpreter is needed.
    ///
    /// Returned when JIT code encounters a compound-value opcode
    /// (FuncApply, RecordGet) that cannot be evaluated natively.
    /// The caller should re-evaluate this state using the bytecode VM
    /// or tree-walker.
    /// Part of #3798.
    pub fn needs_fallback(&self) -> bool {
        self.status == JitStatus::FallbackNeeded || self.status == JitStatus::PartialPass
    }

    /// Check if the JIT evaluated some but not all conjuncts.
    ///
    /// Returned when an invariant is a conjunction and the JIT successfully
    /// verified the scalar conjuncts but could not evaluate compound ones.
    /// The `conjuncts_passed` field tells the caller how many top-level
    /// conjuncts were verified by JIT.
    pub fn is_partial_pass(&self) -> bool {
        self.status == JitStatus::PartialPass
    }

    /// Create a partial-pass result indicating how many conjuncts were verified.
    pub fn partial_pass(conjuncts_passed: u32) -> Self {
        JitCallOut {
            status: JitStatus::PartialPass,
            conjuncts_passed,
            ..Default::default()
        }
    }
}

/// Function pointer type for JIT-compiled constant expressions (Phase 0 ABI).
pub type JitExprFn = unsafe extern "C" fn() -> i64;

/// Function pointer type for JIT'd expressions (Phase 1 ABI).
pub type JitFn0 = unsafe extern "C" fn(out: *mut JitCallOut);

/// Function pointer type for JIT'd invariant functions (Phase B2 ABI).
pub type JitInvariantFn =
    unsafe extern "C" fn(out: *mut JitCallOut, state: *const i64, state_len: u32);

/// Function pointer for JIT-compiled next-state relation.
pub type JitNextStateFn = unsafe extern "C" fn(
    out: *mut JitCallOut,
    state_in: *const i64,
    state_out: *mut i64,
    state_len: u32,
);

/// A binding specialization request: action name + concrete binding values.
///
/// When model checking `\E i \in {0,1,2} : SendMsg(i)`, the action splitter
/// creates three instances sharing the operator "SendMsg" but with different
/// bindings. Each `BindingSpec` requests a specialized compiled function where
/// the binding parameter is baked in as a `LoadImm` constant.
///
/// Lives in `tla-jit-abi` because the trust-codegen backend (`tla-trust_cg`) and
/// `tla-check::trust_cg_dispatch` both reference this stable call-boundary type.
///
/// Part of #4270.
#[derive(Debug, Clone)]
pub struct BindingSpec {
    /// The base action operator name (e.g., "SendMsg").
    pub action_name: String,
    /// Precomputed executable cache key for this binding instance.
    ///
    /// Scalar-only bindings use [`specialized_key`] unchanged. Bindings with
    /// finite compound literals use a typed structural key so trust-codegen dispatch
    /// can address them without lossy raw-i64 extraction.
    pub binding_key: String,
    /// Concrete scalar values encoded in the legacy executable cache key.
    ///
    /// This is populated only when all binding literals fit the raw i64 ABI.
    /// Finite compound bindings use [`Self::binding_key`] plus
    /// [`Self::binding_value_literals`] instead.
    pub binding_values: Vec<i64>,
    /// Typed equivalents of [`Self::binding_values`].
    ///
    /// Raw key values remain part of the dispatch ABI, but specialization must
    /// preserve scalar kind for strings and model values so typed lowering can
    /// prove compact function keys correctly.
    pub binding_value_literals: Vec<tla_value::Value>,
    /// Concrete scalar base-operator formal values in declaration order.
    ///
    /// These are separate from `binding_values` because split action metadata
    /// can contain outer witnesses before or after operator formals. This is
    /// populated only when all formals fit the raw i64 ABI.
    pub formal_values: Vec<i64>,
    /// Typed equivalents of [`Self::formal_values`] in declaration order.
    ///
    /// trust-codegen uses these to bake parameter registers with `LoadBool`/`LoadConst`
    /// when a raw `LoadImm` would erase semantic type information.
    pub formal_value_literals: Vec<tla_value::Value>,
}

/// Construct the specialized cache key for a (action, bindings) pair.
///
/// Returns the base action name if bindings is empty, or
/// `"ActionName__v0_v1_v2"` for bound actions.
///
/// Lives in `tla-jit-abi` alongside [`BindingSpec`]; the cache key format is
/// part of the stable ABI for compiled action dispatch.
pub fn specialized_key(action_name: &str, binding_values: &[i64]) -> String {
    if binding_values.is_empty() {
        action_name.to_string()
    } else {
        let suffix: Vec<String> = binding_values.iter().map(|v| v.to_string()).collect();
        format!("{action_name}__{}", suffix.join("_"))
    }
}

/// Backend-agnostic native runtime error types.
///
/// This is a subset of per-backend error enums without trust_cg-specific variants.
/// Each backend converts `JitRuntimeError` into its backend-specific error type
/// via `From`.
#[derive(Debug, PartialEq, Error)]
pub enum JitRuntimeError {
    /// Failed to compile a function
    #[error("Compilation failed: {0}")]
    CompileError(String),

    /// Unsupported TLA+ expression type
    #[error("Unsupported expression: {0}")]
    UnsupportedExpr(String),

    /// Type mismatch during compilation
    #[error("Type mismatch: expected {expected}, got {actual}")]
    TypeMismatch {
        /// The type the compiler required at this position.
        expected: String,
        /// The type that was actually encountered.
        actual: String,
    },

    /// Bytecode function is not eligible for JIT compilation
    #[error("Not JIT-eligible: {reason}")]
    NotEligible {
        /// Human-readable explanation of why the function was rejected
        /// by the eligibility gate (e.g. an unsupported opcode or shape).
        reason: String,
    },

    /// Unsupported bytecode opcode for JIT lowering
    #[error("Unsupported bytecode opcode: {0}")]
    UnsupportedOpcode(String),

    /// Runtime error from JIT-compiled code (e.g., division by zero)
    #[error("JIT runtime error: {kind:?}")]
    RuntimeError {
        /// The specific runtime fault raised by the compiled code.
        kind: JitRuntimeErrorKind,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    #[test]
    fn test_jit_status_repr() {
        assert_eq!(JitStatus::Ok as u8, 0);
        assert_eq!(JitStatus::RuntimeError as u8, 1);
        assert_eq!(JitStatus::FallbackNeeded as u8, 2);
        assert_eq!(JitStatus::PartialPass as u8, 3);
    }

    #[test]
    fn test_jit_runtime_error_kind_repr() {
        assert_eq!(JitRuntimeErrorKind::DivisionByZero as u8, 1);
        assert_eq!(JitRuntimeErrorKind::ModulusNotPositive as u8, 2);
        assert_eq!(JitRuntimeErrorKind::TypeMismatch as u8, 3);
        assert_eq!(JitRuntimeErrorKind::ArithmeticOverflow as u8, 4);
    }

    #[test]
    fn test_jit_call_out_layout() {
        assert!(mem::size_of::<JitCallOut>() <= 48);
        assert!(mem::align_of::<JitCallOut>() >= 8);
    }

    #[test]
    fn test_jit_call_out_ok() {
        let out = JitCallOut::ok(42);
        assert!(out.is_ok());
        assert!(!out.is_err());
        assert_eq!(out.value, 42);
    }

    #[test]
    fn test_jit_call_out_error() {
        let out = JitCallOut::error(JitRuntimeErrorKind::ArithmeticOverflow, 1, 10, 20);
        assert!(!out.is_ok());
        assert!(out.is_err());
        assert_eq!(out.err_kind, JitRuntimeErrorKind::ArithmeticOverflow);
        assert_eq!(out.err_file_id, 1);
        assert_eq!(out.err_span_start, 10);
        assert_eq!(out.err_span_end, 20);
    }

    #[test]
    fn test_error_kind_names() {
        assert_eq!(
            JitRuntimeErrorKind::DivisionByZero.name(),
            "division by zero"
        );
        assert_eq!(
            JitRuntimeErrorKind::ArithmeticOverflow.name(),
            "arithmetic overflow"
        );
    }

    #[test]
    fn test_jit_runtime_error_runtime_variant() {
        let err = JitRuntimeError::RuntimeError {
            kind: JitRuntimeErrorKind::DivisionByZero,
        };
        let msg = format!("{err}");
        assert!(msg.contains("DivisionByZero"), "got {msg}");
    }

    #[test]
    fn test_specialized_key_empty_bindings() {
        assert_eq!(specialized_key("SendMsg", &[]), "SendMsg");
    }

    #[test]
    fn test_specialized_key_with_bindings() {
        assert_eq!(specialized_key("SendMsg", &[0]), "SendMsg__0");
        assert_eq!(specialized_key("Action", &[1, 2, 3]), "Action__1_2_3");
        assert_eq!(specialized_key("A", &[-1, 42]), "A__-1_42");
    }

    #[test]
    fn test_binding_spec_clone() {
        let spec = BindingSpec {
            action_name: "SendMsg".to_string(),
            binding_key: specialized_key("SendMsg", &[0, 1]),
            binding_values: vec![0, 1],
            binding_value_literals: vec![
                tla_value::Value::SmallInt(0),
                tla_value::Value::Bool(true),
            ],
            formal_values: vec![1],
            formal_value_literals: vec![tla_value::Value::Bool(true)],
        };
        let cloned = spec.clone();
        assert_eq!(cloned.action_name, "SendMsg");
        assert_eq!(cloned.binding_key, "SendMsg__0_1");
        assert_eq!(cloned.binding_values, vec![0, 1]);
        assert_eq!(
            cloned.binding_value_literals,
            vec![tla_value::Value::SmallInt(0), tla_value::Value::Bool(true)]
        );
        assert_eq!(cloned.formal_values, vec![1]);
        assert_eq!(
            cloned.formal_value_literals,
            vec![tla_value::Value::Bool(true)]
        );
    }
}
