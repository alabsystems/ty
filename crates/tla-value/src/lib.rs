// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0
// Allow unsafe in the compact module only (tagged pointer representation).
// All other modules remain safe.
#![deny(unsafe_code)]
#![deny(missing_docs)]

//! TLA+ value representation.
//!
//! This crate defines [`Value`], the runtime value model shared across the `ty`
//! toolchain (the evaluator `tla-eval`, the model checker `tla-check`, and
//! their consumers). It is the layer above the parser/semantic crate
//! `tla-core` and below everything that *computes* with TLA+ values.
//!
//! ## What a value is
//!
//! [`Value`] is an enum covering every kind of value a TLA+ expression can
//! evaluate to: booleans, integers (small-int fast path plus arbitrary-precision
//! `BigInt`), strings, model values, and the structured forms — sets, functions,
//! records, sequences, and tuples. Many set and function forms are stored
//! *lazily* (e.g. `SUBSET S`, `[S -> T]`, `S \cup T`, `{x \in S : P(x)}`,
//! integer intervals `a..b`) so that membership and equality can be decided
//! without materializing potentially astronomically large collections — a
//! central requirement for explicit-state model checking.
//!
//! ## What it provides
//!
//! - **Construction**: direct enum variants plus interning-aware constructors
//!   such as [`val_int`], [`val_true`], [`intern_string`], and builders like
//!   [`SetBuilder`], [`FuncBuilder`], and [`RecordBuilder`].
//! - **Operations**: equality, total ordering, hashing, and set/function/record
//!   queries, all consistent with TLC semantics.
//! - **Fingerprinting**: [`value_fingerprint`] and the TLC-compatible FP64
//!   polynomial rolling hash in [`fingerprint`], used to deduplicate states.
//! - **Model values**: a process-wide registry ([`get_or_assign_model_value_index`],
//!   [`interned_model_value`]) so symmetry reduction can compare them by index.
//! - **Parallel interning**: per-worker intern caches with a frozen shared
//!   snapshot ([`ParallelValueInternRunGuard`], [`WorkerInternGuard`]) for
//!   concurrent model checking.
//! - **Errors**: [`EvalError`]/[`EvalResult`], the failure type for evaluation.
//!
//! ## Invariants
//!
//! Structural identity must agree across `Ord`, `Hash`, and `fingerprint_extend`:
//! two values that compare equal must hash equal and fingerprint equal. Lazy and
//! materialized forms of the same set (e.g. `Seq([])` and `Func([])`) are treated
//! as equal, and interning may canonicalize them to a single representation.
//!
//! ## Modules
//!
//! - [`value`]: The [`Value`] enum and all value operations.
//! - [`error`]: [`EvalError`] and [`EvalResult`] types.
//! - [`dedup_fingerprint`]: State-deduplication fingerprints for model checking.
//! - [`fingerprint`]: TLC-compatible FP64 polynomial rolling hash.
//! - [`itf`]: serialization of values to the Informal Trace Format (ITF).

// debug_flag! macro must be defined before modules that use it
#[macro_use]
pub(crate) mod debug_env;

pub mod churn_stats;
pub mod dedup_fingerprint;
pub mod error;
// `Rp<T>` moved to the dependency-free `tla-rp` crate so `tla-im`'s internal
// node refcounts (`Ref`/`PoolRef`) share the SAME implementation and the SAME
// process-global mode flag (tla-value → tla-im → tla-rp; tla-im cannot depend
// on tla-value). Re-exported here so `tla_value::rp::*` paths are unchanged.
pub use tla_rp as rp;
pub use tla_rp::Rp;
pub mod fingerprint;
pub mod itf;
pub mod value;

// Re-export error types at crate root
pub use error::{EvalError, EvalResult};

// Re-export ITF serialization at crate root
pub use itf::value_to_itf;

// Re-export value types at crate root with explicit API surface (Part of #1582)
// Re-export CompactValue (8-byte tagged pointer representation)
pub use value::compact::CompactValue;

pub use value::{
    // Set builders
    big_union,
    boolean_set,
    // Record hash-consing (post-EXCEPT canonicalization walk)
    canonicalize_records_along_path,
    canonicalize_records_along_paths,
    cartesian_product,
    // Utilities
    checked_interval_len,
    // Value constructors and interning
    clear_int_func_intern_table,
    clear_model_value_registry,
    clear_record_intern_table,
    clear_set_intern_table,
    clear_string_intern_table,
    clear_tlc_string_tokens,
    // Compact bag kill switch
    compact_bags_enabled,
    // Model values
    get_or_assign_model_value_index,
    intern_string,
    interned_model_value,
    lookup_model_value_index,
    lookup_model_value_index_str,
    lookup_tlc_string_token,
    model_value_count,
    powerset,
    range_set,
    set_skip_int_func_interning,
    set_skip_set_interning,
    tlc_string_len,
    tlc_string_subseq_utf16_offsets,
    tlc_string_token,
    val_false,
    val_int,
    val_true,
    BagValue,
    CapturedChain,
    // Core value types
    ClosureValue,
    ComponentDomain,
    FuncBuilder,
    FuncSetValue,
    FuncTakeSource,
    FuncValue,
    IntIntervalFunc,
    IntervalValue,
    KSubsetValue,
    LazyDomain,
    LazyFuncCaptures,
    LazyFuncValue,
    MVPerm,
    RecordBuilder,
    RecordCanonPathElem,
    RecordSetValue,
    RecordValue,
    SeqSetValue,
    SeqValue,
    SetBuilder,
    SetCapValue,
    SetCupValue,
    SetDiffValue,
    SetPredCaptures,
    SetPredIdentity,
    SetPredIdentityVisitor,
    SetPredValue,
    SortedSet,
    SubsetValue,
    TirBody,
    TlcSetIterInline,
    TupleSetValue,
    UnionValue,
    Value,
};

// Re-export value_fingerprint at crate root for extraction (#1269)
pub use fingerprint::value_fingerprint;

// Re-export fingerprint error types at crate root (Part of #3203)
pub use value::value_fingerprint::{FingerprintError, FingerprintResult};

/// Clear thread-local TLC normalization cache between model-checking runs.
pub fn clear_tlc_norm_cache() {
    value::clear_tlc_norm_cache();
}

// Part of #3285, Part of #3334: Re-export parallel interning API for use by tla-check.
// Raw lifecycle toggles (freeze/unfreeze/enable/disable) are now crate-internal;
// external callers use ParallelValueInternRunGuard instead.
pub use value::parallel_intern::{
    parallel_readonly_value_caches_active, read_intern_attribution_counters,
    InternAttributionCounters, ParallelValueInternRunGuard, SharedValueCacheMode,
    WorkerInternGuard,
};
