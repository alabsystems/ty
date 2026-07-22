// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! trust-codegen native compilation dispatch for the BFS model checker.
//!
//! Provides the active opt-in native path by compiling TIR bytecode
//! through the trust-codegen native pipeline: BytecodeFunction -> trust-ir -> trust-codegen ISel ->
//! RegAlloc -> AArch64/x86 encoding -> JIT executable memory. Zero C dependencies.
//! The resulting native functions share the same `extern "C"` ABI as the legacy
//! JIT compatibility layer, so the BFS dispatch logic can stay backend-neutral.
//!
//! # Design rationale (Part of #4251 Stream 6 / #4118)
//!
//! **(a) Spec shape eligibility.** The BFS path compiles the per-action
//! next-state function via `compile_next_state_native_with_constants`. The
//! next-state ABI is `fn(out, state_in, state_out, state_len)` with NO
//! parameters — so actions with `arity > 0` (EXISTS-bound parameters like
//! `\E i \in S : Action(i)`) are ineligible and routed to the interpreter.
//! Unsupported action bytecode fails closed inside the trust-ir lowering pipeline:
//! the action is omitted from the native dispatch cache, the precise lowering
//! error is logged once, and the action is permanently routed to the interpreter
//! for the remainder of the run.
//!
//! **(b) Per-action dispatch.** [`TrustCgNativeCache`] owns a
//! `FxHashMap<String, NativeNextStateFn>` keyed by action name. At BFS
//! setup, each action is compiled eagerly and the resulting function pointer
//! (plus its backing [`NativeLibrary`] mmap) is installed in the cache.
//! Inside the BFS action loop, dispatch flows interpreter-first-unless-compiled:
//! if `cache.contains_action(action_name)` is true the native path runs;
//! otherwise the interpreter runs. There is NO global switch once the cache
//! is built — every action picks its path independently each iteration.
//!
//! **(c) Fallback on compile failure / ineligibility.** Compile failures
//! are absorbed at `TrustCgNativeCache::build` time: the failure is logged
//! once (at cache construction), the failed action is omitted from
//! `next_state_fns`, and no retry is ever attempted. The interpreter then
//! handles every BFS visit of that action's (state, action) pair. This is
//! the "log once, interpreter forever" policy described in the Stream 6
//! handoff.
//!
//! # ABI Compatibility
//!
//! trust-codegen and the stable native ABI use the same function signature:
//! - **Next-state**: `extern "C" fn(out: *mut JitCallOut, state_in: *const i64, state_out: *mut i64, state_len: u32)`
//! - **Invariant**: `extern "C" fn(out: *mut JitCallOut, state: *const i64, state_len: u32)`
//!
//! # Activation
//!
//! The trust-codegen path is always linked in; enable at runtime with
//! either `TY_TRUST_CG=1` (global trust-codegen dispatch) or
//! `TY_TRUST_CG_BFS=1` (BFS-scoped dispatch — identical code path today; reserved as a
//! narrower switch so invariant / liveness trust-codegen wiring can activate
//! independently in the future). `TY_TRUST_CG_ENTRY_COUNTER_GATE=N` is an
//! opt-in diagnostics gate that routes an action back to the interpreter once
//! trust-codegen reports at least `N` JIT entry-counter hits for that action. No
//! system LLVM installation required — the trust-codegen backend is a pure-Rust JIT
//! compiler.
//!
//! # Pipeline
//!
//! ```text
//! BytecodeFunction (per action / invariant)
//!     |  (tla_ir::lower -> trust_ir::Module)
//!     v
//! trust_ir::Module (SSA IR)
//!     |  (tla_trust_cg::compile_module_native -> trust-codegen JIT pipeline)
//!     v
//! NativeLibrary (JIT executable buffer)
//!     |  (get_symbol -> transmute)
//!     v
//! Callable function pointers
//! ```
//!
//! Part of #4118: Wire tla-trust_cg into tla-check BFS loop.
//! Part of #4251 Stream 6: BFS-scoped activation via `TY_TRUST_CG_BFS`.

use rustc_hash::{FxHashMap, FxHashSet};
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;
use tla_ir::module_batch::{
    plan_frontend_neutral_module_batch_partitions, BatchPartitionOptions, BatchPlanningInput,
};
use tla_ir::{KernelFingerprintAdmissionContract, KernelFrontend};
use tla_jit_abi::{BfsStepError, BindingSpec, FlatBfsStepOutput, JitCallOut};

mod abi;
use abi::{NativeImpliedActionFn, NativeInvariantFn, NativeNextStateFn};

mod config;
use config::*;
mod record_set_scalarize;
pub(in crate::check) use record_set_scalarize::RecordSetScalarizeEnv;
mod sum_fold_scalarize;
pub(in crate::check) use config::{
    trust_cg_dump_native_admission_failures_enabled, trust_cg_fused_invariant_min_states,
    trust_cg_fused_level_defer_threshold, trust_cg_lazy_compile_gate_fires,
    trust_cg_lazy_compile_threshold, trust_cg_lazy_compile_work_threshold,
    trust_cg_runtime_telemetry_enabled, trust_cg_setup_timing_enabled,
};

/// Whether the JIT lazy-compile WORK arm is wired in the shipping build — i.e.
/// the default work threshold is flipped off `u64::MAX` (`u64::MAX` ships the
/// arm DARK). The self-liveness model derives its shipping work-arm wiring from
/// this ([`crate::selfliveness::Wiring::shipping`]) so the model can never
/// silently drift from the live constant: flipping the threshold to enable the
/// arm automatically moves the shipping wiring from `bug()` to `fixed()`.
pub(crate) fn work_arm_wired_default() -> bool {
    config::TRUST_CG_LAZY_COMPILE_WORK_THRESHOLD_DEFAULT != u64::MAX
}

mod admission;
pub(in crate::check) use admission::should_defer_pre_layout_trust_cg_cache_build;

mod gpu_lower;

const TRUST_CG_NATIVE_ADMISSION_CONSUMER_MODE: &str = "ty_trust_cg_bfs_runtime";
const TRUST_CG_NATIVE_ADMISSION_KIND: &str = "ty_native_activation";
const TRUST_CG_NATIVE_ADMISSION_EVIDENCE_REPORT_SCHEMA: &str =
    "ty.trust_cg.native_admission_evidence_report.v1";
const TRUST_CG_NATIVE_ADMISSION_EVIDENCE_REPORT_SCHEMA_VERSION: u32 = 1;
const TRUST_CG_NATIVE_ACTION_CALLOUT_BATCH_SETUP_ROW_KIND: &str =
    "ty_native_action_callout_batch_setup";
const TRUST_CG_NATIVE_ACTION_CALLOUT_BATCH_SETUP_SCHEMA: &str =
    "ty.trust_cg.native_action_callout_batch_setup.v1";
const TRUST_CG_NATIVE_ACTION_CALLOUT_BATCH_SETUP_SCHEMA_VERSION: u32 = 1;
const TRUST_CG_NATIVE_ACTION_CALLOUT_SHARD_MAX_ACTIONS: usize = 16;
const TRUST_CG_NATIVE_ACTION_CALLOUT_SHARD_MAX_ESTIMATED_IR_NODES: usize = 8192;
const TRUST_CG_NATIVE_CALLOUT_SELFTEST_REDZONE_SLOTS: usize = 64;
const TRUST_CG_NATIVE_CALLOUT_SELFTEST_INPUT_CANARY: i64 = 0x4123_7b5a_19d0_6ec3;
const TRUST_CG_NATIVE_CALLOUT_SELFTEST_OUTPUT_CANARY: i64 = 0x2f4d_0619_7b5a_34c1;

fn native_fused_callout_sentinel() -> JitCallOut {
    JitCallOut {
        status: tla_jit_abi::JitStatus::RuntimeError,
        ..Default::default()
    }
}

fn maybe_write_callout_replay_artifact(
    kind: &str,
    index: u32,
    symbol_name: &str,
    name: &str,
    func_addr: usize,
    state_len: u32,
    sample: &[i64],
    state_out: Option<&[i64]>,
) {
    let Some(root) = trust_cg_replay_artifact_dir() else {
        return;
    };
    if !trust_cg_replay_filter_allows(symbol_name, name) {
        return;
    }

    static SEQ: AtomicU64 = AtomicU64::new(0);
    let sequence = SEQ.fetch_add(1, AtomicOrdering::Relaxed) + 1;
    let dir = root.join("callouts");
    if let Err(err) = std::fs::create_dir_all(&dir) {
        eprintln!(
            "[trust_cg-selftest][replay-artifact] failed to create {}: {err}",
            dir.display()
        );
        return;
    }

    let stem = format!(
        "{sequence:06}-{}-{}-{}",
        trust_cg_replay_artifact_component(kind),
        index,
        trust_cg_replay_artifact_component(symbol_name)
    );
    let path = dir.join(format!("{stem}.entering.json"));
    let payload = serde_json::json!({
        "schema": "ty.trust_cg.native_callout_selftest.v1",
        "event": "entering",
        "sequence": sequence,
        "kind": kind,
        "index": index,
        "symbol_name": symbol_name,
        "name": name,
        "function_address": format!("0x{func_addr:x}"),
        "abi": {
            "signature": if state_out.is_some() {
                "extern \"C\" fn(*mut JitCallOut, *const i64, *mut i64, u32)"
            } else {
                "extern \"C\" fn(*mut JitCallOut, *const i64, u32)"
            },
            "state_len": state_len,
        },
        "source_revisions": {
            "ty_git_commit": trust_cg_replay_ty_git_commit(),
        },
        "state": {
            "slots": sample,
            "head": &sample[..sample.len().min(16)],
        },
        "state_out_initial": state_out.map(|slots| serde_json::json!({
            "slots": slots,
            "head": &slots[..slots.len().min(16)],
        })),
    });

    let json = match serde_json::to_string_pretty(&payload) {
        Ok(json) => json,
        Err(err) => {
            eprintln!(
                "[trust_cg-selftest][replay-artifact] failed to serialize callout sample for symbol={symbol_name}: {err}"
            );
            return;
        }
    };
    if let Err(err) = std::fs::write(&path, json + "\n") {
        eprintln!(
            "[trust_cg-selftest][replay-artifact] failed to write {}: {err}",
            path.display()
        );
    }
}

/// Cache of trust_cg-compiled native functions for BFS dispatch.
///
/// trust-codegen counterpart of the former `JitNextStateCache`, backed by native shared
/// libraries (`.dylib`/`.so`) instead of JIT-managed executable memory. Holds the
/// `NativeLibrary` handles to keep the dlopen'd code alive.
pub(in crate::check) struct TrustCgNativeCache {
    /// Per-action next-state function pointers, keyed by action name.
    next_state_fns: FxHashMap<String, NativeNextStateFn>,
    /// Per-action multi-successor ("NextStateLoop") record-set kernel function
    /// pointers, keyed by action name. Disjoint from `next_state_fns`: an action
    /// is in at most one of the two maps. These implement the sink-based
    /// `tla_jit_abi::NextStateLoopFn` ABI and are dispatched via the sink call
    /// convention in the native fused BFS loop (see the `is_loop` action flag).
    /// Empty unless `TY_RECORD_SET_NATIVE=1` selected a proven-closed record-set
    /// action.
    next_state_loop_fns: FxHashMap<String, tla_jit_abi::NextStateLoopFn>,
    /// Expected native action keys produced by inner-EXISTS expansion, keyed by
    /// the unexpanded base action key. This tracks missing expanded siblings so
    /// coverage and compiled BFS eligibility fail closed after partial
    /// expansion compile failures.
    inner_exists_expansion_keys: FxHashMap<String, Vec<String>>,
    /// Proof-backed inner-EXISTS expansions that may be consumed by the native
    /// fused parent loop. Absence means the expansion must remain fail-closed
    /// for native-fused admission even if the individual callbacks compiled.
    inner_exists_expansion_proofs: FxHashMap<String, TrustCgInnerExistsExpansionProof>,
    /// Per-invariant function pointers, indexed parallel to the spec's invariant list.
    /// Uses `Option` to maintain index alignment when compilation fails mid-sequence.
    /// `invariant_fns[i]` corresponds to the spec's invariant at index `i`.
    invariant_fns: Vec<Option<NativeInvariantFn>>,
    /// Per-state-constraint function pointers, indexed parallel to the spec's
    /// state constraint list. These use the same native ABI as invariants but
    /// are kept separate because state constraints prune successors rather than
    /// reporting safety violations.
    state_constraint_fns: Vec<Option<NativeInvariantFn>>,
    /// Per-action-property transition predicate function pointers, indexed
    /// parallel to `CompiledSpec::native_implied_actions`.
    implied_action_fns: Vec<Option<NativeImpliedActionFn>>,
    /// Native-library action entry points available for fused-level linking.
    native_action_entries: FxHashMap<String, TrustCgNativeActionEntry>,
    /// Native-library invariant entry points available for fused-level linking.
    ///
    /// This is indexed parallel to `invariant_fns` and the configured
    /// invariant list. A `None` slot means the invariant did not compile and
    /// the native fused level must stay action-only.
    native_invariant_entries: Vec<Option<TrustCgNativeInvariantEntry>>,
    /// Native-library state-constraint entries available for native fused
    /// successor pruning. Indexed parallel to `state_constraint_fns`.
    native_state_constraint_entries: Vec<Option<TrustCgNativeInvariantEntry>>,
    /// Native-library action-property entries available for native fused
    /// transition checking. Indexed parallel to `implied_action_fns`.
    native_implied_action_entries: Vec<Option<TrustCgNativeInvariantEntry>>,
    /// Number of state variables in the model.
    state_var_count: usize,
    /// Optimization level used for this cache's compiled entry points.
    opt_level: tla_trust_cg::OptLevel,
    /// Keeps the native libraries alive (dlopen handles).
    /// Dropping these invalidates the function pointers above.
    _libraries: Vec<tla_trust_cg::NativeLibrary>,
}

thread_local! {
    /// WP-21: the `JitRuntimeErrorKind` (as its ABI `u8`, `0` = none) of the
    /// most recent [`TrustCgNativeCache::eval_action_with_state_len_into`]
    /// call on THIS thread. Cleared on entry, set only on
    /// `JitStatus::RuntimeError`. Lets the hybrid dispatcher classify a
    /// typed `TypeMismatch` shape-guard decline (e.g. a union-arm read on a
    /// path whose enabling guard the bytecode had not yet evaluated) apart
    /// from genuine native runtime errors, WITHOUT widening the
    /// `Option<Result<bool, ()>>` contract every caller matches on.
    /// Thread-local because dispatch and classification happen back-to-back
    /// on the executing thread.
    static LAST_ACTION_RUNTIME_ERROR_KIND: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

/// WP-21: whether the most recent native action eval on this thread failed
/// with the typed `TypeMismatch` runtime guard (the fail-closed shape-guard
/// class: union-arm reads, capacity/universe confinement, set-member
/// not-found). `false` for any other error kind and for non-error outcomes.
pub(in crate::check) fn last_native_action_error_was_shape_guard() -> bool {
    LAST_ACTION_RUNTIME_ERROR_KIND.with(|kind| {
        kind.get() == tla_jit_abi::JitRuntimeErrorKind::TypeMismatch as u8
    })
}

#[derive(Clone)]
struct TrustCgNativeActionEntry {
    library: tla_trust_cg::NativeLibrary,
    symbol_name: String,
    binding_values: Vec<i64>,
    formal_values: Vec<i64>,
    read_vars: Vec<u16>,
    write_vars: Vec<u16>,
    /// Hybrid-placeholder (compound) vars this artifact reads through the
    /// allocation-lean compound-read callout (item 4 M1). Empty unless the
    /// lowering emitted a callout for the var — the M1 admission gate treats
    /// an undeclared compound read as a hard decline.
    compound_read_vars: Vec<u16>,
    batch_shard: Option<TrustCgNativeActionBatchShardMetadata>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TrustCgNativeActionBatchShardMetadata {
    shard_index: usize,
    shard_count: usize,
    shard_stable_id: String,
    shared_shape_id: String,
    plan_reuse_manifest_id: String,
    artifact_identity: String,
    artifact_cache_digest: String,
    warm_cache_status: &'static str,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TrustCgNativeActionBatchShardCompileMetadata {
    shard_index: usize,
    shard_count: usize,
    shard_stable_id: String,
    shared_shape_id: String,
    frontend_neutral_reuse_id: String,
    plan_reuse_manifest_id: String,
}

impl TrustCgNativeActionBatchShardCompileMetadata {
    fn from_plan_shard(
        shard_index: usize,
        shard_count: usize,
        shard: &TrustCgActionCalloutShard,
        plan_reuse_manifest_id: &str,
    ) -> Self {
        Self {
            shard_index,
            shard_count,
            shard_stable_id: shard.stable_id.clone(),
            shared_shape_id: shard.shared_shape_id.clone(),
            frontend_neutral_reuse_id: shard.frontend_neutral_reuse_id.clone(),
            plan_reuse_manifest_id: plan_reuse_manifest_id.to_string(),
        }
    }

    fn with_artifact(
        &self,
        artifact_identity: String,
        artifact_cache_digest: String,
        warm_cache_status: &'static str,
    ) -> TrustCgNativeActionBatchShardMetadata {
        TrustCgNativeActionBatchShardMetadata {
            shard_index: self.shard_index,
            shard_count: self.shard_count,
            shard_stable_id: self.shard_stable_id.clone(),
            shared_shape_id: self.shared_shape_id.clone(),
            plan_reuse_manifest_id: self.plan_reuse_manifest_id.clone(),
            artifact_identity,
            artifact_cache_digest,
            warm_cache_status,
        }
    }
}

#[derive(Clone)]
struct TrustCgNativeInvariantEntry {
    library: tla_trust_cg::NativeLibrary,
    symbol_name: String,
}

#[derive(Clone)]
struct TrustCgNativeCalloutSelftestAction {
    index: u32,
    name: String,
    symbol_name: String,
    func: NativeNextStateFn,
    library: Option<tla_trust_cg::NativeLibrary>,
}

#[derive(Clone)]
struct TrustCgNativeCalloutSelftestPredicate {
    index: u32,
    name: String,
    symbol_name: String,
    func: NativeInvariantFn,
    library: Option<tla_trust_cg::NativeLibrary>,
}

#[derive(Clone)]
struct TrustCgNativeCalloutSelftestMissing {
    kind: &'static str,
    index: u32,
    name: String,
    symbol_name: String,
}

struct TrustCgNativeCalloutGuardedState {
    slots: Vec<i64>,
    payload_start: usize,
    payload_len: usize,
    canary: i64,
}

impl TrustCgNativeCalloutGuardedState {
    fn new(payload: &[i64], canary: i64) -> Self {
        let payload_start = TRUST_CG_NATIVE_CALLOUT_SELFTEST_REDZONE_SLOTS;
        let payload_len = payload.len();
        let total_len = payload_start
            .checked_add(payload_len)
            .and_then(|len| len.checked_add(TRUST_CG_NATIVE_CALLOUT_SELFTEST_REDZONE_SLOTS))
            .expect("native callout selftest guarded buffer length overflow");
        let mut slots = vec![canary; total_len];
        slots[payload_start..payload_start + payload_len].copy_from_slice(payload);
        Self {
            slots,
            payload_start,
            payload_len,
            canary,
        }
    }

    fn payload(&self) -> &[i64] {
        &self.slots[self.payload_start..self.payload_start + self.payload_len]
    }

    fn payload_mut(&mut self) -> &mut [i64] {
        let payload_start = self.payload_start;
        let payload_end = payload_start + self.payload_len;
        &mut self.slots[payload_start..payload_end]
    }

    fn payload_ptr(&self) -> *const i64 {
        self.payload().as_ptr()
    }

    fn payload_mut_ptr(&mut self) -> *mut i64 {
        self.payload_mut().as_mut_ptr()
    }

    fn verify_read_only(
        &self,
        kind: &str,
        index: u32,
        symbol_name: &str,
        name: &str,
        expected_payload: &[i64],
    ) -> Result<(), String> {
        self.verify_canaries(kind, index, symbol_name, name)?;
        if self.payload() != expected_payload {
            return Err(format!(
                "{kind} callout mutated read-only state input: index={index} symbol={symbol_name} name={name}"
            ));
        }
        Ok(())
    }

    fn verify_canaries(
        &self,
        kind: &str,
        index: u32,
        symbol_name: &str,
        name: &str,
    ) -> Result<(), String> {
        for (slot, value) in self.slots[..self.payload_start].iter().enumerate() {
            if *value != self.canary {
                return Err(format!(
                    "{kind} callout wrote before selftest state buffer: index={index} symbol={symbol_name} name={name} redzone_slot={slot}"
                ));
            }
        }
        let suffix_start = self.payload_start + self.payload_len;
        for (offset, value) in self.slots[suffix_start..].iter().enumerate() {
            if *value != self.canary {
                return Err(format!(
                    "{kind} callout wrote past selftest state buffer: index={index} symbol={symbol_name} name={name} redzone_slot={offset}"
                ));
            }
        }
        Ok(())
    }
}

struct TrustCgNativeCalloutSelftest {
    actions: Vec<TrustCgNativeCalloutSelftestAction>,
    state_constraints: Vec<TrustCgNativeCalloutSelftestPredicate>,
    invariants: Vec<TrustCgNativeCalloutSelftestPredicate>,
    missing_expected: Vec<TrustCgNativeCalloutSelftestMissing>,
    fail_closed: bool,
}

impl TrustCgNativeCalloutSelftest {
    fn clear_tla_runtime_arenas_before_callout() {
        tla_trust_cg::runtime_abi::tla_ops::clear_tla_iter_arena();
        tla_trust_cg::runtime_abi::tla_ops::clear_tla_arena();
    }

    fn format_callout_publication_bytes(bytes: Option<&[u8]>) -> String {
        use std::fmt::Write as _;
        match bytes {
            Some(bytes) => bytes.iter().fold(String::new(), |mut acc, byte| {
                let _ = write!(acc, "{byte:02x}");
                acc
            }),
            None => "none".to_string(),
        }
    }

    fn ensure_callout_library_published(
        library: Option<&tla_trust_cg::NativeLibrary>,
        symbol_ptr: *mut std::ffi::c_void,
        kind: &str,
        index: u32,
        symbol_name: &str,
        name: &str,
    ) -> Result<Option<String>, String> {
        let mut proof_fields = None;
        if let Some(library) = library {
            let proof = library
                .diagnose_published_symbol_ptr(symbol_name, symbol_ptr)
                .map_err(|err| {
                format!(
                    "failed to prove/re-publish native {kind} callout symbol executable before selftest call: index={index} symbol={symbol_name} name={name}: {err}"
                )
            })?;
            let first_code_bytes =
                Self::format_callout_publication_bytes(proof.first_code_bytes.as_deref());
            proof_fields = Some(format!(
                "pointer=0x{:x} owner_path={} buffer_base=0x{:x} buffer_end=0x{:x} code_len={} allocation_len={} expected_symbol_offset={} actual_ptr_offset={} exact_symbol_match={} publication.map_jit={} publication.write_protect_supported={} publication.published_rx={} mprotect_rx_ok={} execute_mode_reasserted={} first_code_bytes={}",
                proof.pointer,
                library.path().display(),
                proof.buffer_base,
                proof.buffer_end,
                proof.code_len,
                proof.allocation_len,
                proof.expected_symbol_offset,
                proof.actual_ptr_offset,
                proof.exact_symbol_match,
                proof.publication_contract.map_jit,
                proof.publication_contract.write_protect_supported,
                proof.publication_contract.published_rx,
                proof.mprotect_rx_ok,
                proof.execute_mode_reasserted,
                first_code_bytes,
            ));
        } else {
            Ok::<(), String>(())?;
        }
        tla_trust_cg::ensure_jit_execute_mode();
        Ok(proof_fields)
    }

    fn log_callout_publication_ok(
        library: Option<&tla_trust_cg::NativeLibrary>,
        kind: &str,
        index: u32,
        symbol_name: &str,
        name: &str,
        func_addr: usize,
        publication_proof: Option<&str>,
    ) {
        if let (Some(library), Some(publication_proof)) = (library, publication_proof) {
            eprintln!(
                "[trust_cg-selftest] exact-owner-symbol-publication-ok kind={kind} index={index} symbol={symbol_name} name={name} fn=0x{func_addr:x} {publication_proof} owner_debug={library:?}",
            );
        } else {
            eprintln!(
                "[trust_cg-selftest] registry-publication-ok kind={kind} index={index} symbol={symbol_name} name={name} fn=0x{func_addr:x} owner=none",
            );
        }
        use std::io::Write as _;
        let _ = std::io::stderr().flush();
    }

    fn log_cache_build_without_sample(cache: &TrustCgNativeCache) {
        if !trust_cg_native_callout_selftest_enabled() {
            return;
        }
        telemetry_eprintln!(
            "[trust_cg-selftest] {TRUST_CG_NATIVE_CALLOUT_SELFTEST_ENV} enabled after TrustCgNativeCache::build: compiled actions={}, state_constraints={}, invariants={}, but no real flat state sample is available in cache-build scope; deferring to the first native fused arena if one is built",
            cache.action_count(),
            cache.state_constraint_count(),
            cache.invariant_count(),
        );
    }

    fn log_fused_build_without_sample(state_len: usize) {
        if !trust_cg_native_callout_selftest_enabled() {
            return;
        }
        telemetry_eprintln!(
            "[trust_cg-selftest] {TRUST_CG_NATIVE_CALLOUT_SELFTEST_ENV} enabled before native fused level build: no real flat parent state is in scope (state_len={state_len}); selftest will run on the first native fused arena parent",
        );
    }

    fn from_cache_if_enabled_or_required(
        cache: &TrustCgNativeCache,
        native_actions: &[tla_trust_cg::TrustCgBfsLevelNativeAction],
        native_state_constraints: &[tla_trust_cg::TrustCgBfsLevelNativeStateConstraint],
        native_invariants: &[tla_trust_cg::TrustCgBfsLevelNativeInvariant],
        fail_closed_required: bool,
    ) -> Option<Self> {
        if !fail_closed_required && !trust_cg_native_callout_selftest_enabled() {
            return None;
        }

        let mut missing_expected = Vec::new();

        let mut actions = Vec::with_capacity(native_actions.len());
        for action in native_actions {
            let Some(func) = cache.next_state_fns.get(&action.descriptor.name).copied() else {
                Self::record_missing_expected(
                    &mut missing_expected,
                    "action",
                    action.descriptor.action_idx,
                    &action.symbol_name,
                    &action.descriptor.name,
                );
                continue;
            };
            actions.push(TrustCgNativeCalloutSelftestAction {
                index: action.descriptor.action_idx,
                name: action.descriptor.name.clone(),
                symbol_name: action.symbol_name.clone(),
                func,
                library: Some(action.library.clone()),
            });
        }

        let mut state_constraints = Vec::with_capacity(native_state_constraints.len());
        for constraint in native_state_constraints {
            let idx = constraint.constraint_idx as usize;
            let Some(func) = cache.state_constraint_fns.get(idx).and_then(|slot| *slot) else {
                Self::record_missing_expected(
                    &mut missing_expected,
                    "state-constraint",
                    constraint.constraint_idx,
                    &constraint.symbol_name,
                    &constraint.name,
                );
                continue;
            };
            state_constraints.push(TrustCgNativeCalloutSelftestPredicate {
                index: constraint.constraint_idx,
                name: constraint.name.clone(),
                symbol_name: constraint.symbol_name.clone(),
                func,
                library: Some(constraint.library.clone()),
            });
        }

        let mut invariants = Vec::with_capacity(native_invariants.len());
        for invariant in native_invariants {
            let idx = invariant.descriptor.invariant_idx as usize;
            let Some(func) = cache.invariant_fns.get(idx).and_then(|slot| *slot) else {
                Self::record_missing_expected(
                    &mut missing_expected,
                    "invariant",
                    invariant.descriptor.invariant_idx,
                    &invariant.symbol_name,
                    &invariant.descriptor.name,
                );
                continue;
            };
            invariants.push(TrustCgNativeCalloutSelftestPredicate {
                index: invariant.descriptor.invariant_idx,
                name: invariant.descriptor.name.clone(),
                symbol_name: invariant.symbol_name.clone(),
                func,
                library: Some(invariant.library.clone()),
            });
        }

        let fail_closed = fail_closed_required || trust_cg_native_callout_selftest_fail_closed();
        eprintln!(
            "[trust_cg-selftest] prepared native fused callout selftest: actions={}, state_constraints={}, invariants={}, missing_expected={}, fail_closed={fail_closed}",
            actions.len(),
            state_constraints.len(),
            invariants.len(),
            missing_expected.len(),
        );

        Some(Self {
            actions,
            state_constraints,
            invariants,
            missing_expected,
            fail_closed,
        })
    }

    fn run_on_first_parent(
        &self,
        arena: &[i64],
        parent_count: usize,
        state_len: usize,
    ) -> Result<(), String> {
        if self.fail_closed && !self.missing_expected.is_empty() {
            for missing in &self.missing_expected {
                eprintln!(
                    "[trust_cg-selftest] fail-closed missing expected {} callout function pointer: index={} symbol={} name={}",
                    missing.kind, missing.index, missing.symbol_name, missing.name,
                );
            }
            return Err(format!(
                "missing expected native fused callout function pointer(s): {}",
                self.missing_expected_summary(),
            ));
        }
        if parent_count == 0 {
            return Err("no parent states in fused arena (parent_count=0)".to_string());
        }
        if state_len == 0 {
            return Err("zero-width flat state; no safe real state sample available".to_string());
        }
        let required_slots = parent_count.checked_mul(state_len).ok_or_else(|| {
            format!(
                "fused arena dimensions overflow: parent_count={parent_count}, state_len={state_len}"
            )
        })?;
        if arena.len() < required_slots {
            return Err(format!(
                "fused arena too short for declared parents: parent_count={parent_count}, state_len={state_len}, required_slots={required_slots}, arena_slots={}",
                arena.len()
            ));
        }
        let state_len_u32 = u32::try_from(state_len)
            .map_err(|_| format!("state_len={state_len} does not fit the native callout ABI"))?;
        let sample = &arena[..state_len];
        let mut unexpected_status = false;
        let mut fail_closed_false_predicate = false;
        let selftest_start = std::time::Instant::now();
        let mut action_calls = 0usize;
        let mut predicate_calls = 0usize;

        eprintln!(
            "[trust_cg-selftest] running native fused callout selftest on first real parent: state_len={state_len}, actions={}, state_constraints={}, invariants={}, fail_closed={}",
            self.actions.len(),
            self.state_constraints.len(),
            self.invariants.len(),
            self.fail_closed,
        );

        for action in &self.actions {
            let mut out = native_fused_callout_sentinel();
            let sample_in = TrustCgNativeCalloutGuardedState::new(
                sample,
                TRUST_CG_NATIVE_CALLOUT_SELFTEST_INPUT_CANARY,
            );
            let mut state_out = TrustCgNativeCalloutGuardedState::new(
                sample,
                TRUST_CG_NATIVE_CALLOUT_SELFTEST_OUTPUT_CANARY,
            );
            Self::log_callout_start(
                "action",
                action.index,
                &action.symbol_name,
                &action.name,
                action.func as *const () as usize,
                state_len_u32,
                sample_in.payload(),
                Some(state_out.payload()),
            );
            // SAFETY: Function pointers and ABI metadata come from the same
            // compiled cache used by the native fused level. `sample` is a real
            // flat parent state and `state_out` is a same-width successor buffer.
            Self::clear_tla_runtime_arenas_before_callout();
            let publication_proof = Self::ensure_callout_library_published(
                action.library.as_ref(),
                action.func as *const () as *mut std::ffi::c_void,
                "action",
                action.index,
                &action.symbol_name,
                &action.name,
            )?;
            Self::log_callout_publication_ok(
                action.library.as_ref(),
                "action",
                action.index,
                &action.symbol_name,
                &action.name,
                action.func as *const () as usize,
                publication_proof.as_deref(),
            );
            action_calls += 1;
            unsafe {
                (action.func)(
                    &mut out,
                    sample_in.payload_ptr(),
                    state_out.payload_mut_ptr(),
                    state_len_u32,
                );
            }
            Self::log_callout_out(
                "action",
                action.index,
                &action.symbol_name,
                &action.name,
                out,
            );
            sample_in.verify_read_only(
                "action",
                action.index,
                &action.symbol_name,
                &action.name,
                sample,
            )?;
            state_out.verify_canaries(
                "action state_out",
                action.index,
                &action.symbol_name,
                &action.name,
            )?;
            unexpected_status |= out.status != tla_jit_abi::JitStatus::Ok;

            let action_enabled = Self::decode_ok_boolean_callout(
                "action",
                action.index,
                &action.symbol_name,
                &action.name,
                out,
            )?
            .unwrap_or(false);

            if action_enabled {
                let mut constraints_passed = true;
                for constraint in &self.state_constraints {
                    eprintln!(
                        "[trust_cg-selftest] checking state_constraint after action index={} symbol={} name={} constraint_index={} constraint_symbol={} constraint_name={}",
                        action.index,
                        action.symbol_name,
                        action.name,
                        constraint.index,
                        constraint.symbol_name,
                        constraint.name,
                    );
                    predicate_calls += 1;
                    let out = self.run_predicate_callout(
                        "state_constraint_after_action",
                        constraint,
                        state_out.payload(),
                        state_len_u32,
                    )?;
                    unexpected_status |= out.status != tla_jit_abi::JitStatus::Ok;
                    let constraint_passed = Self::decode_ok_predicate_callout(
                        "state_constraint_after_action",
                        constraint,
                        out,
                    )?
                    .unwrap_or(false);
                    constraints_passed &= constraint_passed;
                }
                if constraints_passed {
                    for invariant in &self.invariants {
                        eprintln!(
                            "[trust_cg-selftest] checking invariant after action index={} symbol={} name={} invariant_index={} invariant_symbol={} invariant_name={}",
                            action.index,
                            action.symbol_name,
                            action.name,
                            invariant.index,
                            invariant.symbol_name,
                            invariant.name,
                        );
                        predicate_calls += 1;
                        let out = self.run_predicate_callout(
                            "invariant_after_action",
                            invariant,
                            state_out.payload(),
                            state_len_u32,
                        )?;
                        unexpected_status |= out.status != tla_jit_abi::JitStatus::Ok;
                        let _ = Self::decode_ok_predicate_callout(
                            "invariant_after_action",
                            invariant,
                            out,
                        )?;
                    }
                }
            }
        }

        for constraint in &self.state_constraints {
            predicate_calls += 1;
            let out =
                self.run_predicate_callout("state_constraint", constraint, sample, state_len_u32)?;
            unexpected_status |= out.status != tla_jit_abi::JitStatus::Ok;
            let predicate_result =
                Self::decode_ok_predicate_callout("state_constraint", constraint, out)?;
            if self.fail_closed
                && Self::standalone_predicate_failed_closed(
                    "state_constraint",
                    constraint,
                    predicate_result,
                )
            {
                fail_closed_false_predicate = true;
            }
        }

        for invariant in &self.invariants {
            predicate_calls += 1;
            let out = self.run_predicate_callout("invariant", invariant, sample, state_len_u32)?;
            unexpected_status |= out.status != tla_jit_abi::JitStatus::Ok;
            let predicate_result = Self::decode_ok_predicate_callout("invariant", invariant, out)?;
            if self.fail_closed
                && Self::standalone_predicate_failed_closed(
                    "invariant",
                    invariant,
                    predicate_result,
                )
            {
                fail_closed_false_predicate = true;
            }
        }

        if trust_cg_setup_timing_enabled() {
            eprintln!(
                "[trust_cg-timing] native_callout_selftest_ms={} action_calls={} predicate_calls={} actions={} state_constraints={} invariants={} fail_closed={}",
                selftest_start.elapsed().as_millis(),
                action_calls,
                predicate_calls,
                self.actions.len(),
                self.state_constraints.len(),
                self.invariants.len(),
                self.fail_closed,
            );
        }

        if unexpected_status {
            return Err("one or more native fused callouts returned a non-Ok status".to_string());
        }
        if fail_closed_false_predicate {
            return Err(
                "one or more fail-closed standalone native fused predicate callouts returned Ok(value=0)"
                    .to_string(),
            );
        }
        Ok(())
    }

    fn record_missing_expected(
        missing_expected: &mut Vec<TrustCgNativeCalloutSelftestMissing>,
        kind: &'static str,
        index: u32,
        symbol_name: &str,
        name: &str,
    ) {
        eprintln!(
            "[trust_cg-selftest] {kind} callout metadata missing function pointer: index={index} symbol={symbol_name} name={name}",
        );
        missing_expected.push(TrustCgNativeCalloutSelftestMissing {
            kind,
            index,
            name: name.to_string(),
            symbol_name: symbol_name.to_string(),
        });
    }

    fn missing_expected_summary(&self) -> String {
        self.missing_expected
            .iter()
            .map(|missing| {
                format!(
                    "{} index={} symbol={} name={}",
                    missing.kind, missing.index, missing.symbol_name, missing.name,
                )
            })
            .collect::<Vec<_>>()
            .join("; ")
    }

    fn run_predicate_callout(
        &self,
        kind: &str,
        callout: &TrustCgNativeCalloutSelftestPredicate,
        sample: &[i64],
        state_len: u32,
    ) -> Result<JitCallOut, String> {
        let mut out = native_fused_callout_sentinel();
        let guarded_sample = TrustCgNativeCalloutGuardedState::new(
            sample,
            TRUST_CG_NATIVE_CALLOUT_SELFTEST_INPUT_CANARY,
        );
        Self::log_callout_start(
            kind,
            callout.index,
            &callout.symbol_name,
            &callout.name,
            callout.func as *const () as usize,
            state_len,
            guarded_sample.payload(),
            None,
        );
        // SAFETY: Function pointer and ABI metadata come from the compiled
        // native cache. `sample` is a real flat parent state with `state_len`
        // addressable i64 slots.
        Self::clear_tla_runtime_arenas_before_callout();
        let publication_proof = Self::ensure_callout_library_published(
            callout.library.as_ref(),
            callout.func as *const () as *mut std::ffi::c_void,
            kind,
            callout.index,
            &callout.symbol_name,
            &callout.name,
        )?;
        Self::log_callout_publication_ok(
            callout.library.as_ref(),
            kind,
            callout.index,
            &callout.symbol_name,
            &callout.name,
            callout.func as *const () as usize,
            publication_proof.as_deref(),
        );
        unsafe {
            (callout.func)(&mut out, guarded_sample.payload_ptr(), state_len);
        }
        Self::log_callout_out(
            kind,
            callout.index,
            &callout.symbol_name,
            &callout.name,
            out,
        );
        guarded_sample.verify_read_only(
            kind,
            callout.index,
            &callout.symbol_name,
            &callout.name,
            sample,
        )?;
        Ok(out)
    }

    fn decode_ok_predicate_callout(
        kind: &str,
        callout: &TrustCgNativeCalloutSelftestPredicate,
        out: JitCallOut,
    ) -> Result<Option<bool>, String> {
        Self::decode_ok_boolean_callout(
            kind,
            callout.index,
            &callout.symbol_name,
            &callout.name,
            out,
        )
    }

    fn decode_ok_boolean_callout(
        kind: &str,
        index: u32,
        symbol_name: &str,
        name: &str,
        out: JitCallOut,
    ) -> Result<Option<bool>, String> {
        if out.status != tla_jit_abi::JitStatus::Ok {
            return Ok(None);
        }
        match out.value {
            0 => Ok(Some(false)),
            1 => Ok(Some(true)),
            value => {
                let reason = format!(
                    "native fused {kind} callout returned noncanonical boolean value {value}: index={index} symbol={symbol_name} name={name}; strict ABI requires 0 or 1"
                );
                eprintln!("[trust_cg-selftest] {reason}");
                Err(reason)
            }
        }
    }

    fn standalone_predicate_failed_closed(
        kind: &str,
        callout: &TrustCgNativeCalloutSelftestPredicate,
        predicate_result: Option<bool>,
    ) -> bool {
        if predicate_result == Some(false) {
            eprintln!(
                "[trust_cg-selftest] fail-closed standalone {kind} callout returned Ok(value=0): index={} symbol={} name={}",
                callout.index, callout.symbol_name, callout.name,
            );
            return true;
        }
        false
    }

    fn log_callout_start(
        kind: &str,
        index: u32,
        symbol_name: &str,
        name: &str,
        func_addr: usize,
        state_len: u32,
        sample: &[i64],
        state_out: Option<&[i64]>,
    ) {
        maybe_write_callout_replay_artifact(
            kind,
            index,
            symbol_name,
            name,
            func_addr,
            state_len,
            sample,
            state_out,
        );
        let sample_head = &sample[..sample.len().min(16)];
        if let Some(state_out) = state_out {
            let out_head = &state_out[..state_out.len().min(16)];
            eprintln!(
                "[trust_cg-selftest] entering {kind} callout index={index} symbol={symbol_name} name={name} fn=0x{func_addr:x} state_len={state_len} state_head={sample_head:?} out_head={out_head:?}",
            );
        } else {
            eprintln!(
                "[trust_cg-selftest] entering {kind} callout index={index} symbol={symbol_name} name={name} fn=0x{func_addr:x} state_len={state_len} state_head={sample_head:?}",
            );
        }
        use std::io::Write as _;
        let _ = std::io::stderr().flush();
    }

    fn log_callout_out(kind: &str, index: u32, symbol_name: &str, name: &str, out: JitCallOut) {
        eprintln!(
            "[trust_cg-selftest] {kind} callout index={index} symbol={symbol_name} name={name} status={:?} value={}",
            out.status, out.value,
        );
        if out.status == tla_jit_abi::JitStatus::RuntimeError {
            eprintln!(
                "[trust_cg-selftest] {kind} callout runtime_error index={index} symbol={symbol_name} err_kind={:?} span={}..{} file_id={}",
                out.err_kind, out.err_span_start, out.err_span_end, out.err_file_id,
            );
        }
    }
}

// SAFETY: After construction, the cache is immutable. Function pointers target
// finalized native code in dlopen'd libraries that are retained for the cache
// lifetime. No interior mutation.
unsafe impl Send for TrustCgNativeCache {}
unsafe impl Sync for TrustCgNativeCache {}

fn count_arity_positive_action_failure(
    exists_enabled: bool,
    specialized_action_names: &FxHashSet<&str>,
    action_name: &str,
) -> bool {
    !(exists_enabled && specialized_action_names.contains(action_name))
}

fn entry_counter_gate_allows_dispatch(entry_count: Option<u64>, limit: u64) -> bool {
    match entry_count {
        Some(count) => count < limit,
        None => false,
    }
}

fn decode_strict_trust_cg_boolean(value: i64, context: &str) -> Result<bool, ()> {
    match value {
        0 => Ok(false),
        1 => Ok(true),
        _ => {
            eprintln!(
                "[trust-cg] {context} returned noncanonical boolean value {value}; strict ABI requires 0 or 1"
            );
            Err(())
        }
    }
}

/// Result of evaluating a single trust_cg-compiled action.
pub(in crate::check) enum TrustCgActionResult {
    /// Action is enabled; `successor` contains the flat i64 output buffer.
    ///
    /// The input (predecessor) state buffer needed by
    /// `unflatten_i64_to_array_state_with_input` for compound value
    /// deserialization (offsets encoded in output reference the input buffer)
    /// is NOT carried here: it is invariant across every dispatch of one
    /// predecessor, so the caller supplies the shared `&jit_state_scratch`
    /// buffer once rather than cloning it per enabled successor.
    Enabled { successor: Vec<i64> },
    /// Action guard evaluated to false; no successor state.
    // Constructed only by the test-exercised `eval_action_with_state_len`
    // path and pattern-matched across the dispatch test suite; keep the
    // variant so those match arms and the result shape stay intact.
    #[allow(dead_code)]
    Disabled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(in crate::check) enum TrustCgActionCalloutBatchFallbackReason {
    #[default]
    NotAttempted,
    NoFallback,
    NoLoweredTasks,
    SingleLoweredTask,
    MixedOptLevels,
    TrustIrBatchAssembly,
    SymbolContract,
    BatchCompile,
    BatchSymbolLookup,
}

impl TrustCgActionCalloutBatchFallbackReason {
    fn code(self) -> &'static str {
        match self {
            Self::NotAttempted => "not_attempted",
            Self::NoFallback => "none",
            Self::NoLoweredTasks => "no_lowered_tasks",
            Self::SingleLoweredTask => "single_lowered_task",
            Self::MixedOptLevels => "mixed_opt_levels",
            Self::TrustIrBatchAssembly => "trust_ir_batch_assembly",
            Self::SymbolContract => "symbol_contract",
            Self::BatchCompile => "batch_compile",
            Self::BatchSymbolLookup => "batch_symbol_lookup",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TrustCgActionCalloutBatchFallback {
    reason: TrustCgActionCalloutBatchFallbackReason,
    message: String,
}

impl TrustCgActionCalloutBatchFallback {
    fn new(reason: TrustCgActionCalloutBatchFallbackReason, message: impl ToString) -> Self {
        Self {
            reason,
            message: message.to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TrustCgActionCalloutShard {
    input_indices: Vec<usize>,
    estimated_ir_nodes: usize,
    stable_id: String,
    shared_shape_id: String,
    frontend_neutral_reuse_id: String,
    digest_input_sha256: String,
}

impl TrustCgActionCalloutShard {
    fn action_count(&self) -> usize {
        self.input_indices.len()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TrustCgActionCalloutShardPlan {
    policy_selected: bool,
    shards: Vec<TrustCgActionCalloutShard>,
    estimated_ir_nodes: usize,
    trust_ir_batch_partition_plan_reuse_manifest_id: String,
}

struct TrustCgNativeActionCalloutShardInput<'a> {
    batch_module_name: String,
    tasks: Vec<(usize, &'a TrustCgLoweredActionCompileTask)>,
    metadata: TrustCgNativeActionBatchShardCompileMetadata,
}

struct TrustCgNativeActionCalloutShardResult {
    indexed_outcomes: Vec<(usize, TrustCgActionCompileOutcome)>,
    stats: TrustCgNativeActionCalloutBatchStats,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TrustCgNativeBatchRuntimeSetupDescriptor {
    semantic_trust_ir_artifact_digest: String,
    process_local_link_digest: String,
    artifact_cache_digest: String,
    compile_preset: &'static str,
    host_symbol_map_count: usize,
    artifact_admission_status: &'static str,
    artifact_admission_fail_closed: bool,
    artifact_admission_missing_fields: Vec<&'static str>,
    artifact_admission_rejection_reasons: Vec<&'static str>,
}

impl TrustCgNativeBatchRuntimeSetupDescriptor {
    fn from_batch_jit_stats(stats: &tla_trust_cg::BatchJitStats) -> Self {
        let telemetry = stats.compile_telemetry();
        let admission =
            tla_trust_cg::admit_batch_jit_artifact(tla_trust_cg::BatchJitArtifactAdmissionInput {
                semantic_trust_ir_artifact_digest: Some(&telemetry.semantic_digest),
                process_local_link_digest: Some(&telemetry.link_digest),
                compile_preset: Some(telemetry.compile_preset),
                opt_level: Some(telemetry.effective_opt_level),
                host_symbol_map_count: Some(telemetry.host_symbol_map_count),
                function_count: Some(stats.function_count),
            });

        Self {
            semantic_trust_ir_artifact_digest: telemetry.semantic_digest,
            process_local_link_digest: telemetry.link_digest,
            artifact_cache_digest: telemetry.cache_digest,
            compile_preset: telemetry.compile_preset.as_str(),
            host_symbol_map_count: telemetry.host_symbol_map_count,
            artifact_admission_status: admission.status.as_str(),
            artifact_admission_fail_closed: admission.is_fail_closed(),
            artifact_admission_missing_fields: admission.missing_fields,
            artifact_admission_rejection_reasons: admission.rejection_reasons,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TrustCgNativeBatchRuntimeCacheLabels {
    temperature: &'static str,
    cache: &'static str,
}

impl TrustCgNativeBatchRuntimeCacheLabels {
    fn from_warm_cache_status(status: &str) -> Self {
        match status {
            "hit" => Self {
                temperature: "warm",
                cache: "warm_cache_hit",
            },
            "miss" => Self {
                temperature: "cold",
                cache: "cold_cache_miss",
            },
            "guard_miss" => Self {
                temperature: "cold",
                cache: "cold_cache_guard_miss",
            },
            "disabled" => Self {
                temperature: "cold",
                cache: "cold_cache_disabled",
            },
            "lock_error" => Self {
                temperature: "cold",
                cache: "cold_cache_lock_error",
            },
            _ => Self {
                temperature: "cold",
                cache: "cold_cache_unknown",
            },
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(in crate::check) struct TrustCgNativeActionCalloutBatchStats {
    pub attempted: bool,
    pub action_count: usize,
    pub input_tasks: usize,
    pub lowered_tasks: usize,
    pub lowering_failed: usize,
    pub setup_ms: u64,
    pub lowering_ms: u64,
    pub batch_assembly_attempted: bool,
    pub batch_assembly_ms: u64,
    pub batch_assembly_failed: bool,
    pub batch_compile_attempted: bool,
    pub batch_compile_ms: u64,
    pub batch_compile_failed: bool,
    pub batch_compiled: usize,
    pub sharding_policy_selected: bool,
    pub shard_count: usize,
    pub shard_action_counts: Vec<usize>,
    pub shard_estimated_ir_nodes: Vec<usize>,
    pub shard_assembly_ms: Vec<u64>,
    pub shard_compile_ms: Vec<u64>,
    pub shard_stable_ids: Vec<String>,
    pub shard_shared_shape_ids: Vec<String>,
    pub shard_frontend_neutral_reuse_ids: Vec<String>,
    pub shard_digest_input_sha256s: Vec<String>,
    pub trust_ir_batch_partition_plan_reuse_manifest_id: Option<String>,
    pub warm_cache_enabled: bool,
    pub warm_cache_lookup_attempted: bool,
    pub warm_cache_hits: usize,
    pub warm_cache_misses: usize,
    pub warm_cache_stores: usize,
    pub warm_cache_lookup_ms: u64,
    pub shard_warm_cache_lookup_ms: Vec<u64>,
    pub shard_warm_cache_statuses: Vec<String>,
    pub shard_warm_cache_keys: Vec<String>,
    pub shard_warm_cache_guard_digests: Vec<String>,
    pub artifact_materialization_ms: u64,
    pub shard_artifact_materialization_ms: Vec<u64>,
    pub estimated_ir_nodes: usize,
    pub fallback_per_action_tasks: usize,
    pub fallback_per_action_compile_ms: u64,
    pub fallback_reason: TrustCgActionCalloutBatchFallbackReason,
    pub artifact_identity_source: Option<&'static str>,
    pub artifact_identity: Option<String>,
    pub artifact_semantic_digest: Option<String>,
    pub artifact_link_digest: Option<String>,
    pub artifact_cache_digest: Option<String>,
    pub batch_compile_preset: Option<String>,
    pub host_symbol_map_count: Option<usize>,
    pub runtime_setup_temperature_label: Option<&'static str>,
    pub runtime_setup_cache_label: Option<&'static str>,
    pub batch_artifact_admission_status: Option<String>,
    pub batch_artifact_admission_fail_closed: Option<bool>,
    pub artifact_cacheable: bool,
    pub artifact_cache_disabled_by_env: bool,
    pub prepared_trust_ir_reuse: Option<&'static str>,
    pub prepared_trust_ir_reuse_identity: Option<String>,
    pub shared_owner: Option<&'static str>,
    pub first_beneficiary: Option<&'static str>,
    pub second_beneficiary: Option<&'static str>,
    pub extraction_status: Option<&'static str>,
    pub compile_telemetry_evidence_row: Option<String>,
    pub shared_engine_adoption_evidence_row: Option<String>,
    pub artifact_identities: Vec<String>,
    pub artifact_semantic_digests: Vec<String>,
    pub artifact_link_digests: Vec<String>,
    pub artifact_cache_digests: Vec<String>,
    pub batch_compile_presets: Vec<String>,
    pub shard_host_symbol_map_counts: Vec<usize>,
    pub runtime_setup_temperature_labels: Vec<String>,
    pub runtime_setup_cache_labels: Vec<String>,
    pub batch_artifact_admission_statuses: Vec<String>,
    pub batch_artifact_admission_fail_closed_values: Vec<bool>,
    pub batch_artifact_admission_missing_fields: Vec<String>,
    pub batch_artifact_admission_rejection_reasons: Vec<String>,
    pub compile_telemetry_evidence_rows: Vec<String>,
    pub shared_engine_adoption_evidence_rows: Vec<String>,
}

impl TrustCgNativeActionCalloutBatchStats {
    fn attempted(input_tasks: usize) -> Self {
        Self {
            attempted: true,
            action_count: input_tasks,
            input_tasks,
            artifact_cache_disabled_by_env: trust_cg_artifact_cache_disabled_by_env(),
            warm_cache_enabled: trust_cg_process_local_warm_artifact_cache_enabled(),
            ..Self::default()
        }
    }

    fn fallback_reason_code(&self) -> &'static str {
        self.fallback_reason.code()
    }

    fn optional_value(value: Option<&str>) -> &str {
        value.unwrap_or("none")
    }

    fn csv_usize(values: &[usize]) -> String {
        if values.is_empty() {
            return "none".to_string();
        }
        values
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",")
    }

    fn csv_u64(values: &[u64]) -> String {
        if values.is_empty() {
            return "none".to_string();
        }
        values
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",")
    }

    fn csv_bools(values: &[bool]) -> String {
        if values.is_empty() {
            return "none".to_string();
        }
        values
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",")
    }

    fn csv_strings(values: &[String]) -> String {
        if values.is_empty() {
            return "none".to_string();
        }
        values.join(",")
    }

    fn optional_usize(value: Option<usize>) -> String {
        value.map_or_else(|| "none".to_string(), |value| value.to_string())
    }

    fn optional_bool(value: Option<bool>) -> &'static str {
        match value {
            Some(true) => "true",
            Some(false) => "false",
            None => "none",
        }
    }

    fn evidence_rows_sha256(
        single_label: &str,
        multi_label: &str,
        rows: &[String],
        fallback_row: Option<&str>,
    ) -> Option<String> {
        if rows.len() == 1 {
            return Some(trust_cg_native_admission_sha256(single_label, &rows[0]));
        }
        if !rows.is_empty() {
            return Some(trust_cg_native_admission_sha256(
                multi_label,
                &rows.join("\n"),
            ));
        }
        fallback_row.map(|row| trust_cg_native_admission_sha256(single_label, row))
    }

    fn compile_telemetry_evidence_row_sha256(&self) -> Option<String> {
        Self::evidence_rows_sha256(
            "native_action_callout_batch_compile_row",
            "native_action_callout_batch_compile_rows",
            &self.compile_telemetry_evidence_rows,
            self.compile_telemetry_evidence_row.as_deref(),
        )
    }

    fn shared_engine_adoption_evidence_row_sha256(&self) -> Option<String> {
        Self::evidence_rows_sha256(
            "native_action_callout_batch_adoption_row",
            "native_action_callout_batch_adoption_rows",
            &self.shared_engine_adoption_evidence_rows,
            self.shared_engine_adoption_evidence_row.as_deref(),
        )
    }

    fn record_shard_plan(&mut self, plan: &TrustCgActionCalloutShardPlan) {
        self.sharding_policy_selected = plan.policy_selected;
        self.shard_count = plan.shards.len();
        self.record_trust_ir_batch_partition_plan_reuse_manifest_id(
            &plan.trust_ir_batch_partition_plan_reuse_manifest_id,
        );
        self.shard_action_counts = plan
            .shards
            .iter()
            .map(TrustCgActionCalloutShard::action_count)
            .collect();
        self.shard_estimated_ir_nodes = plan
            .shards
            .iter()
            .map(|shard| shard.estimated_ir_nodes)
            .collect();
        self.shard_stable_ids = plan
            .shards
            .iter()
            .map(|shard| shard.stable_id.clone())
            .collect();
        self.shard_shared_shape_ids = plan
            .shards
            .iter()
            .map(|shard| shard.shared_shape_id.clone())
            .collect();
        self.shard_frontend_neutral_reuse_ids = plan
            .shards
            .iter()
            .map(|shard| shard.frontend_neutral_reuse_id.clone())
            .collect();
        self.shard_digest_input_sha256s = plan
            .shards
            .iter()
            .map(|shard| shard.digest_input_sha256.clone())
            .collect();
        self.estimated_ir_nodes = plan.estimated_ir_nodes;
    }

    fn record_trust_ir_batch_partition_plan_reuse_manifest_id(&mut self, manifest_id: &str) {
        if !manifest_id.is_empty() {
            self.trust_ir_batch_partition_plan_reuse_manifest_id = Some(manifest_id.to_string());
        }
    }

    fn record_warm_cache_lookup(
        &mut self,
        key: &TrustCgNativeBatchWarmArtifactKey,
        status: &'static str,
        lookup_ms: u64,
    ) {
        self.warm_cache_lookup_attempted = true;
        self.warm_cache_lookup_ms = self.warm_cache_lookup_ms.saturating_add(lookup_ms);
        self.shard_warm_cache_lookup_ms.push(lookup_ms);
        self.shard_warm_cache_statuses.push(status.to_string());
        self.shard_warm_cache_keys
            .push(key.shared_engine_identity.clone());
        self.shard_warm_cache_guard_digests
            .push(key.guard_digest.clone());
        match status {
            "hit" => self.warm_cache_hits += 1,
            _ => self.warm_cache_misses += 1,
        }
        let labels = TrustCgNativeBatchRuntimeCacheLabels::from_warm_cache_status(status);
        self.runtime_setup_temperature_label = Some(labels.temperature);
        self.runtime_setup_cache_label = Some(labels.cache);
        self.runtime_setup_temperature_labels
            .push(labels.temperature.to_string());
        self.runtime_setup_cache_labels
            .push(labels.cache.to_string());
    }

    fn record_warm_cache_store(&mut self) {
        self.warm_cache_stores += 1;
    }

    fn record_artifact_materialization(&mut self, materialization_ms: u64) {
        self.artifact_materialization_ms = self
            .artifact_materialization_ms
            .saturating_add(materialization_ms);
        self.shard_artifact_materialization_ms
            .push(materialization_ms);
    }

    fn merge_compiled_shard_stats(&mut self, shard: TrustCgNativeActionCalloutBatchStats) {
        self.batch_assembly_attempted |= shard.batch_assembly_attempted;
        self.batch_assembly_ms = self
            .batch_assembly_ms
            .saturating_add(shard.batch_assembly_ms);
        self.batch_assembly_failed |= shard.batch_assembly_failed;
        self.batch_compile_attempted |= shard.batch_compile_attempted;
        self.batch_compile_ms = self.batch_compile_ms.saturating_add(shard.batch_compile_ms);
        self.batch_compile_failed |= shard.batch_compile_failed;
        self.warm_cache_lookup_attempted |= shard.warm_cache_lookup_attempted;
        self.warm_cache_hits = self.warm_cache_hits.saturating_add(shard.warm_cache_hits);
        self.warm_cache_misses = self
            .warm_cache_misses
            .saturating_add(shard.warm_cache_misses);
        self.warm_cache_stores = self
            .warm_cache_stores
            .saturating_add(shard.warm_cache_stores);
        self.warm_cache_lookup_ms = self
            .warm_cache_lookup_ms
            .saturating_add(shard.warm_cache_lookup_ms);
        self.artifact_materialization_ms = self
            .artifact_materialization_ms
            .saturating_add(shard.artifact_materialization_ms);
        if self
            .trust_ir_batch_partition_plan_reuse_manifest_id
            .is_none()
        {
            self.trust_ir_batch_partition_plan_reuse_manifest_id =
                shard.trust_ir_batch_partition_plan_reuse_manifest_id;
        }
        if shard.fallback_reason != TrustCgActionCalloutBatchFallbackReason::NotAttempted
            && shard.fallback_reason != TrustCgActionCalloutBatchFallbackReason::NoFallback
        {
            self.fallback_reason = shard.fallback_reason;
        }

        self.shard_assembly_ms.extend(shard.shard_assembly_ms);
        self.shard_compile_ms.extend(shard.shard_compile_ms);
        self.shard_warm_cache_lookup_ms
            .extend(shard.shard_warm_cache_lookup_ms);
        self.shard_warm_cache_statuses
            .extend(shard.shard_warm_cache_statuses);
        self.shard_warm_cache_keys
            .extend(shard.shard_warm_cache_keys);
        self.shard_warm_cache_guard_digests
            .extend(shard.shard_warm_cache_guard_digests);
        self.shard_artifact_materialization_ms
            .extend(shard.shard_artifact_materialization_ms);
        self.artifact_identities.extend(shard.artifact_identities);
        self.artifact_semantic_digests
            .extend(shard.artifact_semantic_digests);
        self.artifact_link_digests
            .extend(shard.artifact_link_digests);
        self.artifact_cache_digests
            .extend(shard.artifact_cache_digests);
        self.batch_compile_presets
            .extend(shard.batch_compile_presets);
        self.shard_host_symbol_map_counts
            .extend(shard.shard_host_symbol_map_counts);
        self.runtime_setup_temperature_labels
            .extend(shard.runtime_setup_temperature_labels);
        self.runtime_setup_cache_labels
            .extend(shard.runtime_setup_cache_labels);
        self.batch_artifact_admission_statuses
            .extend(shard.batch_artifact_admission_statuses);
        self.batch_artifact_admission_fail_closed_values
            .extend(shard.batch_artifact_admission_fail_closed_values);
        self.compile_telemetry_evidence_rows
            .extend(shard.compile_telemetry_evidence_rows);
        self.shared_engine_adoption_evidence_rows
            .extend(shard.shared_engine_adoption_evidence_rows);

        self.artifact_identity_source = shard
            .artifact_identity_source
            .or(self.artifact_identity_source);
        self.artifact_identity = shard.artifact_identity.or(self.artifact_identity.take());
        self.artifact_semantic_digest = shard
            .artifact_semantic_digest
            .or(self.artifact_semantic_digest.take());
        self.artifact_link_digest = shard
            .artifact_link_digest
            .or(self.artifact_link_digest.take());
        self.artifact_cache_digest = shard
            .artifact_cache_digest
            .or(self.artifact_cache_digest.take());
        self.batch_compile_preset = shard
            .batch_compile_preset
            .or(self.batch_compile_preset.take());
        self.host_symbol_map_count = shard.host_symbol_map_count.or(self.host_symbol_map_count);
        self.runtime_setup_temperature_label = shard
            .runtime_setup_temperature_label
            .or(self.runtime_setup_temperature_label);
        self.runtime_setup_cache_label = shard
            .runtime_setup_cache_label
            .or(self.runtime_setup_cache_label);
        self.batch_artifact_admission_status = shard
            .batch_artifact_admission_status
            .or(self.batch_artifact_admission_status.take());
        self.batch_artifact_admission_fail_closed = shard
            .batch_artifact_admission_fail_closed
            .or(self.batch_artifact_admission_fail_closed);
        self.artifact_cacheable |= shard.artifact_cacheable;
        self.artifact_cache_disabled_by_env = trust_cg_artifact_cache_disabled_by_env();
        self.warm_cache_enabled = trust_cg_process_local_warm_artifact_cache_enabled();
        self.prepared_trust_ir_reuse = shard
            .prepared_trust_ir_reuse
            .or(self.prepared_trust_ir_reuse);
        self.prepared_trust_ir_reuse_identity = shard
            .prepared_trust_ir_reuse_identity
            .or(self.prepared_trust_ir_reuse_identity.take());
        self.shared_owner = shard.shared_owner.or(self.shared_owner);
        self.first_beneficiary = shard.first_beneficiary.or(self.first_beneficiary);
        self.second_beneficiary = shard.second_beneficiary.or(self.second_beneficiary);
        self.extraction_status = shard.extraction_status.or(self.extraction_status);
        self.compile_telemetry_evidence_row = shard
            .compile_telemetry_evidence_row
            .or(self.compile_telemetry_evidence_row.take());
        self.shared_engine_adoption_evidence_row = shard
            .shared_engine_adoption_evidence_row
            .or(self.shared_engine_adoption_evidence_row.take());
    }

    fn refresh_compiled_artifact_aggregate(&mut self) {
        if self.artifact_identities.len() <= 1 {
            return;
        }
        self.artifact_identity_source = Some("trust_cg_compiled_batch_shard_stats");
        let artifact_digest = trust_cg_native_admission_sha256(
            "native_action_callout_batch_shard_artifact_identities",
            &self.artifact_identities.join(","),
        );
        self.artifact_identity = Some(format!("trust_cg_batch_jit_shards:{artifact_digest}"));
        self.artifact_semantic_digest = Some(trust_cg_native_admission_sha256(
            "native_action_callout_batch_shard_semantic_digests",
            &self.artifact_semantic_digests.join(","),
        ));
        self.artifact_link_digest = Some(trust_cg_native_admission_sha256(
            "native_action_callout_batch_shard_link_digests",
            &self.artifact_link_digests.join(","),
        ));
        self.artifact_cache_digest = Some(trust_cg_native_admission_sha256(
            "native_action_callout_batch_shard_cache_digests",
            &self.artifact_cache_digests.join(","),
        ));
        if !self.batch_compile_presets.is_empty() {
            let first = self.batch_compile_presets[0].as_str();
            self.batch_compile_preset = Some(
                if self
                    .batch_compile_presets
                    .iter()
                    .all(|preset| preset == first)
                {
                    first.to_string()
                } else {
                    "mixed".to_string()
                },
            );
        }
        if !self.shard_host_symbol_map_counts.is_empty() {
            self.host_symbol_map_count = Some(self.shard_host_symbol_map_counts.iter().sum());
        }
        if !self.batch_artifact_admission_statuses.is_empty() {
            let first = self.batch_artifact_admission_statuses[0].as_str();
            self.batch_artifact_admission_status = Some(
                if self
                    .batch_artifact_admission_statuses
                    .iter()
                    .all(|status| status == first)
                {
                    first.to_string()
                } else {
                    "mixed".to_string()
                },
            );
        }
        if !self.batch_artifact_admission_fail_closed_values.is_empty() {
            self.batch_artifact_admission_fail_closed = Some(
                self.batch_artifact_admission_fail_closed_values
                    .iter()
                    .any(|fail_closed| *fail_closed),
            );
        }
        self.artifact_cache_disabled_by_env = trust_cg_artifact_cache_disabled_by_env();
        self.warm_cache_enabled = trust_cg_process_local_warm_artifact_cache_enabled();
        self.artifact_cacheable = !self.artifact_cache_disabled_by_env
            && !self.artifact_cache_digests.is_empty()
            && self
                .artifact_cache_digests
                .iter()
                .all(|digest| !digest.is_empty());
    }

    fn setup_evidence_row(&self) -> Option<String> {
        if !self.attempted {
            return None;
        }
        let fingerprint_admission =
            KernelFingerprintAdmissionContract::for_frontend(&KernelFrontend::Tla);
        let batch_telemetry_descriptor = tla_trust_cg::batch_jit_compile_telemetry_descriptor();
        Some(format!(
            "trust-cg {} schema={} schema_version={} attempted={} action_count={} input_tasks={} lowered_tasks={} lowering_failed={} setup_ms={} lowering_ms={} assembly_attempted={} assembly_ms={} assembly_failed={} compile_attempted={} compile_ms={} compile_failed={} batch_compiled={} sharding_policy_selected={} shard_count={} shard_action_counts={} shard_estimated_ir_nodes={} shard_assembly_ms={} shard_compile_ms={} shard_stable_ids={} shard_shared_shape_ids={} shard_frontend_neutral_reuse_ids={} shard_digest_input_sha256s={} warm_cache_enabled={} warm_cache_lookup_attempted={} warm_cache_hits={} warm_cache_misses={} warm_cache_stores={} warm_cache_lookup_ms={} shard_warm_cache_lookup_ms={} shard_warm_cache_statuses={} shard_warm_cache_keys={} shard_warm_cache_guard_digests={} artifact_materialization_ms={} shard_artifact_materialization_ms={} estimated_ir_nodes={} fallback_per_action_tasks={} fallback_per_action_compile_ms={} fallback_reason={} artifact_identity_source={} artifact_identity={} artifact_semantic_digest={} artifact_link_digest={} artifact_cache_digest={} semantic_trust_ir_artifact_digest={} process_local_link_digest={} artifact_semantic_digests={} artifact_link_digests={} batch_compile_telemetry_schema={} batch_compile_telemetry_schema_version={} batch_compile_telemetry_row_kind={} batch_compile_preset={} batch_compile_presets={} host_symbol_map_count={} shard_host_symbol_map_counts={} runtime_setup_temperature_label={} runtime_setup_temperature_labels={} runtime_setup_cache_label={} runtime_setup_cache_labels={} batch_artifact_admission_schema={} batch_artifact_admission_schema_version={} batch_artifact_admission_status={} batch_artifact_admission_statuses={} batch_artifact_admission_fail_closed={} batch_artifact_admission_fail_closed_values={} batch_artifact_admission_missing_fields={} batch_artifact_admission_rejection_reasons={} artifact_cacheable={} artifact_cache_disabled_by_env={} artifact_count={} artifact_identities={} artifact_cache_digests={} prepared_trust_ir_reuse={} prepared_trust_ir_reuse_identity={} fingerprint_admission_surface={} fingerprint_admission_semantics={} fingerprint_admission_compatible_frontend_families={} fingerprint_admission_default_frontend_families={} fingerprint_admission_blocked_frontend_families={} shared_owner={} first_beneficiary={} second_beneficiary={} extraction_status={} compile_telemetry_row_count={} shared_engine_adoption_row_count={} compile_telemetry_row_sha256={} shared_engine_adoption_row_sha256={}",
            TRUST_CG_NATIVE_ACTION_CALLOUT_BATCH_SETUP_ROW_KIND,
            TRUST_CG_NATIVE_ACTION_CALLOUT_BATCH_SETUP_SCHEMA,
            TRUST_CG_NATIVE_ACTION_CALLOUT_BATCH_SETUP_SCHEMA_VERSION,
            self.attempted,
            self.action_count,
            self.input_tasks,
            self.lowered_tasks,
            self.lowering_failed,
            self.setup_ms,
            self.lowering_ms,
            self.batch_assembly_attempted,
            self.batch_assembly_ms,
            self.batch_assembly_failed,
            self.batch_compile_attempted,
            self.batch_compile_ms,
            self.batch_compile_failed,
            self.batch_compiled,
            self.sharding_policy_selected,
            self.shard_count,
            Self::csv_usize(&self.shard_action_counts),
            Self::csv_usize(&self.shard_estimated_ir_nodes),
            Self::csv_u64(&self.shard_assembly_ms),
            Self::csv_u64(&self.shard_compile_ms),
            Self::csv_strings(&self.shard_stable_ids),
            Self::csv_strings(&self.shard_shared_shape_ids),
            Self::csv_strings(&self.shard_frontend_neutral_reuse_ids),
            Self::csv_strings(&self.shard_digest_input_sha256s),
            self.warm_cache_enabled,
            self.warm_cache_lookup_attempted,
            self.warm_cache_hits,
            self.warm_cache_misses,
            self.warm_cache_stores,
            self.warm_cache_lookup_ms,
            Self::csv_u64(&self.shard_warm_cache_lookup_ms),
            Self::csv_strings(&self.shard_warm_cache_statuses),
            Self::csv_strings(&self.shard_warm_cache_keys),
            Self::csv_strings(&self.shard_warm_cache_guard_digests),
            self.artifact_materialization_ms,
            Self::csv_u64(&self.shard_artifact_materialization_ms),
            self.estimated_ir_nodes,
            self.fallback_per_action_tasks,
            self.fallback_per_action_compile_ms,
            self.fallback_reason_code(),
            Self::optional_value(self.artifact_identity_source),
            Self::optional_value(self.artifact_identity.as_deref()),
            Self::optional_value(self.artifact_semantic_digest.as_deref()),
            Self::optional_value(self.artifact_link_digest.as_deref()),
            Self::optional_value(self.artifact_cache_digest.as_deref()),
            Self::optional_value(self.artifact_semantic_digest.as_deref()),
            Self::optional_value(self.artifact_link_digest.as_deref()),
            Self::csv_strings(&self.artifact_semantic_digests),
            Self::csv_strings(&self.artifact_link_digests),
            batch_telemetry_descriptor.schema,
            batch_telemetry_descriptor.schema_version,
            batch_telemetry_descriptor.row_kind,
            Self::optional_value(self.batch_compile_preset.as_deref()),
            Self::csv_strings(&self.batch_compile_presets),
            Self::optional_usize(self.host_symbol_map_count),
            Self::csv_usize(&self.shard_host_symbol_map_counts),
            Self::optional_value(self.runtime_setup_temperature_label),
            Self::csv_strings(&self.runtime_setup_temperature_labels),
            Self::optional_value(self.runtime_setup_cache_label),
            Self::csv_strings(&self.runtime_setup_cache_labels),
            tla_trust_cg::TRUST_CG_BATCH_JIT_ARTIFACT_ADMISSION_SCHEMA,
            tla_trust_cg::TRUST_CG_BATCH_JIT_ARTIFACT_ADMISSION_SCHEMA_VERSION,
            Self::optional_value(self.batch_artifact_admission_status.as_deref()),
            Self::csv_strings(&self.batch_artifact_admission_statuses),
            Self::optional_bool(self.batch_artifact_admission_fail_closed),
            Self::csv_bools(&self.batch_artifact_admission_fail_closed_values),
            Self::csv_strings(&self.batch_artifact_admission_missing_fields),
            Self::csv_strings(&self.batch_artifact_admission_rejection_reasons),
            self.artifact_cacheable,
            self.artifact_cache_disabled_by_env,
            self.artifact_identities.len(),
            Self::csv_strings(&self.artifact_identities),
            Self::csv_strings(&self.artifact_cache_digests),
            Self::optional_value(self.prepared_trust_ir_reuse),
            Self::optional_value(self.prepared_trust_ir_reuse_identity.as_deref()),
            fingerprint_admission.surface,
            fingerprint_admission.semantics,
            fingerprint_admission.compatible_frontend_families,
            fingerprint_admission.default_frontend_families,
            fingerprint_admission.blocked_frontend_families,
            Self::optional_value(self.shared_owner),
            Self::optional_value(self.first_beneficiary),
            Self::optional_value(self.second_beneficiary),
            Self::optional_value(self.extraction_status),
            self.compile_telemetry_evidence_rows.len(),
            self.shared_engine_adoption_evidence_rows.len(),
            Self::optional_value(self.compile_telemetry_evidence_row_sha256().as_deref()),
            Self::optional_value(self.shared_engine_adoption_evidence_row_sha256().as_deref()),
        ))
    }

    fn setup_evidence_row_sha256(&self) -> Option<String> {
        self.setup_evidence_row().map(|row| {
            trust_cg_native_admission_sha256("native_action_callout_batch_setup_row", &row)
        })
    }

    fn record_batch_jit_stats(
        &mut self,
        stats: &tla_trust_cg::BatchJitStats,
        source: &'static str,
        compiled_artifact: bool,
    ) {
        let descriptor = TrustCgNativeBatchRuntimeSetupDescriptor::from_batch_jit_stats(stats);
        let artifact_identity = stats.artifact_identity();
        let shared_engine_identity = artifact_identity.shared_engine_identity();
        let compile_telemetry_evidence_row =
            stats.render_compile_telemetry_evidence_row("trust-cg");
        let shared_engine_adoption_evidence_row =
            stats.render_shared_engine_adoption_evidence_row("trust-cg");
        self.artifact_identity_source = Some(source);
        self.artifact_identity = Some(shared_engine_identity.clone());
        self.artifact_semantic_digest = Some(descriptor.semantic_trust_ir_artifact_digest.clone());
        self.artifact_link_digest = Some(descriptor.process_local_link_digest.clone());
        self.artifact_cache_digest = Some(descriptor.artifact_cache_digest.clone());
        self.batch_compile_preset = Some(descriptor.compile_preset.to_string());
        self.host_symbol_map_count = Some(descriptor.host_symbol_map_count);
        self.batch_artifact_admission_status =
            Some(descriptor.artifact_admission_status.to_string());
        self.batch_artifact_admission_fail_closed = Some(descriptor.artifact_admission_fail_closed);
        self.batch_artifact_admission_missing_fields = descriptor
            .artifact_admission_missing_fields
            .iter()
            .map(|field| (*field).to_string())
            .collect();
        self.batch_artifact_admission_rejection_reasons = descriptor
            .artifact_admission_rejection_reasons
            .iter()
            .map(|reason| (*reason).to_string())
            .collect();
        self.artifact_cache_disabled_by_env = trust_cg_artifact_cache_disabled_by_env();
        self.warm_cache_enabled = trust_cg_process_local_warm_artifact_cache_enabled();
        self.artifact_cacheable = compiled_artifact
            && !self.artifact_cache_disabled_by_env
            && !descriptor.artifact_cache_digest.is_empty();
        self.prepared_trust_ir_reuse = Some(artifact_identity.prepared_trust_ir_reuse);
        self.prepared_trust_ir_reuse_identity =
            Some(artifact_identity.prepared_trust_ir_reuse_identity());
        self.shared_owner = Some(artifact_identity.shared_owner);
        self.first_beneficiary = Some(artifact_identity.first_beneficiary);
        self.second_beneficiary = Some(artifact_identity.second_beneficiary);
        self.extraction_status = Some(artifact_identity.extraction_status);
        self.compile_telemetry_evidence_row = Some(compile_telemetry_evidence_row.clone());
        self.shared_engine_adoption_evidence_row =
            Some(shared_engine_adoption_evidence_row.clone());
        if compiled_artifact {
            self.artifact_identities.push(shared_engine_identity);
            self.artifact_semantic_digests
                .push(descriptor.semantic_trust_ir_artifact_digest);
            self.artifact_link_digests
                .push(descriptor.process_local_link_digest);
            self.artifact_cache_digests
                .push(descriptor.artifact_cache_digest);
            self.batch_compile_presets
                .push(descriptor.compile_preset.to_string());
            self.shard_host_symbol_map_counts
                .push(descriptor.host_symbol_map_count);
            self.batch_artifact_admission_statuses
                .push(descriptor.artifact_admission_status.to_string());
            self.batch_artifact_admission_fail_closed_values
                .push(descriptor.artifact_admission_fail_closed);
            self.compile_telemetry_evidence_rows
                .push(compile_telemetry_evidence_row);
            self.shared_engine_adoption_evidence_rows
                .push(shared_engine_adoption_evidence_row);
        }
    }

    fn record_batch_fallback(&mut self, fallback: &TrustCgActionCalloutBatchFallback) {
        self.fallback_reason = fallback.reason;
        self.artifact_cacheable = false;
        match fallback.reason {
            TrustCgActionCalloutBatchFallbackReason::TrustIrBatchAssembly => {
                self.batch_assembly_failed = true;
            }
            TrustCgActionCalloutBatchFallbackReason::BatchCompile => {
                self.batch_compile_failed = true;
            }
            _ => {}
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TrustCgNativeBatchWarmArtifactKey {
    shared_engine_identity: String,
    cache_digest: String,
    export_surface_digest: String,
    native_requirements_digest: String,
    guard_digest: String,
}

impl TrustCgNativeBatchWarmArtifactKey {
    fn from_stats(stats: &tla_trust_cg::BatchJitStats) -> Self {
        let identity = stats.artifact_identity();
        Self::from_artifact_identity(identity)
    }

    fn from_artifact_identity(identity: &tla_trust_cg::BatchJitArtifactIdentity) -> Self {
        let guard_input = format!(
            "cache_digest={};export_surface_digest={};native_requirements_digest={}",
            identity.cache_digest,
            identity.export_surface_digest,
            identity.native_requirements_digest
        );
        Self {
            shared_engine_identity: identity.shared_engine_identity(),
            cache_digest: identity.cache_digest.clone(),
            export_surface_digest: identity.export_surface_digest.clone(),
            native_requirements_digest: identity.native_requirements_digest.clone(),
            guard_digest: trust_cg_native_admission_sha256(
                "native_action_callout_batch_warm_cache_guard",
                &guard_input,
            ),
        }
    }

    fn guard_matches(&self, other: &Self) -> bool {
        self.cache_digest == other.cache_digest
            && self.export_surface_digest == other.export_surface_digest
            && self.native_requirements_digest == other.native_requirements_digest
    }
}

#[derive(Clone)]
struct TrustCgNativeBatchWarmArtifact {
    key: TrustCgNativeBatchWarmArtifactKey,
    library: tla_trust_cg::NativeLibrary,
    stats: tla_trust_cg::BatchJitStats,
}

fn trust_cg_native_batch_warm_artifact_cache(
) -> &'static Mutex<FxHashMap<String, Vec<TrustCgNativeBatchWarmArtifact>>> {
    static CACHE: OnceLock<Mutex<FxHashMap<String, Vec<TrustCgNativeBatchWarmArtifact>>>> =
        OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(FxHashMap::default()))
}

#[cfg(test)]
fn clear_trust_cg_native_batch_warm_artifact_cache_for_tests() {
    if let Ok(mut guard) = trust_cg_native_batch_warm_artifact_cache().lock() {
        guard.clear();
    }
}

#[derive(Clone)]
struct TrustCgNativeBatchWarmArtifactLookup {
    key: TrustCgNativeBatchWarmArtifactKey,
    status: &'static str,
    artifact: Option<TrustCgNativeBatchWarmArtifact>,
}

fn lookup_trust_cg_native_batch_warm_artifact(
    candidate_identity: &tla_trust_cg::BatchJitArtifactIdentity,
) -> TrustCgNativeBatchWarmArtifactLookup {
    let key = TrustCgNativeBatchWarmArtifactKey::from_artifact_identity(candidate_identity);
    if !trust_cg_process_local_warm_artifact_cache_enabled() {
        return TrustCgNativeBatchWarmArtifactLookup {
            key,
            status: "disabled",
            artifact: None,
        };
    }
    let Ok(guard) = trust_cg_native_batch_warm_artifact_cache().lock() else {
        return TrustCgNativeBatchWarmArtifactLookup {
            key,
            status: "lock_error",
            artifact: None,
        };
    };
    let Some(entries) = guard.get(&key.shared_engine_identity) else {
        return TrustCgNativeBatchWarmArtifactLookup {
            key,
            status: "miss",
            artifact: None,
        };
    };
    if let Some(entry) = entries
        .iter()
        .find(|entry| entry.key.guard_matches(&key))
        .cloned()
    {
        return TrustCgNativeBatchWarmArtifactLookup {
            key,
            status: "hit",
            artifact: Some(entry),
        };
    }
    TrustCgNativeBatchWarmArtifactLookup {
        key,
        status: "guard_miss",
        artifact: None,
    }
}

fn store_trust_cg_native_batch_warm_artifact(
    stats: &tla_trust_cg::BatchJitStats,
    library: &tla_trust_cg::NativeLibrary,
) -> bool {
    if !trust_cg_process_local_warm_artifact_cache_enabled() {
        return false;
    }
    let key = TrustCgNativeBatchWarmArtifactKey::from_stats(stats);
    let Ok(mut guard) = trust_cg_native_batch_warm_artifact_cache().lock() else {
        return false;
    };
    let entries = guard
        .entry(key.shared_engine_identity.clone())
        .or_insert_with(Vec::new);
    if let Some(entry) = entries
        .iter_mut()
        .find(|entry| entry.key.guard_matches(&key))
    {
        entry.library = library.clone();
        entry.stats = stats.clone();
    } else {
        entries.push(TrustCgNativeBatchWarmArtifact {
            key,
            library: library.clone(),
            stats: stats.clone(),
        });
    }
    true
}

/// Statistics from trust-codegen cache construction.
#[derive(Debug, Clone, Default)]
pub(in crate::check) struct TrustCgBuildStats {
    /// Number of actions successfully compiled.
    pub actions_compiled: usize,
    /// Number of actions that failed compilation (fell back to interpreter).
    pub actions_failed: usize,
    /// Number of native action callout compile tasks scheduled before executable
    /// split-action coverage rewrites the public action counters.
    pub native_action_callouts_planned: usize,
    /// Number of native action callout compile tasks that produced callable code
    /// before executable split-action coverage rewrites the public counters.
    pub native_action_callouts_compiled: usize,
    /// Number of raw split-action bytecodes skipped because a BindingSpec alias
    /// is the executable dispatch key for the same action instance.
    pub native_action_callouts_skipped_shadowed: usize,
    /// Number of invariants successfully compiled.
    pub invariants_compiled: usize,
    /// Number of invariants that failed compilation.
    pub invariants_failed: usize,
    /// Number of state constraints successfully compiled.
    pub state_constraints_compiled: usize,
    /// Number of state constraints that failed compilation.
    pub state_constraints_failed: usize,
    /// Wall-clock time spent planning native action callout compile tasks.
    pub native_action_callout_planning_ms: u64,
    /// Wall-clock time spent compiling planned native action callouts.
    pub native_action_callout_compile_ms: u64,
    /// Detailed native action callout batch setup telemetry.
    pub native_action_callout_batch: TrustCgNativeActionCalloutBatchStats,
    /// Wall-clock time spent compiling native invariant callouts.
    pub native_invariant_callout_compile_ms: u64,
    /// Wall-clock time spent compiling native state-constraint callouts.
    pub native_state_constraint_callout_compile_ms: u64,
    /// Total wall-clock time for all compilations.
    pub total_compile_ms: u64,
    /// First action compile failure, kept for strict native-fused diagnostics.
    pub first_action_failure: Option<String>,
    /// Per-action native-admission failures (action_name, reason), recorded
    /// alongside `first_action_failure` for the observability-only
    /// `TY_TRUST_CG_DUMP_NATIVE_ADMISSION_FAILURES` dump. Bounded by the action
    /// count (only failing actions push); never consulted for dispatch.
    pub per_action_failures: Vec<(String, String)>,
    /// Number of actions recognized as the runtime-domain multi-successor
    /// ("NextStateLoop") shape that the single-successor native ABI cannot
    /// express. These are *counted* (included in `actions_failed`) and routed
    /// to the interpreter today; the count makes the future
    /// [`tla_jit_abi::NextStateLoopFn`] target observable without changing any
    /// dispatch behavior.
    pub next_state_loop_recognized_unsupported: usize,
    /// Native `ProofAnnotation::BoundedLoop` facts seen in trust-ir modules lowered
    /// on the trust-codegen cache-build hot path.
    pub trust_ir_bounded_loop_headers: usize,
    /// Largest native `ProofAnnotation::BoundedLoop(N)` bound seen on that path.
    pub trust_ir_max_bounded_loop_bound: Option<u64>,
    /// Native `ProofAnnotation::ParallelMap` facts seen in trust-ir modules lowered
    /// on the trust-codegen cache-build hot path.
    pub trust_ir_parallel_map_headers: usize,
    /// Shared trust-codegen install-gate admission summary for this runtime build.
    pub native_admission_summary: Option<tla_trust_cg::NativeInstallGateAdmissionSummary>,
    /// Stable key/value evidence row rendered from `native_admission_summary`.
    pub native_admission_evidence_row: Option<String>,
    /// Structured sidecar wrapping the stable evidence row for JSONL/reporting sinks.
    pub native_admission_evidence_report: Option<TrustCgNativeAdmissionEvidenceReport>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::check) struct TrustCgNativeAdmissionEvidenceReport {
    evidence_row: String,
    sidecar_evidence_rows: Vec<String>,
    fields: Vec<(String, String)>,
}

impl TrustCgNativeAdmissionEvidenceReport {
    fn new(
        prefix: &str,
        fields: Vec<(String, String)>,
        sidecar_evidence_rows: Vec<String>,
    ) -> Self {
        let mut evidence_row = prefix.to_string();
        for (key, value) in &fields {
            evidence_row.push(' ');
            evidence_row.push_str(key);
            evidence_row.push('=');
            evidence_row.push_str(value);
        }
        Self {
            evidence_row,
            sidecar_evidence_rows,
            fields,
        }
    }

    pub(in crate::check) fn evidence_row(&self) -> &str {
        &self.evidence_row
    }

    pub(in crate::check) fn evidence_rows(&self) -> Vec<&str> {
        std::iter::once(self.evidence_row.as_str())
            .chain(self.sidecar_evidence_rows.iter().map(String::as_str))
            .collect()
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(in crate::check) fn field(&self, key: &str) -> Option<&str> {
        self.fields
            .iter()
            .find_map(|(field_key, value)| (field_key == key).then_some(value.as_str()))
    }

    pub(in crate::check) fn to_json_value(&self) -> serde_json::Value {
        let mut fields = serde_json::Map::new();
        for (key, value) in &self.fields {
            fields.insert(key.clone(), serde_json::Value::String(value.clone()));
        }

        let evidence_rows = self
            .evidence_rows()
            .into_iter()
            .map(|row| serde_json::Value::String(row.to_string()))
            .collect::<Vec<_>>();

        serde_json::json!({
            "schema": TRUST_CG_NATIVE_ADMISSION_EVIDENCE_REPORT_SCHEMA,
            "schema_version": TRUST_CG_NATIVE_ADMISSION_EVIDENCE_REPORT_SCHEMA_VERSION,
            "backend": "trust-cg",
            "kind": TRUST_CG_NATIVE_ADMISSION_KIND,
            "evidence": evidence_rows,
            "fields": serde_json::Value::Object(fields),
        })
    }
}

struct TrustCgActionCompileTask {
    action_name: String,
    func: tla_tir::bytecode::BytecodeFunction,
    state_layout: Option<Arc<tla_jit_abi::StateLayout>>,
    opt_level: tla_trust_cg::OptLevel,
    const_pool: Option<Arc<tla_tir::bytecode::ConstantPool>>,
    chunk: Option<Arc<tla_tir::bytecode::BytecodeChunk>>,
    /// Chunk-wide callee return shapes inferred ONCE per source chunk and
    /// shared by every task planned against that chunk (including binding
    /// specializations, whose chunks differ only by appended constant-pool
    /// entries — see the reuse contract on `ChunkCalleeReturnShapes`).
    /// `None` falls back to per-task inference inside `tla_ir::lower`.
    chunk_callee_shapes: Option<tla_ir::lower::ChunkCalleeReturnShapes>,
    action_local_set_domain_proof: Option<tla_ir::lower::ActionLocalSetDomainProof>,
    binding_values: Vec<i64>,
    formal_values: Vec<i64>,
    read_vars: Vec<u16>,
    write_vars: Vec<u16>,
    /// Hybrid-placeholder (compound) vars this action's lowering will service
    /// through the allocation-lean compound-read callout (item 4 M1). Derived
    /// from the SAME `tla_ir::lower::compound_read_callout_vars` analysis the
    /// lowering itself runs, so the declaration cannot claim a var the emitted
    /// code does not actually read through the published parent context.
    compound_read_vars: Vec<u16>,
    /// When true this action lowers through the multi-successor record-set
    /// kernel (`lower_next_state_loop_scaffold`) and its compiled symbol is a
    /// [`tla_jit_abi::NextStateLoopFn`] dispatched via the sink call
    /// convention — NOT the single-successor `NativeNextStateFn`. Default
    /// `false` (byte-identical single-successor path).
    next_state_loop: bool,
}

struct TrustCgLoweredActionCompileTask {
    action_name: String,
    opt_level: tla_trust_cg::OptLevel,
    trust_ir_module: trust_ir::Module,
    symbol_name: String,
    binding_values: Vec<i64>,
    formal_values: Vec<i64>,
    read_vars: Vec<u16>,
    write_vars: Vec<u16>,
    compound_read_vars: Vec<u16>,
    trust_ir_proof_facts: tla_ir::annotations::NativeProofAnnotationSummary,
    /// Carries [`TrustCgActionCompileTask::next_state_loop`] through lowering so
    /// the compiled symbol is routed to the `NextStateLoopFn` map.
    next_state_loop: bool,
}

// The `Compiled` payload (native library handle + per-action metadata) is
// intentionally large and the dominant variant for compiled actions; boxing it
// would add an allocation/indirection on the hot success path purely to shrink
// the rare `Failed` case, so the size disparity is accepted.
#[allow(clippy::large_enum_variant)]
enum TrustCgActionCompileOutcome {
    Compiled {
        action_name: String,
        fn_ptr: NativeNextStateFn,
        library: tla_trust_cg::NativeLibrary,
        symbol_name: String,
        binding_values: Vec<i64>,
        formal_values: Vec<i64>,
        read_vars: Vec<u16>,
        write_vars: Vec<u16>,
        compound_read_vars: Vec<u16>,
        trust_ir_proof_facts: tla_ir::annotations::NativeProofAnnotationSummary,
        batch_shard: Option<TrustCgNativeActionBatchShardMetadata>,
        /// True when `fn_ptr` is really a [`tla_jit_abi::NextStateLoopFn`]
        /// (multi-successor record-set kernel) and must be routed to
        /// `next_state_loop_fns`, never the single-successor `next_state_fns`.
        next_state_loop: bool,
    },
    Failed {
        action_name: String,
        message: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TrustCgInnerExistsExpansionProof {
    expansion_count: usize,
    kind: TrustCgInnerExistsExpansionProofKind,
}

impl TrustCgInnerExistsExpansionProof {
    fn native_fused_safe(&self, expansion_count: usize) -> bool {
        if self.expansion_count != expansion_count {
            return false;
        }
        match &self.kind {
            TrustCgInnerExistsExpansionProofKind::StaticFiniteDomain { binding_values } => {
                if binding_values.len() != expansion_count {
                    return false;
                }
                let mut seen = FxHashSet::default();
                binding_values.iter().all(|values| seen.insert(values))
            }
            TrustCgInnerExistsExpansionProofKind::RuntimeGuardedFiniteDomain { binding_values } => {
                if binding_values.len() != expansion_count {
                    return false;
                }
                let mut seen = FxHashSet::default();
                binding_values.iter().all(|value| seen.insert(value))
            }
            TrustCgInnerExistsExpansionProofKind::ActionLocalTaggedScalarOrSet {
                scalar_kind,
                universe_values,
                ..
            } => {
                universe_values.len() == expansion_count
                    && runtime_scalar_elements_match_kind(universe_values, *scalar_kind)
            }
            // Top-level disjunction split (Wall 3). The base action
            // `guard /\ (D1 \/ ... \/ Dn)` was split into per-disjunct
            // sub-actions, each independently inner-EXISTS-expanded into native
            // functions that ARE native-fused-safe (verified at build time
            // before this proof is emitted). The union of their final native
            // keys is the base action's expansion key set. Distinctness of
            // successors holds because (a) within each sub-action the per-witness
            // binding values were already proven distinct, and (b) across
            // sub-actions the native functions are structurally distinct (they
            // execute different per-disjunct StoreVar sets), so the BFS
            // fingerprint set folds any incidental successor overlap. The only
            // remaining check is that the recorded total matches the key count.
            TrustCgInnerExistsExpansionProofKind::SplitDisjunction { total_keys } => {
                *total_keys == expansion_count
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum TrustCgInnerExistsExpansionProofKind {
    StaticFiniteDomain {
        binding_values: Vec<Vec<i64>>,
    },
    RuntimeGuardedFiniteDomain {
        binding_values: Vec<RuntimeInnerExistsBindingValue>,
    },
    ActionLocalTaggedScalarOrSet {
        source_var_idx: u16,
        key_reg: u8,
        domain_reg: u8,
        key_values: Vec<tla_jit_abi::SetBitmaskElement>,
        scalar_kind: tla_jit_abi::ScalarSlotKind,
        proof_source: tla_core::NameId,
        universe_values: Vec<tla_jit_abi::SetBitmaskElement>,
    },
    /// A top-level action-disjunction split (Wall 3): the base action's
    /// expansion keys are the union of its per-disjunct sub-actions' native
    /// keys, each of which is independently native-fused-safe.
    SplitDisjunction {
        total_keys: usize,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RuntimeTaggedScalarOrSetTypeProof {
    scalar_kind: tla_jit_abi::ScalarSlotKind,
    proof_source: tla_core::NameId,
    universe_values: Vec<tla_jit_abi::SetBitmaskElement>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RuntimeTaggedScalarOrSetReadProof {
    source_var_idx: u16,
    key_reg: u8,
    domain_reg: u8,
    key_values: Vec<tla_jit_abi::SetBitmaskElement>,
    proof: RuntimeTaggedScalarOrSetTypeProof,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum RuntimeInnerExistsRegisterShape {
    Scalar {
        value: Option<tla_jit_abi::SetBitmaskElement>,
    },
    /// An integer scalar that is *provably* in `0..=upper`.
    ///
    /// Produced by `Len` applied to a proven-capacity sequence: the runtime
    /// length is unknown but bounded above by the proven sequence capacity.
    /// Used to expand `\E i \in lo..Len(seq)` into candidates `lo..=upper`,
    /// each filtered by a runtime `i \in lo..Len(seq)` membership guard.
    ScalarIntUpperBound { upper: i64 },
    Function {
        source_var_idx: Option<u16>,
        key_values: Option<Vec<tla_jit_abi::SetBitmaskElement>>,
        value: Option<Box<RuntimeInnerExistsRegisterShape>>,
        /// `Some(C)` when this function is a sequence whose capacity `C` is a
        /// *proven* upper bound on `Len`. Carries the proof from the layout to
        /// the `Len` builtin so capacity-driven domain enumeration stays sound.
        sequence_capacity_proof: Option<usize>,
    },
    SetBitmask {
        values: Vec<tla_jit_abi::SetBitmaskElement>,
    },
    Powerset {
        base_values: Vec<tla_jit_abi::SetBitmaskElement>,
    },
    KSubset {
        base_values: Vec<tla_jit_abi::SetBitmaskElement>,
        k: usize,
    },
    TaggedScalarOrSet {
        proof: RuntimeTaggedScalarOrSetTypeProof,
    },
    TaggedScalarOrSetRead {
        read: RuntimeTaggedScalarOrSetReadProof,
    },
    // A `func[key]` read of a tagged-scalar-or-set value whose key is NOT
    // statically proven to lie in the function domain (e.g. a key/domain type
    // collision such as a bool key against an int domain). The element universe
    // is still known, so the inner-exists can be expanded via the runtime-guarded
    // interpreter path, but no native-fused proof may be emitted: a mistyped key
    // read cannot be soundly fused, so it must fail closed.
    TaggedScalarOrSetUnprovenRead {
        proof: RuntimeTaggedScalarOrSetTypeProof,
    },
}

#[derive(Clone, Debug)]
struct RuntimeGuardedInnerExistsExpansion {
    action: tla_tir::bytecode::ExpandedAction,
    const_pool: Option<tla_tir::bytecode::ConstantPool>,
    action_local_set_domain_proof: Option<tla_ir::lower::ActionLocalSetDomainProof>,
    native_fused_proof: Option<TrustCgInnerExistsExpansionProofKind>,
    inner_binding_literals: Option<Vec<tla_value::Value>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RuntimeInnerExistsDomainProof {
    values: Vec<RuntimeInnerExistsBindingValue>,
    action_local_set_domain_proof: Option<tla_ir::lower::ActionLocalSetDomainProof>,
    native_fused_proof: Option<TrustCgInnerExistsExpansionProofKind>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum RuntimeInnerExistsBindingValue {
    Scalar(tla_jit_abi::SetBitmaskElement),
    FiniteSet(Vec<tla_jit_abi::SetBitmaskElement>),
}

fn runtime_scalar_element_raw(element: &tla_jit_abi::SetBitmaskElement) -> i64 {
    match *element {
        tla_jit_abi::SetBitmaskElement::Int(value) => value,
        tla_jit_abi::SetBitmaskElement::Bool(value) => i64::from(value),
        tla_jit_abi::SetBitmaskElement::String(name)
        | tla_jit_abi::SetBitmaskElement::ModelValue(name) => i64::from(name.0),
    }
}

fn runtime_scalar_element_kind(
    element: &tla_jit_abi::SetBitmaskElement,
) -> tla_jit_abi::ScalarSlotKind {
    match *element {
        tla_jit_abi::SetBitmaskElement::Int(_) => tla_jit_abi::ScalarSlotKind::Int,
        tla_jit_abi::SetBitmaskElement::Bool(_) => tla_jit_abi::ScalarSlotKind::Bool,
        tla_jit_abi::SetBitmaskElement::String(_) => tla_jit_abi::ScalarSlotKind::String,
        tla_jit_abi::SetBitmaskElement::ModelValue(_) => tla_jit_abi::ScalarSlotKind::ModelValue,
    }
}

fn runtime_scalar_element_from_const_value(
    value: &tla_value::Value,
) -> Option<tla_jit_abi::SetBitmaskElement> {
    match value {
        tla_value::Value::SmallInt(value) => Some(tla_jit_abi::SetBitmaskElement::Int(*value)),
        tla_value::Value::Int(value) => {
            use num_traits::ToPrimitive;
            value.to_i64().map(tla_jit_abi::SetBitmaskElement::Int)
        }
        tla_value::Value::Bool(value) => Some(tla_jit_abi::SetBitmaskElement::Bool(*value)),
        tla_value::Value::String(value) => Some(tla_jit_abi::SetBitmaskElement::String(
            tla_core::intern_name(value.as_ref()),
        )),
        tla_value::Value::ModelValue(value) => Some(tla_jit_abi::SetBitmaskElement::ModelValue(
            tla_core::intern_name(value.as_ref()),
        )),
        _ => None,
    }
}

fn runtime_binding_value_raw(value: &RuntimeInnerExistsBindingValue) -> Option<i64> {
    match value {
        RuntimeInnerExistsBindingValue::Scalar(value) => Some(runtime_scalar_element_raw(value)),
        RuntimeInnerExistsBindingValue::FiniteSet(_) => None,
    }
}

fn runtime_binding_value_literal(value: &RuntimeInnerExistsBindingValue) -> tla_value::Value {
    match value {
        RuntimeInnerExistsBindingValue::Scalar(value) => {
            tla_jit_abi::set_bitmask_element_to_value(*value)
        }
        RuntimeInnerExistsBindingValue::FiniteSet(elements) => {
            tla_jit_abi::set_bitmask_elements_to_value(elements)
        }
    }
}

fn runtime_scalar_binding_values(
    values: &[tla_jit_abi::SetBitmaskElement],
) -> Vec<RuntimeInnerExistsBindingValue> {
    values
        .iter()
        .copied()
        .map(RuntimeInnerExistsBindingValue::Scalar)
        .collect()
}

fn runtime_scalar_values_from_const_finite_set(
    value: &tla_value::Value,
) -> Option<Vec<tla_jit_abi::SetBitmaskElement>> {
    let values = match value {
        tla_value::Value::Set(set) => set
            .iter()
            .map(runtime_scalar_element_from_const_value)
            .collect::<Option<Vec<_>>>()?,
        tla_value::Value::Interval(interval) => {
            use num_traits::ToPrimitive;
            let lo = interval.low().to_i64()?;
            let hi = interval.high().to_i64()?;
            if hi < lo {
                return Some(Vec::new());
            }
            let len = usize::try_from(hi.checked_sub(lo)?.checked_add(1)?).ok()?;
            if len > tla_tir::bytecode::MAX_INNER_DOMAIN_SIZE {
                return None;
            }
            (lo..=hi).map(tla_jit_abi::SetBitmaskElement::Int).collect()
        }
        _ => return None,
    };
    runtime_typed_scalar_values_from_expansion_domain(&values)
}

fn runtime_powerset_binding_values(
    base_values: &[tla_jit_abi::SetBitmaskElement],
) -> Option<Vec<RuntimeInnerExistsBindingValue>> {
    let subset_count = checked_runtime_powerset_len(base_values.len())?;
    if subset_count > tla_tir::bytecode::MAX_INNER_DOMAIN_SIZE {
        return None;
    }

    let mut values = Vec::with_capacity(subset_count);
    values.push(RuntimeInnerExistsBindingValue::FiniteSet(Vec::new()));
    for k in 1..=base_values.len() {
        let mut indices: Vec<usize> = (0..k).collect();
        loop {
            values.push(RuntimeInnerExistsBindingValue::FiniteSet(
                indices.iter().map(|&idx| base_values[idx]).collect(),
            ));

            let mut i = k;
            while i > 0 {
                i -= 1;
                if indices[i] < base_values.len() - k + i {
                    break;
                }
            }
            if i == 0 && indices[0] >= base_values.len() - k {
                break;
            }
            indices[i] += 1;
            for j in (i + 1)..k {
                indices[j] = indices[j - 1] + 1;
            }
        }
    }
    Some(values)
}

fn runtime_ksubset_binding_values(
    base_values: &[tla_jit_abi::SetBitmaskElement],
    k: usize,
) -> Option<Vec<RuntimeInnerExistsBindingValue>> {
    let subset_count = checked_runtime_combination_len_capped(
        base_values.len(),
        k,
        tla_tir::bytecode::MAX_INNER_DOMAIN_SIZE,
    )?;
    if subset_count > tla_tir::bytecode::MAX_INNER_DOMAIN_SIZE {
        return None;
    }

    let mut values = Vec::with_capacity(subset_count);
    push_runtime_k_subset_binding_values(base_values, k, &mut values);
    Some(values)
}

fn push_runtime_k_subset_binding_values(
    base_values: &[tla_jit_abi::SetBitmaskElement],
    k: usize,
    values: &mut Vec<RuntimeInnerExistsBindingValue>,
) {
    if k > base_values.len() {
        return;
    }
    if k == 0 {
        values.push(RuntimeInnerExistsBindingValue::FiniteSet(Vec::new()));
        return;
    }

    let mut indices: Vec<usize> = (0..k).collect();
    loop {
        values.push(RuntimeInnerExistsBindingValue::FiniteSet(
            indices.iter().map(|&idx| base_values[idx]).collect(),
        ));

        let mut i = k;
        while i > 0 {
            i -= 1;
            if indices[i] < base_values.len() - k + i {
                break;
            }
        }
        if i == 0 && indices[0] >= base_values.len() - k {
            break;
        }
        indices[i] += 1;
        for j in (i + 1)..k {
            indices[j] = indices[j - 1] + 1;
        }
    }
}

fn checked_runtime_powerset_len(base_len: usize) -> Option<usize> {
    if base_len >= usize::BITS as usize {
        return None;
    }
    1usize.checked_shl(u32::try_from(base_len).ok()?)
}

fn checked_runtime_combination_len_capped(n: usize, k: usize, cap: usize) -> Option<usize> {
    if k > n {
        return Some(0);
    }
    let k = k.min(n - k);
    let mut result = 1u128;
    for i in 1..=k {
        result = result.checked_mul((n - k + i) as u128)? / (i as u128);
        if result > cap as u128 {
            return cap.checked_add(1);
        }
    }
    usize::try_from(result).ok()
}

fn runtime_typed_scalar_values_from_bitmask_universe(
    universe: &[tla_jit_abi::SetBitmaskElement],
) -> Option<Vec<tla_jit_abi::SetBitmaskElement>> {
    runtime_typed_scalar_values_from_bounded_domain(universe, 63)
}

fn runtime_typed_scalar_values_from_expansion_domain(
    universe: &[tla_jit_abi::SetBitmaskElement],
) -> Option<Vec<tla_jit_abi::SetBitmaskElement>> {
    runtime_typed_scalar_values_from_bounded_domain(
        universe,
        tla_tir::bytecode::MAX_INNER_DOMAIN_SIZE,
    )
}

fn runtime_typed_scalar_values_from_bounded_domain(
    universe: &[tla_jit_abi::SetBitmaskElement],
    max_len: usize,
) -> Option<Vec<tla_jit_abi::SetBitmaskElement>> {
    if universe.len() > max_len {
        return None;
    }
    let mut kind = None;
    let mut seen = FxHashSet::default();
    let mut values = Vec::with_capacity(universe.len());
    for element in universe {
        let current_kind = runtime_scalar_element_kind(element);
        if kind
            .replace(current_kind)
            .is_some_and(|existing| existing != current_kind)
        {
            return None;
        }
        if !seen.insert(*element) {
            return None;
        }
        values.push(*element);
    }
    Some(values)
}

fn runtime_scalar_elements_match_kind(
    values: &[tla_jit_abi::SetBitmaskElement],
    expected: tla_jit_abi::ScalarSlotKind,
) -> bool {
    values
        .iter()
        .all(|element| runtime_scalar_element_kind(element) == expected)
}

impl TrustCgActionCompileOutcome {
    fn action_name(&self) -> &str {
        match self {
            Self::Compiled { action_name, .. } | Self::Failed { action_name, .. } => action_name,
        }
    }

    fn is_compiled(&self) -> bool {
        matches!(self, Self::Compiled { .. })
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(in crate::check) struct TrustCgActionDispatchStats {
    pub enabled: usize,
    pub disabled: usize,
    pub runtime_errors: usize,
}

impl TrustCgBuildStats {
    fn record_first_action_failure(&mut self, action_name: &str, message: impl AsRef<str>) {
        let message = message.as_ref();
        if self.first_action_failure.is_none() {
            self.first_action_failure = Some(format!("{action_name}: {message}"));
        }
        // Observability-only: retain every per-action rejection reason for the
        // `TY_TRUST_CG_DUMP_NATIVE_ADMISSION_FAILURES` dump. Pushed on the same
        // (bounded) path that records the first failure; never read for dispatch.
        self.per_action_failures
            .push((action_name.to_string(), message.to_string()));
    }

    /// Dump, per action that failed native admission, the precise rejection
    /// reason, plus the batch artifact-admission rejection reasons and a
    /// native-eligibility pre-screen line. Observability-only (gated by
    /// `TY_TRUST_CG_DUMP_NATIVE_ADMISSION_FAILURES`); emits nothing and reads
    /// nothing that influences dispatch.
    fn dump_native_admission_failures(&self) {
        if !trust_cg_dump_native_admission_failures_enabled() {
            return;
        }
        let total = self.actions_total();
        let compiled = self.actions_compiled;
        // Pre-screen line (optional plan item #3): native-eligible actions / total.
        eprintln!(
            "[trust_cg-admission] native_eligible_actions={compiled}/{total} \
             (actions_failed={}, next_state_loop_recognized_unsupported={})",
            self.actions_failed, self.next_state_loop_recognized_unsupported,
        );
        if self.per_action_failures.is_empty() {
            eprintln!(
                "[trust_cg-admission] no per-action native-admission failures recorded for this build"
            );
        } else {
            for (action_name, reason) in &self.per_action_failures {
                eprintln!(
                    "[trust_cg-admission] action='{action_name}' native_admission=declined reason: {reason}"
                );
            }
        }
        // Surface the batch artifact-admission rejection reasons (the install-gate
        // disposition), which are recorded independently of per-action lowering.
        let batch = &self.native_action_callout_batch;
        if !batch.batch_artifact_admission_rejection_reasons.is_empty() {
            eprintln!(
                "[trust_cg-admission] batch_artifact_admission_rejection_reasons=[{}]",
                batch.batch_artifact_admission_rejection_reasons.join("; "),
            );
        }
        if let Some(summary) = &self.native_admission_summary {
            eprintln!(
                "[trust_cg-admission] install_gate disposition={} install_authority={} reason_code={}",
                summary.disposition,
                summary.install_authority,
                summary.reason_code.unwrap_or("none"),
            );
        }
    }

    fn record_trust_ir_proof_facts(
        &mut self,
        facts: tla_ir::annotations::NativeProofAnnotationSummary,
    ) {
        self.trust_ir_bounded_loop_headers += facts.bounded_loop_headers;
        self.trust_ir_parallel_map_headers += facts.parallel_map_headers;
        self.trust_ir_max_bounded_loop_bound = match (
            self.trust_ir_max_bounded_loop_bound,
            facts.max_bounded_loop_bound,
        ) {
            (Some(left), Some(right)) => Some(left.max(right)),
            (Some(bound), None) | (None, Some(bound)) => Some(bound),
            (None, None) => None,
        };
    }

    /// Total action instances represented by the telemetry counters.
    #[inline]
    pub(in crate::check) fn actions_total(&self) -> usize {
        self.actions_compiled + self.actions_failed
    }

    /// Total invariant slots represented by the telemetry counters.
    #[inline]
    pub(in crate::check) fn invariants_total(&self) -> usize {
        self.invariants_compiled + self.invariants_failed
    }

    /// Total state-constraint slots represented by the telemetry counters.
    #[inline]
    pub(in crate::check) fn state_constraints_total(&self) -> usize {
        self.state_constraints_compiled + self.state_constraints_failed
    }

    /// True when trust-codegen saw action work but produced no executable native action.
    ///
    /// Predicate-only cache builds used by focused invariant canaries have zero
    /// action work and are not considered zero-coverage BFS builds.
    #[inline]
    pub(in crate::check) fn zero_action_coverage(&self) -> bool {
        self.actions_total() > 0 && self.actions_compiled == 0
    }

    /// Replace raw action-attempt counters with executable split-action coverage.
    ///
    /// The trust-codegen builder may see arity-positive bytecode wrappers that are not
    /// BFS dispatch targets. Benchmark coverage gates need the action instances
    /// that compiled BFS would actually invoke, so `run_helpers` rewrites the
    /// action counters from split-action metadata before telemetry is emitted.
    #[inline]
    pub(in crate::check) fn record_executable_action_coverage(
        &mut self,
        compiled: usize,
        total: usize,
    ) {
        self.actions_compiled = compiled;
        self.actions_failed = total.saturating_sub(compiled);
    }

    /// Attach the shared trust-codegen install-gate admission summary for this runtime
    /// build and return the rendered evidence row.
    pub(in crate::check) fn record_native_admission_evidence(
        &mut self,
        state_var_count: usize,
    ) -> String {
        let summary = trust_cg_native_admission_summary_for_runtime(self, state_var_count);
        let report = render_trust_cg_native_admission_evidence_report(&summary, self);
        let row = report.evidence_row().to_string();
        self.native_admission_summary = Some(summary);
        self.native_admission_evidence_row = Some(row.clone());
        self.native_admission_evidence_report = Some(report);
        row
    }

    /// Emit the native action batch setup evidence rows without requiring
    /// `TY_TRUST_CG_SETUP_TIMING`.
    ///
    /// Cold-start reports consume these same key/value evidence rows to
    /// attribute setup, compile, cache lookup, and artifact materialization
    /// slices on ordinary runs. The broader human timing summary remains gated
    /// behind the explicit timing env var.
    pub(in crate::check) fn emit_native_action_callout_batch_setup_evidence_rows(&self) {
        let batch = &self.native_action_callout_batch;
        if let Some(row) = batch.setup_evidence_row() {
            eprintln!("[trust_cg-evidence] {row}");
        }
        if batch.compile_telemetry_evidence_rows.is_empty() {
            if let Some(row) = &batch.compile_telemetry_evidence_row {
                eprintln!("[trust_cg-evidence] {row}");
            }
        } else {
            for row in &batch.compile_telemetry_evidence_rows {
                eprintln!("[trust_cg-evidence] {row}");
            }
        }
        if batch.shared_engine_adoption_evidence_rows.is_empty() {
            if let Some(row) = &batch.shared_engine_adoption_evidence_row {
                eprintln!("[trust_cg-evidence] {row}");
            }
        } else {
            for row in &batch.shared_engine_adoption_evidence_rows {
                eprintln!("[trust_cg-evidence] {row}");
            }
        }
    }

    fn maybe_log_native_cache_build_profile(&self) {
        if !trust_cg_setup_timing_enabled() {
            return;
        }
        eprintln!(
            "[trust_cg-timing] native_cache_build_ms={} action_planning_ms={} action_compile_ms={} invariant_compile_ms={} state_constraint_compile_ms={} action_callouts_planned={} action_callouts_compiled={} actions_compiled={} actions_failed={} invariants_compiled={} invariants_failed={} state_constraints_compiled={} state_constraints_failed={}",
            self.total_compile_ms,
            self.native_action_callout_planning_ms,
            self.native_action_callout_compile_ms,
            self.native_invariant_callout_compile_ms,
            self.native_state_constraint_callout_compile_ms,
            self.native_action_callouts_planned,
            self.native_action_callouts_compiled,
            self.actions_compiled,
            self.actions_failed,
            self.invariants_compiled,
            self.invariants_failed,
            self.state_constraints_compiled,
            self.state_constraints_failed,
        );
        let batch = &self.native_action_callout_batch;
        eprintln!(
            "[trust_cg-timing] native_action_callout_batch_summary attempted={} action_count={} input_tasks={} lowered_tasks={} lowering_failed={} setup_ms={} lowering_ms={} assembly_attempted={} assembly_ms={} assembly_failed={} compile_attempted={} compile_ms={} compile_failed={} batch_compiled={} sharding_policy_selected={} shard_count={} shard_action_counts={} shard_assembly_ms={} shard_compile_ms={} shard_stable_ids={} shard_frontend_neutral_reuse_ids={} shard_digest_input_sha256s={} warm_cache_enabled={} warm_cache_lookup_attempted={} warm_cache_hits={} warm_cache_misses={} warm_cache_stores={} warm_cache_lookup_ms={} shard_warm_cache_lookup_ms={} artifact_materialization_ms={} shard_artifact_materialization_ms={} shard_warm_cache_statuses={} shard_warm_cache_keys={} shard_warm_cache_guard_digests={} fallback_per_action_tasks={} fallback_per_action_compile_ms={} fallback_reason={} artifact_identity={} artifact_cacheable={} artifact_cache_digest={} artifact_count={} prepared_trust_ir_reuse={}",
            batch.attempted,
            batch.action_count,
            batch.input_tasks,
            batch.lowered_tasks,
            batch.lowering_failed,
            batch.setup_ms,
            batch.lowering_ms,
            batch.batch_assembly_attempted,
            batch.batch_assembly_ms,
            batch.batch_assembly_failed,
            batch.batch_compile_attempted,
            batch.batch_compile_ms,
            batch.batch_compile_failed,
            batch.batch_compiled,
            batch.sharding_policy_selected,
            batch.shard_count,
            TrustCgNativeActionCalloutBatchStats::csv_usize(&batch.shard_action_counts),
            TrustCgNativeActionCalloutBatchStats::csv_u64(&batch.shard_assembly_ms),
            TrustCgNativeActionCalloutBatchStats::csv_u64(&batch.shard_compile_ms),
            TrustCgNativeActionCalloutBatchStats::csv_strings(&batch.shard_stable_ids),
            TrustCgNativeActionCalloutBatchStats::csv_strings(
                &batch.shard_frontend_neutral_reuse_ids,
            ),
            TrustCgNativeActionCalloutBatchStats::csv_strings(&batch.shard_digest_input_sha256s),
            batch.warm_cache_enabled,
            batch.warm_cache_lookup_attempted,
            batch.warm_cache_hits,
            batch.warm_cache_misses,
            batch.warm_cache_stores,
            batch.warm_cache_lookup_ms,
            TrustCgNativeActionCalloutBatchStats::csv_u64(&batch.shard_warm_cache_lookup_ms),
            batch.artifact_materialization_ms,
            TrustCgNativeActionCalloutBatchStats::csv_u64(
                &batch.shard_artifact_materialization_ms,
            ),
            TrustCgNativeActionCalloutBatchStats::csv_strings(&batch.shard_warm_cache_statuses),
            TrustCgNativeActionCalloutBatchStats::csv_strings(&batch.shard_warm_cache_keys),
            TrustCgNativeActionCalloutBatchStats::csv_strings(
                &batch.shard_warm_cache_guard_digests,
            ),
            batch.fallback_per_action_tasks,
            batch.fallback_per_action_compile_ms,
            batch.fallback_reason_code(),
            TrustCgNativeActionCalloutBatchStats::optional_value(batch.artifact_identity.as_deref()),
            batch.artifact_cacheable,
            TrustCgNativeActionCalloutBatchStats::optional_value(
                batch.artifact_cache_digest.as_deref(),
            ),
            batch.artifact_identities.len(),
            TrustCgNativeActionCalloutBatchStats::optional_value(batch.prepared_trust_ir_reuse),
        );
    }
}

fn trust_cg_native_admission_summary_for_runtime(
    stats: &TrustCgBuildStats,
    state_var_count: usize,
) -> tla_trust_cg::NativeInstallGateAdmissionSummary {
    let facts = trust_cg_native_admission_runtime_facts(stats, state_var_count);
    let manifest_checksum = trust_cg_native_admission_checksum("manifest_absent", &facts);
    let target_checksum = trust_cg_native_admission_checksum("target", &facts);
    let abi_checksum = trust_cg_native_admission_checksum(
        "abi",
        "ty.trust_cg.native_callout_abi.v1:next_state,invariant,state_constraint",
    );
    let layout_checksum = trust_cg_native_admission_checksum(
        "layout",
        &format!("state_var_count={state_var_count};runtime=ty_bfs"),
    );
    let proof_policy_checksum =
        trust_cg_native_admission_checksum("proof_policy", "ty.native_activation.pending_proof");
    let invalidation_checksum =
        trust_cg_native_admission_checksum("invalidation", "ty.trust_cg.native_runtime.v1");
    let artifact_id = format!(
        "ty-trust-cg-native-runtime-a{}-of{}-i{}-of{}-sc{}-of{}-sv{}",
        stats.actions_compiled,
        stats.actions_total(),
        stats.invariants_compiled,
        stats.invariants_total(),
        stats.state_constraints_compiled,
        stats.state_constraints_total(),
        state_var_count,
    );
    let payload_identity = tla_trust_cg::NativeInstallGatePayloadIdentity {
        source_sha256: trust_cg_native_admission_sha256("source", &facts),
        trust_ir_sha256: trust_cg_native_admission_sha256(
            "trust_ir",
            &format!(
                "{facts};bounded_loop_headers={};max_bounded_loop_bound={:?};parallel_map_headers={}",
                stats.trust_ir_bounded_loop_headers,
                stats.trust_ir_max_bounded_loop_bound,
                stats.trust_ir_parallel_map_headers,
            ),
        ),
        native_payload_sha256: trust_cg_native_admission_sha256("native_payload", &facts),
    };
    let input = tla_trust_cg::NativeInstallGateInput {
        consumer: "ty".to_string(),
        consumer_mode: TRUST_CG_NATIVE_ADMISSION_CONSUMER_MODE.to_string(),
        surface: tla_trust_cg::NativeInstallGateSurface::TyActivation,
        candidate_disposition: tla_trust_cg::NativeInstallGateDisposition::Installable,
        requested_authority: tla_trust_cg::NativeInstallGateAuthority::ActiveCallable,
        manifest: None,
        manifest_reference: None,
        expected: tla_trust_cg::NativeInstallGateExpectedBindings {
            artifact_id,
            manifest_checksum,
            target_checksum,
            abi_checksum,
            layout_checksum,
            proof_policy_checksum,
            invalidation_checksum,
            current_generation: 1,
        },
        payload_identity: payload_identity.clone(),
        candidate_payload_identity: payload_identity,
        layout_evidence: None,
        proof_evidence: None,
        current_invalidation_checksum: invalidation_checksum,
        artifact_generation: 1,
        current_generation: 1,
        revoked: false,
        deny_control: None,
        replay_identity: None,
        telemetry: None,
    };

    tla_trust_cg::validate_native_install_gate(&input).admission_summary()
}

fn trust_cg_native_admission_runtime_facts(
    stats: &TrustCgBuildStats,
    state_var_count: usize,
) -> String {
    use std::fmt::Write as _;

    let batch = &stats.native_action_callout_batch;
    let batch_setup_row_sha256 = batch.setup_evidence_row_sha256();
    let batch_compile_row_sha256 = batch.compile_telemetry_evidence_row_sha256();
    let batch_adoption_row_sha256 = batch.shared_engine_adoption_evidence_row_sha256();
    let mut facts = format!(
        "actions_compiled={};actions_total={};native_action_callouts_planned={};native_action_callouts_compiled={};native_action_callouts_skipped_shadowed={};native_action_callout_planning_ms={};native_action_callout_compile_ms={};native_action_callout_batch_attempted={};native_action_callout_batch_action_count={};native_action_callout_batch_input_tasks={};native_action_callout_batch_lowered_tasks={};native_action_callout_batch_lowering_failed={};native_action_callout_batch_setup_ms={};native_action_callout_batch_lowering_ms={};native_action_callout_batch_assembly_attempted={};native_action_callout_batch_assembly_ms={};native_action_callout_batch_assembly_failed={};native_action_callout_batch_compile_attempted={};native_action_callout_batch_compile_ms={};native_action_callout_batch_compile_failed={};native_action_callout_batch_compiled={};native_action_callout_batch_fallback_per_action_tasks={};native_action_callout_batch_fallback_per_action_compile_ms={};native_action_callout_batch_fallback_reason={};native_action_callout_batch_artifact_identity_source={};native_action_callout_batch_artifact_identity={};native_action_callout_batch_artifact_semantic_digest={};native_action_callout_batch_artifact_link_digest={};native_action_callout_batch_artifact_cache_digest={};native_action_callout_batch_artifact_cacheable={};native_action_callout_batch_artifact_cache_disabled_by_env={};native_action_callout_batch_prepared_trust_ir_reuse={};native_action_callout_batch_prepared_trust_ir_reuse_identity={};native_action_callout_batch_shared_owner={};native_action_callout_batch_first_beneficiary={};native_action_callout_batch_second_beneficiary={};native_action_callout_batch_extraction_status={};native_action_callout_batch_setup_evidence_row_sha256={};native_action_callout_batch_compile_telemetry_row_sha256={};native_action_callout_batch_shared_engine_adoption_row_sha256={};native_action_callout_batch_sharding_policy_selected={};native_action_callout_batch_shard_count={};native_action_callout_batch_shard_action_counts={};native_action_callout_batch_shard_estimated_ir_nodes={};native_action_callout_batch_shard_assembly_ms={};native_action_callout_batch_shard_compile_ms={};native_action_callout_batch_shard_stable_ids={};native_action_callout_batch_shard_shared_shape_ids={};native_action_callout_batch_shard_frontend_neutral_reuse_ids={};native_action_callout_batch_shard_digest_input_sha256s={};native_action_callout_batch_warm_cache_enabled={};native_action_callout_batch_warm_cache_lookup_attempted={};native_action_callout_batch_warm_cache_hits={};native_action_callout_batch_warm_cache_misses={};native_action_callout_batch_warm_cache_stores={};native_action_callout_batch_warm_cache_lookup_ms={};native_action_callout_batch_shard_warm_cache_lookup_ms={};native_action_callout_batch_shard_warm_cache_statuses={};native_action_callout_batch_shard_warm_cache_keys={};native_action_callout_batch_shard_warm_cache_guard_digests={};native_action_callout_batch_artifact_materialization_ms={};native_action_callout_batch_shard_artifact_materialization_ms={};native_action_callout_batch_estimated_ir_nodes={};native_action_callout_batch_artifact_count={};native_action_callout_batch_artifact_identities={};native_action_callout_batch_artifact_cache_digests={};native_action_callout_batch_compile_telemetry_row_count={};native_action_callout_batch_shared_engine_adoption_row_count={};invariants_compiled={};invariants_total={};native_invariant_callout_compile_ms={};state_constraints_compiled={};state_constraints_total={};native_state_constraint_callout_compile_ms={};state_var_count={};total_compile_ms={}",
        stats.actions_compiled,
        stats.actions_total(),
        stats.native_action_callouts_planned,
        stats.native_action_callouts_compiled,
        stats.native_action_callouts_skipped_shadowed,
        stats.native_action_callout_planning_ms,
        stats.native_action_callout_compile_ms,
        batch.attempted,
        batch.action_count,
        batch.input_tasks,
        batch.lowered_tasks,
        batch.lowering_failed,
        batch.setup_ms,
        batch.lowering_ms,
        batch.batch_assembly_attempted,
        batch.batch_assembly_ms,
        batch.batch_assembly_failed,
        batch.batch_compile_attempted,
        batch.batch_compile_ms,
        batch.batch_compile_failed,
        batch.batch_compiled,
        batch.fallback_per_action_tasks,
        batch.fallback_per_action_compile_ms,
        batch.fallback_reason_code(),
        TrustCgNativeActionCalloutBatchStats::optional_value(batch.artifact_identity_source),
        TrustCgNativeActionCalloutBatchStats::optional_value(batch.artifact_identity.as_deref()),
        TrustCgNativeActionCalloutBatchStats::optional_value(batch.artifact_semantic_digest.as_deref()),
        TrustCgNativeActionCalloutBatchStats::optional_value(batch.artifact_link_digest.as_deref()),
        TrustCgNativeActionCalloutBatchStats::optional_value(batch.artifact_cache_digest.as_deref()),
        batch.artifact_cacheable,
        batch.artifact_cache_disabled_by_env,
        TrustCgNativeActionCalloutBatchStats::optional_value(batch.prepared_trust_ir_reuse),
        TrustCgNativeActionCalloutBatchStats::optional_value(batch.prepared_trust_ir_reuse_identity.as_deref()),
        TrustCgNativeActionCalloutBatchStats::optional_value(batch.shared_owner),
        TrustCgNativeActionCalloutBatchStats::optional_value(batch.first_beneficiary),
        TrustCgNativeActionCalloutBatchStats::optional_value(batch.second_beneficiary),
        TrustCgNativeActionCalloutBatchStats::optional_value(batch.extraction_status),
        TrustCgNativeActionCalloutBatchStats::optional_value(batch_setup_row_sha256.as_deref()),
        TrustCgNativeActionCalloutBatchStats::optional_value(batch_compile_row_sha256.as_deref()),
        TrustCgNativeActionCalloutBatchStats::optional_value(batch_adoption_row_sha256.as_deref()),
        batch.sharding_policy_selected,
        batch.shard_count,
        TrustCgNativeActionCalloutBatchStats::csv_usize(&batch.shard_action_counts),
        TrustCgNativeActionCalloutBatchStats::csv_usize(&batch.shard_estimated_ir_nodes),
        TrustCgNativeActionCalloutBatchStats::csv_u64(&batch.shard_assembly_ms),
        TrustCgNativeActionCalloutBatchStats::csv_u64(&batch.shard_compile_ms),
        TrustCgNativeActionCalloutBatchStats::csv_strings(&batch.shard_stable_ids),
        TrustCgNativeActionCalloutBatchStats::csv_strings(&batch.shard_shared_shape_ids),
        TrustCgNativeActionCalloutBatchStats::csv_strings(
            &batch.shard_frontend_neutral_reuse_ids,
        ),
        TrustCgNativeActionCalloutBatchStats::csv_strings(&batch.shard_digest_input_sha256s),
        batch.warm_cache_enabled,
        batch.warm_cache_lookup_attempted,
        batch.warm_cache_hits,
        batch.warm_cache_misses,
        batch.warm_cache_stores,
        batch.warm_cache_lookup_ms,
        TrustCgNativeActionCalloutBatchStats::csv_u64(&batch.shard_warm_cache_lookup_ms),
        TrustCgNativeActionCalloutBatchStats::csv_strings(&batch.shard_warm_cache_statuses),
        TrustCgNativeActionCalloutBatchStats::csv_strings(&batch.shard_warm_cache_keys),
        TrustCgNativeActionCalloutBatchStats::csv_strings(&batch.shard_warm_cache_guard_digests),
        batch.artifact_materialization_ms,
        TrustCgNativeActionCalloutBatchStats::csv_u64(&batch.shard_artifact_materialization_ms),
        batch.estimated_ir_nodes,
        batch.artifact_identities.len(),
        TrustCgNativeActionCalloutBatchStats::csv_strings(&batch.artifact_identities),
        TrustCgNativeActionCalloutBatchStats::csv_strings(&batch.artifact_cache_digests),
        batch.compile_telemetry_evidence_rows.len(),
        batch.shared_engine_adoption_evidence_rows.len(),
        stats.invariants_compiled,
        stats.invariants_total(),
        stats.native_invariant_callout_compile_ms,
        stats.state_constraints_compiled,
        stats.state_constraints_total(),
        stats.native_state_constraint_callout_compile_ms,
        state_var_count,
        stats.total_compile_ms,
    );
    let _ = write!(
        facts,
        ";native_action_callout_batch_semantic_trust_ir_artifact_digest={};native_action_callout_batch_process_local_link_digest={};native_action_callout_batch_artifact_semantic_digests={};native_action_callout_batch_artifact_link_digests={};native_action_callout_batch_compile_preset={};native_action_callout_batch_compile_presets={};native_action_callout_batch_host_symbol_map_count={};native_action_callout_batch_shard_host_symbol_map_counts={};native_action_callout_batch_runtime_setup_temperature_label={};native_action_callout_batch_runtime_setup_temperature_labels={};native_action_callout_batch_runtime_setup_cache_label={};native_action_callout_batch_runtime_setup_cache_labels={};native_action_callout_batch_artifact_admission_status={};native_action_callout_batch_artifact_admission_statuses={};native_action_callout_batch_artifact_admission_fail_closed={};native_action_callout_batch_artifact_admission_fail_closed_values={};native_action_callout_batch_artifact_admission_missing_fields={};native_action_callout_batch_artifact_admission_rejection_reasons={}",
        TrustCgNativeActionCalloutBatchStats::optional_value(batch.artifact_semantic_digest.as_deref()),
        TrustCgNativeActionCalloutBatchStats::optional_value(batch.artifact_link_digest.as_deref()),
        TrustCgNativeActionCalloutBatchStats::csv_strings(&batch.artifact_semantic_digests),
        TrustCgNativeActionCalloutBatchStats::csv_strings(&batch.artifact_link_digests),
        TrustCgNativeActionCalloutBatchStats::optional_value(batch.batch_compile_preset.as_deref()),
        TrustCgNativeActionCalloutBatchStats::csv_strings(&batch.batch_compile_presets),
        TrustCgNativeActionCalloutBatchStats::optional_usize(batch.host_symbol_map_count),
        TrustCgNativeActionCalloutBatchStats::csv_usize(&batch.shard_host_symbol_map_counts),
        TrustCgNativeActionCalloutBatchStats::optional_value(batch.runtime_setup_temperature_label),
        TrustCgNativeActionCalloutBatchStats::csv_strings(&batch.runtime_setup_temperature_labels),
        TrustCgNativeActionCalloutBatchStats::optional_value(batch.runtime_setup_cache_label),
        TrustCgNativeActionCalloutBatchStats::csv_strings(&batch.runtime_setup_cache_labels),
        TrustCgNativeActionCalloutBatchStats::optional_value(
            batch.batch_artifact_admission_status.as_deref(),
        ),
        TrustCgNativeActionCalloutBatchStats::csv_strings(&batch.batch_artifact_admission_statuses),
        TrustCgNativeActionCalloutBatchStats::optional_bool(
            batch.batch_artifact_admission_fail_closed,
        ),
        TrustCgNativeActionCalloutBatchStats::csv_bools(
            &batch.batch_artifact_admission_fail_closed_values,
        ),
        TrustCgNativeActionCalloutBatchStats::csv_strings(
            &batch.batch_artifact_admission_missing_fields,
        ),
        TrustCgNativeActionCalloutBatchStats::csv_strings(
            &batch.batch_artifact_admission_rejection_reasons,
        ),
    );
    facts
}

fn trust_cg_native_admission_checksum(label: &str, value: &str) -> tla_trust_cg::ArtifactChecksum {
    tla_trust_cg::ArtifactChecksum::for_bytes(
        format!("ty.trust_cg.native_admission.{label}.v1\0{value}").as_bytes(),
    )
}

pub(in crate::check) fn trust_cg_native_admission_sha256(label: &str, value: &str) -> String {
    use std::fmt::Write as _;

    let mut hasher = Sha256::new();
    hasher.update(b"ty.trust_cg.native_admission.sha256.v1");
    hasher.update([0]);
    hasher.update(label.as_bytes());
    hasher.update([0]);
    hasher.update(value.as_bytes());
    let digest = hasher.finalize();

    let mut out = String::from("sha256:");
    for byte in digest {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

fn render_trust_cg_native_admission_evidence_report(
    summary: &tla_trust_cg::NativeInstallGateAdmissionSummary,
    stats: &TrustCgBuildStats,
) -> TrustCgNativeAdmissionEvidenceReport {
    let reason_code = summary.reason_code.unwrap_or("none");
    let production_selected = summary.disposition == "installable"
        && summary.install_authority != "none"
        && summary.actions.ty_native_activate;
    let fail_closed = !production_selected;
    let proof_report_sha256 = summary.proof_report_sha256.as_deref().unwrap_or("none");
    let telemetry_event_id = summary.telemetry_event_id.as_deref().unwrap_or("none");
    let telemetry_record_sha256 = summary.telemetry_record_sha256.as_deref().unwrap_or("none");
    let replay_root_sha256 = summary.replay_root_sha256.as_deref().unwrap_or("none");
    let install_consumer_verdict_sha256 = summary
        .install_consumer_verdict_sha256
        .as_deref()
        .unwrap_or("none");
    let admission_evidence_sha256 = summary
        .admission_evidence_sha256
        .as_deref()
        .unwrap_or("none");
    let batch = &stats.native_action_callout_batch;
    let batch_setup_evidence_row = batch.setup_evidence_row();
    let batch_setup_row_sha256 = batch
        .setup_evidence_row_sha256()
        .unwrap_or_else(|| "none".to_string());
    let batch_compile_row_sha256 = batch
        .compile_telemetry_evidence_row_sha256()
        .unwrap_or_else(|| "none".to_string());
    let batch_adoption_row_sha256 = batch
        .shared_engine_adoption_evidence_row_sha256()
        .unwrap_or_else(|| "none".to_string());

    let mut fields = Vec::new();
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "source",
        "NativeInstallGateAdmissionSummary",
    );
    push_trust_cg_native_admission_evidence_field(&mut fields, "schema", summary.schema);
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "schema_version",
        summary.schema_version,
    );
    push_trust_cg_native_admission_evidence_field(&mut fields, "consumer", &summary.consumer);
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "consumer_mode",
        &summary.consumer_mode,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "kind",
        TRUST_CG_NATIVE_ADMISSION_KIND,
    );
    push_trust_cg_native_admission_evidence_field(&mut fields, "surface", summary.surface);
    push_trust_cg_native_admission_evidence_field(&mut fields, "disposition", summary.disposition);
    push_trust_cg_native_admission_evidence_field(&mut fields, "status_code", summary.disposition);
    push_trust_cg_native_admission_evidence_field(&mut fields, "rejection_code", reason_code);
    push_trust_cg_native_admission_evidence_field(&mut fields, "reason_code", reason_code);
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "requested_authority",
        summary.requested_authority,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "install_authority",
        summary.install_authority,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "production_selected",
        production_selected,
    );
    push_trust_cg_native_admission_evidence_field(&mut fields, "fail_closed", fail_closed);
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "actions_expose_callable",
        summary.actions.expose_callable,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "actions_typed_symbol_lookup",
        summary.actions.typed_symbol_lookup,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "actions_insert_installable_cache",
        summary.actions.insert_installable_cache,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "actions_accept_installable_cache_hit",
        summary.actions.accept_installable_cache_hit,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "actions_release_installable",
        summary.actions.release_installable,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "actions_ay_registry_insert",
        summary.actions.ay_registry_insert,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "actions_ty_native_activate",
        summary.actions.ty_native_activate,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "actions_useful_native_eligible",
        summary.actions.useful_native_eligible,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "useful_native_delta",
        summary.useful_native_delta,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "actions_compiled",
        stats.actions_compiled,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "actions_total",
        stats.actions_total(),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callouts_planned",
        stats.native_action_callouts_planned,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callouts_compiled",
        stats.native_action_callouts_compiled,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callouts_skipped_shadowed",
        stats.native_action_callouts_skipped_shadowed,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_planning_ms",
        stats.native_action_callout_planning_ms,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_compile_ms",
        stats.native_action_callout_compile_ms,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_attempted",
        batch.attempted,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_action_count",
        batch.action_count,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_input_tasks",
        batch.input_tasks,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_lowered_tasks",
        batch.lowered_tasks,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_lowering_failed",
        batch.lowering_failed,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_setup_ms",
        batch.setup_ms,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_lowering_ms",
        batch.lowering_ms,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_assembly_attempted",
        batch.batch_assembly_attempted,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_assembly_ms",
        batch.batch_assembly_ms,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_assembly_failed",
        batch.batch_assembly_failed,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_compile_attempted",
        batch.batch_compile_attempted,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_compile_ms",
        batch.batch_compile_ms,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_compile_failed",
        batch.batch_compile_failed,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_compiled",
        batch.batch_compiled,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_sharding_policy_selected",
        batch.sharding_policy_selected,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_shard_count",
        batch.shard_count,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_shard_action_counts",
        TrustCgNativeActionCalloutBatchStats::csv_usize(&batch.shard_action_counts),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_shard_estimated_ir_nodes",
        TrustCgNativeActionCalloutBatchStats::csv_usize(&batch.shard_estimated_ir_nodes),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_shard_assembly_ms",
        TrustCgNativeActionCalloutBatchStats::csv_u64(&batch.shard_assembly_ms),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_shard_compile_ms",
        TrustCgNativeActionCalloutBatchStats::csv_u64(&batch.shard_compile_ms),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_shard_stable_ids",
        TrustCgNativeActionCalloutBatchStats::csv_strings(&batch.shard_stable_ids),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_shard_shared_shape_ids",
        TrustCgNativeActionCalloutBatchStats::csv_strings(&batch.shard_shared_shape_ids),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_shard_frontend_neutral_reuse_ids",
        TrustCgNativeActionCalloutBatchStats::csv_strings(&batch.shard_frontend_neutral_reuse_ids),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_shard_digest_input_sha256s",
        TrustCgNativeActionCalloutBatchStats::csv_strings(&batch.shard_digest_input_sha256s),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_warm_cache_enabled",
        batch.warm_cache_enabled,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_warm_cache_lookup_attempted",
        batch.warm_cache_lookup_attempted,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_warm_cache_hits",
        batch.warm_cache_hits,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_warm_cache_misses",
        batch.warm_cache_misses,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_warm_cache_stores",
        batch.warm_cache_stores,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_warm_cache_lookup_ms",
        batch.warm_cache_lookup_ms,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_shard_warm_cache_lookup_ms",
        TrustCgNativeActionCalloutBatchStats::csv_u64(&batch.shard_warm_cache_lookup_ms),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_shard_warm_cache_statuses",
        TrustCgNativeActionCalloutBatchStats::csv_strings(&batch.shard_warm_cache_statuses),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_shard_warm_cache_keys",
        TrustCgNativeActionCalloutBatchStats::csv_strings(&batch.shard_warm_cache_keys),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_shard_warm_cache_guard_digests",
        TrustCgNativeActionCalloutBatchStats::csv_strings(&batch.shard_warm_cache_guard_digests),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_artifact_materialization_ms",
        batch.artifact_materialization_ms,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_shard_artifact_materialization_ms",
        TrustCgNativeActionCalloutBatchStats::csv_u64(&batch.shard_artifact_materialization_ms),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_estimated_ir_nodes",
        batch.estimated_ir_nodes,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_fallback_per_action_tasks",
        batch.fallback_per_action_tasks,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_fallback_per_action_compile_ms",
        batch.fallback_per_action_compile_ms,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_fallback_reason",
        batch.fallback_reason_code(),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_artifact_identity_source",
        TrustCgNativeActionCalloutBatchStats::optional_value(batch.artifact_identity_source),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_artifact_identity",
        TrustCgNativeActionCalloutBatchStats::optional_value(batch.artifact_identity.as_deref()),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_artifact_semantic_digest",
        TrustCgNativeActionCalloutBatchStats::optional_value(
            batch.artifact_semantic_digest.as_deref(),
        ),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_artifact_link_digest",
        TrustCgNativeActionCalloutBatchStats::optional_value(batch.artifact_link_digest.as_deref()),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_artifact_cache_digest",
        TrustCgNativeActionCalloutBatchStats::optional_value(
            batch.artifact_cache_digest.as_deref(),
        ),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_semantic_trust_ir_artifact_digest",
        TrustCgNativeActionCalloutBatchStats::optional_value(
            batch.artifact_semantic_digest.as_deref(),
        ),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_process_local_link_digest",
        TrustCgNativeActionCalloutBatchStats::optional_value(batch.artifact_link_digest.as_deref()),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_artifact_semantic_digests",
        TrustCgNativeActionCalloutBatchStats::csv_strings(&batch.artifact_semantic_digests),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_artifact_link_digests",
        TrustCgNativeActionCalloutBatchStats::csv_strings(&batch.artifact_link_digests),
    );
    let batch_telemetry_descriptor = tla_trust_cg::batch_jit_compile_telemetry_descriptor();
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_compile_telemetry_schema",
        batch_telemetry_descriptor.schema,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_compile_telemetry_schema_version",
        batch_telemetry_descriptor.schema_version,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_compile_telemetry_row_kind",
        batch_telemetry_descriptor.row_kind,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_compile_preset",
        TrustCgNativeActionCalloutBatchStats::optional_value(batch.batch_compile_preset.as_deref()),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_compile_presets",
        TrustCgNativeActionCalloutBatchStats::csv_strings(&batch.batch_compile_presets),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_host_symbol_map_count",
        TrustCgNativeActionCalloutBatchStats::optional_usize(batch.host_symbol_map_count),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_shard_host_symbol_map_counts",
        TrustCgNativeActionCalloutBatchStats::csv_usize(&batch.shard_host_symbol_map_counts),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_runtime_setup_temperature_label",
        TrustCgNativeActionCalloutBatchStats::optional_value(batch.runtime_setup_temperature_label),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_runtime_setup_temperature_labels",
        TrustCgNativeActionCalloutBatchStats::csv_strings(&batch.runtime_setup_temperature_labels),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_runtime_setup_cache_label",
        TrustCgNativeActionCalloutBatchStats::optional_value(batch.runtime_setup_cache_label),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_runtime_setup_cache_labels",
        TrustCgNativeActionCalloutBatchStats::csv_strings(&batch.runtime_setup_cache_labels),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_artifact_admission_schema",
        tla_trust_cg::TRUST_CG_BATCH_JIT_ARTIFACT_ADMISSION_SCHEMA,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_artifact_admission_schema_version",
        tla_trust_cg::TRUST_CG_BATCH_JIT_ARTIFACT_ADMISSION_SCHEMA_VERSION,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_artifact_admission_status",
        TrustCgNativeActionCalloutBatchStats::optional_value(
            batch.batch_artifact_admission_status.as_deref(),
        ),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_artifact_admission_statuses",
        TrustCgNativeActionCalloutBatchStats::csv_strings(&batch.batch_artifact_admission_statuses),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_artifact_admission_fail_closed",
        TrustCgNativeActionCalloutBatchStats::optional_bool(
            batch.batch_artifact_admission_fail_closed,
        ),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_artifact_admission_fail_closed_values",
        TrustCgNativeActionCalloutBatchStats::csv_bools(
            &batch.batch_artifact_admission_fail_closed_values,
        ),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_artifact_admission_missing_fields",
        TrustCgNativeActionCalloutBatchStats::csv_strings(
            &batch.batch_artifact_admission_missing_fields,
        ),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_artifact_admission_rejection_reasons",
        TrustCgNativeActionCalloutBatchStats::csv_strings(
            &batch.batch_artifact_admission_rejection_reasons,
        ),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_artifact_cacheable",
        batch.artifact_cacheable,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_artifact_cache_disabled_by_env",
        batch.artifact_cache_disabled_by_env,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_artifact_count",
        batch.artifact_identities.len(),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_artifact_identities",
        TrustCgNativeActionCalloutBatchStats::csv_strings(&batch.artifact_identities),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_artifact_cache_digests",
        TrustCgNativeActionCalloutBatchStats::csv_strings(&batch.artifact_cache_digests),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_prepared_trust_ir_reuse",
        TrustCgNativeActionCalloutBatchStats::optional_value(batch.prepared_trust_ir_reuse),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_prepared_trust_ir_reuse_identity",
        TrustCgNativeActionCalloutBatchStats::optional_value(
            batch.prepared_trust_ir_reuse_identity.as_deref(),
        ),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_shared_owner",
        TrustCgNativeActionCalloutBatchStats::optional_value(batch.shared_owner),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_first_beneficiary",
        TrustCgNativeActionCalloutBatchStats::optional_value(batch.first_beneficiary),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_second_beneficiary",
        TrustCgNativeActionCalloutBatchStats::optional_value(batch.second_beneficiary),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_extraction_status",
        TrustCgNativeActionCalloutBatchStats::optional_value(batch.extraction_status),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_setup_evidence_row_sha256",
        &batch_setup_row_sha256,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_compile_telemetry_row_sha256",
        &batch_compile_row_sha256,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_shared_engine_adoption_row_sha256",
        &batch_adoption_row_sha256,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_compile_telemetry_row_count",
        batch.compile_telemetry_evidence_rows.len(),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_action_callout_batch_shared_engine_adoption_row_count",
        batch.shared_engine_adoption_evidence_rows.len(),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "invariants_compiled",
        stats.invariants_compiled,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "invariants_total",
        stats.invariants_total(),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_invariant_callout_compile_ms",
        stats.native_invariant_callout_compile_ms,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "state_constraints_compiled",
        stats.state_constraints_compiled,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "state_constraints_total",
        stats.state_constraints_total(),
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_state_constraint_callout_compile_ms",
        stats.native_state_constraint_callout_compile_ms,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "total_compile_ms",
        stats.total_compile_ms,
    );
    push_trust_cg_native_admission_evidence_field(&mut fields, "packet_hash", summary.packet_hash);
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "persisted_packet_hash",
        summary.persisted_packet_hash,
    );
    push_trust_cg_native_admission_evidence_field(&mut fields, "artifact_id", &summary.artifact_id);
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "manifest_checksum",
        summary.manifest_checksum,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "source_sha256",
        &summary.source_sha256,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "trust_ir_sha256",
        &summary.trust_ir_sha256,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "native_payload_sha256",
        &summary.native_payload_sha256,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "target_checksum",
        summary.target_checksum,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "abi_checksum",
        summary.abi_checksum,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "layout_checksum",
        summary.layout_checksum,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "proof_policy_checksum",
        summary.proof_policy_checksum,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "invalidation_checksum",
        summary.invalidation_checksum,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "proof_report_sha256",
        proof_report_sha256,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "counter_scope",
        &summary.counter_scope,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "telemetry_event_id",
        telemetry_event_id,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "telemetry_record_sha256",
        telemetry_record_sha256,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "replay_root_sha256",
        replay_root_sha256,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "install_consumer_verdict_sha256",
        install_consumer_verdict_sha256,
    );
    push_trust_cg_native_admission_evidence_field(
        &mut fields,
        "admission_evidence_sha256",
        admission_evidence_sha256,
    );

    let mut sidecar_evidence_rows = batch_setup_evidence_row.into_iter().collect::<Vec<_>>();
    if batch.compile_telemetry_evidence_rows.is_empty() {
        sidecar_evidence_rows.extend(batch.compile_telemetry_evidence_row.iter().cloned());
    } else {
        sidecar_evidence_rows.extend(batch.compile_telemetry_evidence_rows.iter().cloned());
    }
    if batch.shared_engine_adoption_evidence_rows.is_empty() {
        sidecar_evidence_rows.extend(batch.shared_engine_adoption_evidence_row.iter().cloned());
    } else {
        sidecar_evidence_rows.extend(batch.shared_engine_adoption_evidence_rows.iter().cloned());
    }

    TrustCgNativeAdmissionEvidenceReport::new(
        "trust-cg trust_cg_admission_blocker",
        fields,
        sidecar_evidence_rows,
    )
}

fn push_trust_cg_native_admission_evidence_field(
    fields: &mut Vec<(String, String)>,
    key: &str,
    value: impl ToString,
) {
    fields.push((key.to_string(), value.to_string()));
}

impl TrustCgNativeCache {
    /// Check whether EXISTS-binding specialization (#4270) is enabled.
    ///
    /// Enabled by default; set `TY_TRUST_CG_EXISTS=0` to opt out. When enabled,
    /// `TrustCgNativeCache::build` consumes `BindingSpec` entries extracted from
    /// `split_action_meta`, runs a typed bytecode specialization to bake each
    /// binding register, and compiles the resulting arity-0 function through the
    /// trust-ir → trust-codegen pipeline. When disabled, the
    /// cache ignores specializations and behaves as before (binding-heavy
    /// actions fall back to the interpreter).
    pub(in crate::check) fn exists_enabled() -> bool {
        std::env::var("TY_TRUST_CG_EXISTS").map_or(true, |v| v != "0")
    }

    fn has_residual_exists_opcode(func: &tla_tir::bytecode::BytecodeFunction) -> bool {
        func.instructions.iter().any(|op| {
            matches!(
                op,
                tla_tir::bytecode::Opcode::ExistsBegin { .. }
                    | tla_tir::bytecode::Opcode::ExistsNext { .. }
            )
        })
    }

    fn specialization_formal_value_literals<'a>(
        spec: &'a tla_jit_abi::BindingSpec,
        base_arity: u8,
        key: &str,
    ) -> Result<&'a [tla_value::Value], String> {
        let formal_arity = usize::from(base_arity);
        let expected_key =
            tla_jit_abi::binding_key_for_values(&spec.action_name, &spec.binding_value_literals)
                .ok_or_else(|| {
                    format!(
                        "[trust-cg] specialization '{key}': binding literals for base action '{}' are not finite LoadConst-compatible values",
                        spec.action_name,
                    )
                })?;
        if expected_key != spec.binding_key {
            return Err(format!(
                "[trust-cg] specialization '{key}': precomputed binding key for base action '{}' is inconsistent with typed literals (expected '{expected_key}')",
                spec.action_name,
            ));
        }
        Self::specialization_values_match_raw(
            key,
            &spec.action_name,
            "binding",
            &spec.binding_values,
            &spec.binding_value_literals,
        )?;
        if spec.formal_value_literals.len() == formal_arity {
            Self::specialization_values_match_raw(
                key,
                &spec.action_name,
                "formal",
                &spec.formal_values,
                &spec.formal_value_literals,
            )?;
            Ok(&spec.formal_value_literals)
        } else {
            Err(format!(
                "[trust-cg] specialization '{key}': formal binding arity mismatch for base action '{}' ({} raw formal values, {} typed formal values, {} typed key values, arity {})",
                spec.action_name,
                spec.formal_values.len(),
                spec.formal_value_literals.len(),
                spec.binding_value_literals.len(),
                base_arity,
            ))
        }
    }

    fn raw_specialization_value(value: &tla_value::Value) -> Option<i64> {
        match value {
            tla_value::Value::SmallInt(value) => Some(*value),
            tla_value::Value::Int(value) => {
                use num_traits::ToPrimitive;
                value.to_i64()
            }
            tla_value::Value::Bool(value) => Some(i64::from(*value)),
            tla_value::Value::String(value) | tla_value::Value::ModelValue(value) => {
                Some(i64::from(tla_core::intern_name(value.as_ref()).0))
            }
            _ => None,
        }
    }

    fn specialization_values_match_raw(
        key: &str,
        action_name: &str,
        label: &str,
        raw_values: &[i64],
        typed_values: &[tla_value::Value],
    ) -> Result<(), String> {
        if !tla_jit_abi::values_are_finite_binding_literals(typed_values) {
            return Err(format!(
                "[trust-cg] specialization '{key}': {label} binding for base action '{action_name}' has unsupported non-finite typed literals",
            ));
        }
        if typed_values
            .iter()
            .any(|typed_value| Self::raw_specialization_value(typed_value).is_none())
        {
            if raw_values.is_empty() {
                return Ok(());
            }
            return Err(format!(
                "[trust-cg] specialization '{key}': {label} binding for base action '{action_name}' mixes raw values with finite compound typed literals",
            ));
        }
        if raw_values.len() != typed_values.len() {
            return Err(format!(
                "[trust-cg] specialization '{key}': {label} binding arity mismatch for base action '{action_name}' ({} raw values, {} typed values)",
                raw_values.len(),
                typed_values.len(),
            ));
        }
        for (idx, (raw_value, typed_value)) in
            raw_values.iter().zip(typed_values.iter()).enumerate()
        {
            let Some(typed_raw_value) = Self::raw_specialization_value(typed_value) else {
                return Err(format!(
                    "[trust-cg] specialization '{key}': {label} binding value {idx} for base action '{action_name}' has unsupported non-scalar type {}",
                    typed_value.type_name(),
                ));
            };
            if typed_raw_value != *raw_value {
                return Err(format!(
                    "[trust-cg] specialization '{key}': {label} binding value {idx} for base action '{action_name}' encodes raw {typed_raw_value}, expected {raw_value}",
                ));
            }
        }
        Ok(())
    }

    fn typed_specialization_raw_key_collisions(
        specializations: &[BindingSpec],
    ) -> FxHashSet<String> {
        let mut seen: FxHashMap<String, (Vec<tla_value::Value>, Vec<tla_value::Value>)> =
            FxHashMap::default();
        let mut collisions = FxHashSet::default();
        for spec in specializations {
            let key = spec.binding_key.clone();
            let signature = (
                spec.binding_value_literals.clone(),
                spec.formal_value_literals.clone(),
            );
            if let Some(existing) = seen.get(&key) {
                if existing != &signature {
                    collisions.insert(key);
                }
            } else {
                seen.insert(key, signature);
            }
        }
        collisions
    }

    fn bytecode_function_uses_const_pool(func: &tla_tir::bytecode::BytecodeFunction) -> bool {
        func.instructions.iter().any(|op| {
            matches!(
                op,
                tla_tir::bytecode::Opcode::LoadConst { .. }
                    | tla_tir::bytecode::Opcode::RecordNew { .. }
                    | tla_tir::bytecode::Opcode::RecordGet { .. }
                    | tla_tir::bytecode::Opcode::RecordSet { .. }
                    | tla_tir::bytecode::Opcode::Unchanged { .. }
                    | tla_tir::bytecode::Opcode::MakeClosure { .. }
                    | tla_tir::bytecode::Opcode::CallExternal { .. }
            )
        })
    }

    fn sorted_inner_exists_expansions(
        func: &tla_tir::bytecode::BytecodeFunction,
        const_pool: Option<&tla_tir::bytecode::ConstantPool>,
    ) -> Option<Vec<tla_tir::bytecode::ExpandedAction>> {
        let mut expanded =
            tla_tir::bytecode::expand_inner_exists_preserving_offsets(func, const_pool)?;
        expanded.sort_by(|a, b| a.inner_binding_values.cmp(&b.inner_binding_values));
        Some(expanded)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn sorted_runtime_guarded_inner_exists_expansions(
        func: &tla_tir::bytecode::BytecodeFunction,
        state_layout: Option<&tla_jit_abi::StateLayout>,
    ) -> Option<Vec<tla_tir::bytecode::ExpandedAction>> {
        Self::sorted_runtime_guarded_inner_exists_expansion_plans(func, state_layout, None)
            .map(|plans| plans.into_iter().map(|plan| plan.action).collect())
    }

    fn sorted_runtime_guarded_inner_exists_expansion_plans(
        func: &tla_tir::bytecode::BytecodeFunction,
        state_layout: Option<&tla_jit_abi::StateLayout>,
        const_pool: Option<&tla_tir::bytecode::ConstantPool>,
    ) -> Option<Vec<RuntimeGuardedInnerExistsExpansion>> {
        let state_layout = state_layout?;
        let info = Self::single_inner_exists_info(func)?;
        // SOUNDNESS GUARD (fail-closed): same disjunctive-sibling-drop hazard as
        // the static path. Pinning the single `\E` to per-witness specializations
        // and short-circuiting an enclosing `\/` with `JumpTrue` would silently
        // drop the sibling arm's successors. Reject -> interpreter fallback.
        if tla_tir::bytecode::static_expansion_drops_sibling_successor(
            func,
            std::slice::from_ref(&info),
        ) {
            return None;
        }
        let domain =
            Self::runtime_inner_exists_domain_proof(func, &info, state_layout, const_pool)?;
        if const_pool.is_none()
            && func
                .instructions
                .iter()
                .any(|op| matches!(op, tla_tir::bytecode::Opcode::LoadConst { .. }))
        {
            return None;
        }
        if domain.values.is_empty()
            || domain.values.len() > tla_tir::bytecode::MAX_INNER_DOMAIN_SIZE
        {
            return None;
        }

        let mut domain_values = domain.values;
        domain_values.sort();
        let mut expanded = Vec::with_capacity(domain_values.len());
        for (witness_idx, value) in domain_values.into_iter().enumerate() {
            let mut expansion_const_pool = const_pool.cloned().unwrap_or_default();
            let binding_literal = runtime_binding_value_literal(&value);
            // WP-16: only the canonical (first, in sorted binding order)
            // witness kernel may report enablement via a path that bypasses
            // the exists region; every sibling is participation-gated so
            // witness-independent successors (and their transition counts)
            // are emitted exactly once, matching the interpreter's
            // enumeration accounting.
            let specialized = Self::rewrite_runtime_guarded_inner_exists(
                func,
                &info,
                value.clone(),
                &mut expansion_const_pool,
                witness_idx > 0,
            )?;
            let raw_values = runtime_binding_value_raw(&value)
                .map(|raw| vec![raw])
                .unwrap_or_default();
            expanded.push(RuntimeGuardedInnerExistsExpansion {
                action: tla_tir::bytecode::ExpandedAction {
                    func: specialized,
                    inner_binding_values: raw_values,
                },
                const_pool: Some(expansion_const_pool),
                action_local_set_domain_proof: domain.action_local_set_domain_proof.clone(),
                native_fused_proof: domain.native_fused_proof.clone(),
                inner_binding_literals: Some(vec![binding_literal]),
            });
        }
        expanded.sort_by(|a, b| {
            a.inner_binding_literals
                .cmp(&b.inner_binding_literals)
                .then_with(|| {
                    a.action
                        .inner_binding_values
                        .cmp(&b.action.inner_binding_values)
                })
        });
        Some(expanded)
    }

    /// Find the single inner `ExistsBegin`/`ExistsNext` pair eligible for
    /// runtime-guarded expansion, or `None` when there is not exactly one.
    ///
    /// SOUNDNESS — why this is capped at exactly one pair (fail closed on 2+):
    ///
    /// The runtime-guarded expansion pins each inner-EXISTS binding to a
    /// concrete witness from a compile-time-constant SUPERSET universe and
    /// guards it with a runtime `SetIn`; the compiled action then produces at
    /// most one successor per expanded key. With a SINGLE pair, the union of
    /// per-witness keys reproduces the full successor set of `\E x \in S: B(x)`.
    ///
    /// With TWO OR MORE pairs the only expansion this framework can emit is the
    /// CARTESIAN PRODUCT of the pairs' witnesses (each specialized function pins
    /// EVERY pair, because `rewrite_runtime_guarded_inner_exists` replaces every
    /// `ExistsBegin`/`ExistsNext` with concrete loads — it cannot leave a
    /// residual loop, which `has_residual_exists_opcode` rejects). That product
    /// is sound ONLY for CONJUNCTIVELY-nested existentials, where every pair's
    /// body must hold simultaneously. It is UNSOUND for DISJUNCTIVELY-separated
    /// existentials (`(\E k \in S: B1) \/ (\E i \in T: B2)`, e.g. Bakery `e3`):
    /// the bytecode short-circuits the disjunction with a `JumpTrue`, so once
    /// the first disjunct's pinned witness succeeds (its enumerated universe is
    /// an exact superset, so its `SetIn` guard never rejects when the witness is
    /// in-domain), the second disjunct's body is never executed in ANY product
    /// function — silently dropping all of the second arm's reachable
    /// successors. We cannot distinguish "sound conjunctive product" from
    /// "unsound disjunctive product" cheaply enough to expand only the former,
    /// so we fail closed on every multi-pair function and let the interpreter
    /// generate successors. (The statically-resolvable conjunctive case is
    /// already covered by `expand_inner_exists_preserving_offsets`.)
    fn single_inner_exists_info(
        func: &tla_tir::bytecode::BytecodeFunction,
    ) -> Option<tla_tir::bytecode::InnerExistsInfo> {
        let mut result = None;
        for (pc, op) in func.instructions.iter().enumerate() {
            let tla_tir::bytecode::Opcode::ExistsBegin {
                rd,
                r_binding,
                r_domain,
                loop_end,
            } = *op
            else {
                continue;
            };
            // Fail closed on a second pair (see soundness note above), and on
            // ill-formed pairs where the result register aliases a binding or
            // domain register.
            if result.is_some() || rd == r_binding || rd == r_domain {
                return None;
            }
            let target_pc = Self::runtime_exists_target_pc(func.instructions.len(), pc, loop_end)?;
            let mut next_pc = None;
            for scan_pc in (pc + 1)..func.instructions.len().min(target_pc + 1) {
                if let tla_tir::bytecode::Opcode::ExistsNext { loop_begin, .. } =
                    func.instructions[scan_pc]
                {
                    let jump_target = Self::runtime_exists_target_pc(
                        func.instructions.len(),
                        scan_pc,
                        loop_begin,
                    )?;
                    if jump_target == pc + 1 {
                        next_pc = Some(scan_pc);
                        break;
                    }
                }
            }
            let next_pc = next_pc?;
            result = Some(tla_tir::bytecode::InnerExistsInfo {
                begin_pc: pc,
                next_pc,
                r_binding,
                r_domain,
                rd,
                domain: None,
                loop_end_offset: loop_end,
            });
        }
        result
    }

    fn runtime_exists_target_pc(instruction_len: usize, pc: usize, offset: i32) -> Option<usize> {
        let target = i64::try_from(pc).ok()?.checked_add(i64::from(offset))?;
        if target < 0 || target > i64::try_from(instruction_len).ok()? {
            return None;
        }
        usize::try_from(target).ok()
    }

    fn runtime_inner_exists_domain_proof(
        func: &tla_tir::bytecode::BytecodeFunction,
        info: &tla_tir::bytecode::InnerExistsInfo,
        state_layout: &tla_jit_abi::StateLayout,
        const_pool: Option<&tla_tir::bytecode::ConstantPool>,
    ) -> Option<RuntimeInnerExistsDomainProof> {
        let mut shapes: FxHashMap<u8, RuntimeInnerExistsRegisterShape> = FxHashMap::default();
        for op in &func.instructions[..info.begin_pc] {
            match *op {
                tla_tir::bytecode::Opcode::LoadVar { rd, var_idx } => {
                    let shape = state_layout
                        .var_layout(usize::from(var_idx))
                        .and_then(|layout| Self::runtime_shape_from_var_layout(var_idx, layout));
                    Self::runtime_set_register_shape(&mut shapes, rd, shape);
                }
                tla_tir::bytecode::Opcode::LoadPrime { rd, var_idx } => {
                    let shape = state_layout
                        .var_layout(usize::from(var_idx))
                        .and_then(|layout| {
                            Self::runtime_shape_from_var_layout_without_source(layout)
                        });
                    Self::runtime_set_register_shape(&mut shapes, rd, shape);
                }
                tla_tir::bytecode::Opcode::LoadImm { rd, value } => {
                    Self::runtime_set_register_shape(
                        &mut shapes,
                        rd,
                        Some(RuntimeInnerExistsRegisterShape::Scalar {
                            value: Some(tla_jit_abi::SetBitmaskElement::Int(value)),
                        }),
                    );
                }
                tla_tir::bytecode::Opcode::LoadBool { rd, value } => {
                    Self::runtime_set_register_shape(
                        &mut shapes,
                        rd,
                        Some(RuntimeInnerExistsRegisterShape::Scalar {
                            value: Some(tla_jit_abi::SetBitmaskElement::Bool(value)),
                        }),
                    );
                }
                tla_tir::bytecode::Opcode::LoadConst { rd, idx } => {
                    let shape = const_pool
                        .map(|pool| pool.get_value(idx))
                        .and_then(Self::runtime_shape_from_const_value);
                    Self::runtime_set_register_shape(&mut shapes, rd, shape);
                }
                tla_tir::bytecode::Opcode::Move { rd, rs } => {
                    let shape = shapes.get(&rs).cloned();
                    Self::runtime_set_register_shape(&mut shapes, rd, shape);
                }
                tla_tir::bytecode::Opcode::SetEnum { rd, start, count } => {
                    let values = (0..count)
                        .map(|offset| {
                            let reg = start.checked_add(offset)?;
                            match shapes.get(&reg) {
                                Some(RuntimeInnerExistsRegisterShape::Scalar {
                                    value: Some(value),
                                }) => Some(*value),
                                _ => None,
                            }
                        })
                        .collect::<Option<Vec<_>>>()
                        .and_then(|values| {
                            runtime_typed_scalar_values_from_expansion_domain(&values)
                        });
                    let shape =
                        values.map(|values| RuntimeInnerExistsRegisterShape::SetBitmask { values });
                    Self::runtime_set_register_shape(&mut shapes, rd, shape);
                }
                tla_tir::bytecode::Opcode::Range { rd, lo, hi } => {
                    // `lo` must be a compile-time constant integer. `hi` may be
                    // either an exact constant or a *proven upper bound* (e.g.
                    // `Len(seq)` over a proven-capacity sequence). In both cases
                    // the candidate set `lo..=hi_upper` is a superset of the
                    // runtime range `lo..hi`, and the per-candidate runtime
                    // membership guard `i \in (lo..hi)` discards candidates that
                    // are not in the actual range. The set is sound precisely
                    // because `hi_upper` is an upper bound on the runtime `hi`.
                    let lo_val = match shapes.get(&lo) {
                        Some(RuntimeInnerExistsRegisterShape::Scalar {
                            value: Some(tla_jit_abi::SetBitmaskElement::Int(lo)),
                        }) => Some(*lo),
                        _ => None,
                    };
                    let hi_upper = match shapes.get(&hi) {
                        Some(RuntimeInnerExistsRegisterShape::Scalar {
                            value: Some(tla_jit_abi::SetBitmaskElement::Int(hi)),
                        }) => Some(*hi),
                        Some(RuntimeInnerExistsRegisterShape::ScalarIntUpperBound { upper }) => {
                            Some(*upper)
                        }
                        _ => None,
                    };
                    let shape = match (lo_val, hi_upper) {
                        (Some(lo), Some(hi)) if hi >= lo => {
                            let len = hi.checked_sub(lo).and_then(|n| n.checked_add(1));
                            len.and_then(|n| usize::try_from(n).ok())
                                .filter(|&n| n <= tla_tir::bytecode::MAX_INNER_DOMAIN_SIZE)
                                .and_then(|_| {
                                    let values = (lo..=hi)
                                        .map(tla_jit_abi::SetBitmaskElement::Int)
                                        .collect::<Vec<_>>();
                                    runtime_typed_scalar_values_from_expansion_domain(&values)
                                })
                                .map(|values| RuntimeInnerExistsRegisterShape::SetBitmask {
                                    values,
                                })
                        }
                        _ => None,
                    };
                    Self::runtime_set_register_shape(&mut shapes, rd, shape);
                }
                tla_tir::bytecode::Opcode::CallBuiltin {
                    rd,
                    builtin: tla_tir::bytecode::BuiltinOp::Len,
                    args_start,
                    argc: 1,
                } => {
                    // `Len(seq)` where `seq` is a proven-capacity sequence is an
                    // integer provably in `0..=C`. Carry the proven capacity `C`
                    // forward as an upper bound so a wrapping `lo..Len(seq)`
                    // range can be enumerated soundly.
                    let shape = match shapes.get(&args_start) {
                        Some(RuntimeInnerExistsRegisterShape::Function {
                            sequence_capacity_proof: Some(capacity),
                            ..
                        }) => i64::try_from(*capacity).ok().map(|upper| {
                            RuntimeInnerExistsRegisterShape::ScalarIntUpperBound { upper }
                        }),
                        _ => None,
                    };
                    Self::runtime_set_register_shape(&mut shapes, rd, shape);
                }
                tla_tir::bytecode::Opcode::FuncApply { rd, func, arg } => {
                    let shape = match shapes.get(&func).cloned() {
                        Some(RuntimeInnerExistsRegisterShape::Function {
                            source_var_idx,
                            key_values,
                            value: Some(value),
                            ..
                        }) => Self::runtime_shape_from_func_apply_value(
                            rd,
                            arg,
                            source_var_idx,
                            key_values.as_deref(),
                            &value,
                            &shapes,
                        ),
                        _ => None,
                    };
                    Self::runtime_set_register_shape(&mut shapes, rd, shape);
                }
                tla_tir::bytecode::Opcode::FuncExcept { rd, func, .. } => {
                    let shape = match shapes.get(&func).cloned() {
                        Some(RuntimeInnerExistsRegisterShape::Function {
                            key_values,
                            value,
                            sequence_capacity_proof,
                            ..
                        }) => Some(RuntimeInnerExistsRegisterShape::Function {
                            source_var_idx: None,
                            key_values,
                            value,
                            // `EXCEPT` preserves the domain/length, so a proven
                            // capacity bound still holds.
                            sequence_capacity_proof,
                        }),
                        other => other,
                    };
                    Self::runtime_set_register_shape(&mut shapes, rd, shape);
                }
                tla_tir::bytecode::Opcode::SetDiff { rd, r1, .. }
                | tla_tir::bytecode::Opcode::SetIntersect { rd, r1, .. } => {
                    let shape = match shapes.get(&r1) {
                        Some(RuntimeInnerExistsRegisterShape::SetBitmask { .. }) => {
                            shapes.get(&r1).cloned()
                        }
                        Some(RuntimeInnerExistsRegisterShape::TaggedScalarOrSet { proof }) => {
                            Some(RuntimeInnerExistsRegisterShape::SetBitmask {
                                values: proof.universe_values.clone(),
                            })
                        }
                        Some(RuntimeInnerExistsRegisterShape::TaggedScalarOrSetRead { read }) => {
                            Some(RuntimeInnerExistsRegisterShape::SetBitmask {
                                values: read.proof.universe_values.clone(),
                            })
                        }
                        _ => None,
                    };
                    Self::runtime_set_register_shape(&mut shapes, rd, shape);
                }
                tla_tir::bytecode::Opcode::Domain { rd, rs } => {
                    let shape = match shapes.get(&rs) {
                        Some(RuntimeInnerExistsRegisterShape::Function {
                            key_values: Some(values),
                            ..
                        }) => runtime_typed_scalar_values_from_expansion_domain(values)
                            .map(|values| RuntimeInnerExistsRegisterShape::SetBitmask { values }),
                        _ => None,
                    };
                    Self::runtime_set_register_shape(&mut shapes, rd, shape);
                }
                tla_tir::bytecode::Opcode::SetUnion { rd, r1, r2 } => {
                    let shape = match (shapes.get(&r1), shapes.get(&r2)) {
                        (
                            Some(RuntimeInnerExistsRegisterShape::SetBitmask { values: left }),
                            Some(RuntimeInnerExistsRegisterShape::SetBitmask { values: right }),
                        ) => {
                            let mut values = left.clone();
                            values.extend_from_slice(right);
                            values.sort_unstable();
                            values.dedup();
                            runtime_typed_scalar_values_from_expansion_domain(&values).map(
                                |values| RuntimeInnerExistsRegisterShape::SetBitmask { values },
                            )
                        }
                        _ => None,
                    };
                    Self::runtime_set_register_shape(&mut shapes, rd, shape);
                }
                tla_tir::bytecode::Opcode::Powerset { rd, rs } => {
                    let shape = match shapes.get(&rs) {
                        Some(RuntimeInnerExistsRegisterShape::SetBitmask { values }) => {
                            Some(RuntimeInnerExistsRegisterShape::Powerset {
                                base_values: values.clone(),
                            })
                        }
                        Some(RuntimeInnerExistsRegisterShape::TaggedScalarOrSet { proof }) => {
                            Some(RuntimeInnerExistsRegisterShape::Powerset {
                                base_values: proof.universe_values.clone(),
                            })
                        }
                        Some(RuntimeInnerExistsRegisterShape::TaggedScalarOrSetRead { read }) => {
                            Some(RuntimeInnerExistsRegisterShape::Powerset {
                                base_values: read.proof.universe_values.clone(),
                            })
                        }
                        _ => None,
                    };
                    Self::runtime_set_register_shape(&mut shapes, rd, shape);
                }
                tla_tir::bytecode::Opcode::KSubset { rd, base, k } => {
                    let base_values = match shapes.get(&base) {
                        Some(RuntimeInnerExistsRegisterShape::SetBitmask { values }) => {
                            Some(values.clone())
                        }
                        Some(RuntimeInnerExistsRegisterShape::TaggedScalarOrSet { proof }) => {
                            Some(proof.universe_values.clone())
                        }
                        Some(RuntimeInnerExistsRegisterShape::TaggedScalarOrSetRead { read }) => {
                            Some(read.proof.universe_values.clone())
                        }
                        _ => None,
                    };
                    let k = match shapes.get(&k) {
                        Some(RuntimeInnerExistsRegisterShape::Scalar {
                            value: Some(tla_jit_abi::SetBitmaskElement::Int(value)),
                        }) if *value >= 0 => usize::try_from(*value).ok(),
                        _ => None,
                    };
                    let shape = base_values.zip(k).map(|(base_values, k)| {
                        RuntimeInnerExistsRegisterShape::KSubset { base_values, k }
                    });
                    Self::runtime_set_register_shape(&mut shapes, rd, shape);
                }
                tla_tir::bytecode::Opcode::SetFilterBegin {
                    rd, r_domain: base, ..
                } => {
                    // `{ x \in BASE : pred(x) }`. The filtered result is a subset
                    // of BASE, so when BASE is a compile-time-constant finite set
                    // we enumerate BASE as a sound SUPERSET universe and let the
                    // runtime `SetIn` guard reject witnesses failing the predicate.
                    // Non-constant bases fail closed (shape = None).
                    let shape = Self::runtime_set_filter_output_shape_from_base(shapes.get(&base));
                    Self::runtime_set_register_shape(&mut shapes, rd, shape);
                }
                _ => {
                    if let Some(rd) = op.dest_register() {
                        shapes.remove(&rd);
                    }
                }
            }
        }

        match shapes.get(&info.r_domain)? {
            RuntimeInnerExistsRegisterShape::SetBitmask { values } => {
                let values = runtime_scalar_binding_values(values);
                Some(RuntimeInnerExistsDomainProof {
                    native_fused_proof: Self::runtime_guarded_inner_exists_native_fused_proof(
                        &values,
                    ),
                    values,
                    action_local_set_domain_proof: None,
                })
            }
            RuntimeInnerExistsRegisterShape::Powerset { base_values } => {
                let values = runtime_powerset_binding_values(base_values)?;
                Some(RuntimeInnerExistsDomainProof {
                    native_fused_proof: Self::runtime_guarded_inner_exists_native_fused_proof(
                        &values,
                    ),
                    values,
                    action_local_set_domain_proof: None,
                })
            }
            RuntimeInnerExistsRegisterShape::KSubset { base_values, k } => {
                let values = runtime_ksubset_binding_values(base_values, *k)?;
                Some(RuntimeInnerExistsDomainProof {
                    native_fused_proof: Self::runtime_guarded_inner_exists_native_fused_proof(
                        &values,
                    ),
                    values,
                    action_local_set_domain_proof: None,
                })
            }
            RuntimeInnerExistsRegisterShape::TaggedScalarOrSet { proof } => {
                let values = runtime_scalar_binding_values(&proof.universe_values);
                Some(RuntimeInnerExistsDomainProof {
                    native_fused_proof: Self::runtime_guarded_inner_exists_native_fused_proof(
                        &values,
                    ),
                    values,
                    action_local_set_domain_proof: None,
                })
            }
            RuntimeInnerExistsRegisterShape::TaggedScalarOrSetRead { read } => {
                let native_fused_proof =
                    TrustCgInnerExistsExpansionProofKind::ActionLocalTaggedScalarOrSet {
                        source_var_idx: read.source_var_idx,
                        key_reg: read.key_reg,
                        domain_reg: read.domain_reg,
                        key_values: read.key_values.clone(),
                        scalar_kind: read.proof.scalar_kind,
                        proof_source: read.proof.proof_source,
                        universe_values: read.proof.universe_values.clone(),
                    };
                Some(RuntimeInnerExistsDomainProof {
                    values: runtime_scalar_binding_values(&read.proof.universe_values),
                    action_local_set_domain_proof: None,
                    native_fused_proof: Some(native_fused_proof),
                })
            }
            RuntimeInnerExistsRegisterShape::TaggedScalarOrSetUnprovenRead { proof } => {
                Some(RuntimeInnerExistsDomainProof {
                    values: runtime_scalar_binding_values(&proof.universe_values),
                    action_local_set_domain_proof: None,
                    native_fused_proof: None,
                })
            }
            _ => None,
        }
    }

    fn runtime_guarded_inner_exists_native_fused_proof(
        values: &[RuntimeInnerExistsBindingValue],
    ) -> Option<TrustCgInnerExistsExpansionProofKind> {
        if values.is_empty() {
            return None;
        }
        let mut seen = FxHashSet::default();
        if !values.iter().all(|value| seen.insert(value)) {
            return None;
        }
        Some(
            TrustCgInnerExistsExpansionProofKind::RuntimeGuardedFiniteDomain {
                binding_values: values.to_vec(),
            },
        )
    }

    fn runtime_shape_from_func_apply_value(
        domain_reg: u8,
        key_reg: u8,
        source_var_idx: Option<u16>,
        key_values: Option<&[tla_jit_abi::SetBitmaskElement]>,
        value: &RuntimeInnerExistsRegisterShape,
        shapes: &FxHashMap<u8, RuntimeInnerExistsRegisterShape>,
    ) -> Option<RuntimeInnerExistsRegisterShape> {
        match value {
            RuntimeInnerExistsRegisterShape::TaggedScalarOrSet { proof } => {
                let Some(source_var_idx) = source_var_idx else {
                    return Some(RuntimeInnerExistsRegisterShape::TaggedScalarOrSet {
                        proof: proof.clone(),
                    });
                };
                let Some(key_values) = key_values else {
                    return Some(RuntimeInnerExistsRegisterShape::TaggedScalarOrSet {
                        proof: proof.clone(),
                    });
                };
                if !Self::runtime_func_apply_key_proven(shapes.get(&key_reg), key_values) {
                    return Some(
                        RuntimeInnerExistsRegisterShape::TaggedScalarOrSetUnprovenRead {
                            proof: proof.clone(),
                        },
                    );
                }
                Some(RuntimeInnerExistsRegisterShape::TaggedScalarOrSetRead {
                    read: RuntimeTaggedScalarOrSetReadProof {
                        source_var_idx,
                        key_reg,
                        domain_reg,
                        key_values: key_values.to_vec(),
                        proof: proof.clone(),
                    },
                })
            }
            other => Some(other.clone()),
        }
    }

    fn runtime_func_apply_key_proven(
        key_shape: Option<&RuntimeInnerExistsRegisterShape>,
        key_values: &[tla_jit_abi::SetBitmaskElement],
    ) -> bool {
        let Some(RuntimeInnerExistsRegisterShape::Scalar { value: Some(value) }) = key_shape else {
            return false;
        };
        key_values.iter().any(|candidate| candidate == value)
    }

    fn runtime_set_register_shape(
        shapes: &mut FxHashMap<u8, RuntimeInnerExistsRegisterShape>,
        rd: u8,
        shape: Option<RuntimeInnerExistsRegisterShape>,
    ) {
        if let Some(shape) = shape {
            shapes.insert(rd, shape);
        } else {
            shapes.remove(&rd);
        }
    }

    /// Derive a sound enumeration shape for the OUTPUT of a `SetFilterBegin`
    /// (`{ x \in BASE : pred(x) }`) from the shape of its filtered BASE.
    ///
    /// Soundness: the filtered set is always a SUBSET of `BASE`. The runtime
    /// inner-EXISTS expansion enumerates this shape as the candidate witness
    /// universe and emits a `SetIn` guard that can only REJECT witnesses that
    /// fail the (runtime) filter predicate. Enumerating `BASE` is therefore a
    /// constant SUPERSET of the true (narrowed) runtime domain, so no reachable
    /// witness is ever dropped.
    ///
    /// This is ONLY sound when `BASE` is a compile-time-CONSTANT finite set.
    /// For scalar-element bases we collapse any read-backed proof shape into a
    /// plain `SetBitmask` over the constant universe, discarding runtime-read
    /// semantics (the filter output is a narrower set than the base read, so we
    /// must not claim it equals that read). For set-element bases (`Powerset` /
    /// `KSubset`) we propagate the constant base shape unchanged. Any base that
    /// is not provably constant-finite (runtime variable set, unresolved Range,
    /// observed/sampled-bound register, etc.) yields `None`, which fails the
    /// domain proof closed.
    fn runtime_set_filter_output_shape_from_base(
        base_shape: Option<&RuntimeInnerExistsRegisterShape>,
    ) -> Option<RuntimeInnerExistsRegisterShape> {
        match base_shape? {
            RuntimeInnerExistsRegisterShape::SetBitmask { values } => {
                Some(RuntimeInnerExistsRegisterShape::SetBitmask {
                    values: values.clone(),
                })
            }
            RuntimeInnerExistsRegisterShape::TaggedScalarOrSet { proof } => {
                Some(RuntimeInnerExistsRegisterShape::SetBitmask {
                    values: proof.universe_values.clone(),
                })
            }
            RuntimeInnerExistsRegisterShape::TaggedScalarOrSetRead { read } => {
                Some(RuntimeInnerExistsRegisterShape::SetBitmask {
                    values: read.proof.universe_values.clone(),
                })
            }
            RuntimeInnerExistsRegisterShape::Powerset { base_values } => {
                Some(RuntimeInnerExistsRegisterShape::Powerset {
                    base_values: base_values.clone(),
                })
            }
            RuntimeInnerExistsRegisterShape::KSubset { base_values, k } => {
                Some(RuntimeInnerExistsRegisterShape::KSubset {
                    base_values: base_values.clone(),
                    k: *k,
                })
            }
            // Scalar / Function / unresolved bases are not constant finite sets:
            // fail closed so the inner-EXISTS domain proof is rejected and the
            // action falls back to the interpreter.
            //
            // `TaggedScalarOrSetUnprovenRead` is a `func[key]` read whose key is
            // NOT statically proven to lie in the function domain: per its own
            // contract a mistyped-key read cannot be soundly fused, so it must
            // fail closed here too rather than emit a native-fused proof.
            RuntimeInnerExistsRegisterShape::Scalar { .. }
            | RuntimeInnerExistsRegisterShape::ScalarIntUpperBound { .. }
            | RuntimeInnerExistsRegisterShape::Function { .. }
            | RuntimeInnerExistsRegisterShape::TaggedScalarOrSetUnprovenRead { .. } => None,
        }
    }

    fn runtime_shape_from_var_layout(
        var_idx: u16,
        layout: &tla_jit_abi::VarLayout,
    ) -> Option<RuntimeInnerExistsRegisterShape> {
        match layout {
            tla_jit_abi::VarLayout::ScalarInt | tla_jit_abi::VarLayout::ScalarBool => {
                Some(RuntimeInnerExistsRegisterShape::Scalar { value: None })
            }
            tla_jit_abi::VarLayout::Compound(layout) => {
                Self::runtime_shape_from_compound_layout(layout, Some(var_idx))
            }
            _ => None,
        }
    }

    fn runtime_shape_from_var_layout_without_source(
        layout: &tla_jit_abi::VarLayout,
    ) -> Option<RuntimeInnerExistsRegisterShape> {
        match layout {
            tla_jit_abi::VarLayout::ScalarInt | tla_jit_abi::VarLayout::ScalarBool => {
                Some(RuntimeInnerExistsRegisterShape::Scalar { value: None })
            }
            tla_jit_abi::VarLayout::Compound(layout) => {
                Self::runtime_shape_from_compound_layout(layout, None)
            }
            _ => None,
        }
    }

    fn runtime_shape_from_compound_layout(
        layout: &tla_jit_abi::CompoundLayout,
        source_var_idx: Option<u16>,
    ) -> Option<RuntimeInnerExistsRegisterShape> {
        match layout {
            tla_jit_abi::CompoundLayout::Function {
                key_layout,
                value_layout,
                pair_count,
                domain_lo,
                ..
            } => Some(RuntimeInnerExistsRegisterShape::Function {
                source_var_idx,
                key_values: Self::runtime_scalar_values_from_function_key_layout(
                    key_layout,
                    *pair_count,
                    *domain_lo,
                ),
                value: Self::runtime_shape_from_compound_layout(value_layout, None).map(Box::new),
                // A plain function is not a sequence: no capacity bound.
                sequence_capacity_proof: None,
            }),
            tla_jit_abi::CompoundLayout::SetBitmask { universe, .. } => {
                let values = runtime_typed_scalar_values_from_bitmask_universe(universe)?;
                Some(RuntimeInnerExistsRegisterShape::SetBitmask { values })
            }
            tla_jit_abi::CompoundLayout::TaggedScalarOrSet {
                scalar_kind,
                set_universe,
                proof_source,
            } => {
                let values = runtime_typed_scalar_values_from_bitmask_universe(set_universe)?;
                if !runtime_scalar_elements_match_kind(&values, *scalar_kind) {
                    return None;
                }
                Some(RuntimeInnerExistsRegisterShape::TaggedScalarOrSet {
                    proof: RuntimeTaggedScalarOrSetTypeProof {
                        scalar_kind: *scalar_kind,
                        proof_source: *proof_source,
                        universe_values: values,
                    },
                })
            }
            tla_jit_abi::CompoundLayout::Sequence {
                element_layout,
                element_count: Some(element_count),
                capacity_proven,
            } => {
                if *element_count > tla_tir::bytecode::MAX_INNER_DOMAIN_SIZE {
                    return None;
                }
                let value =
                    Self::runtime_shape_from_compound_layout(element_layout, None).map(Box::new);
                let key_values = (1..=*element_count)
                    .map(|idx| {
                        i64::try_from(idx)
                            .ok()
                            .map(tla_jit_abi::SetBitmaskElement::Int)
                    })
                    .collect::<Option<Vec<_>>>()?;
                Some(RuntimeInnerExistsRegisterShape::Function {
                    source_var_idx,
                    key_values: Some(key_values),
                    value,
                    // Only a proven capacity is a sound upper bound on `Len`;
                    // an observed bound must not drive domain enumeration.
                    sequence_capacity_proof: capacity_proven.then_some(*element_count),
                })
            }
            tla_jit_abi::CompoundLayout::Int
            | tla_jit_abi::CompoundLayout::Bool
            | tla_jit_abi::CompoundLayout::String => {
                Some(RuntimeInnerExistsRegisterShape::Scalar { value: None })
            }
            _ => None,
        }
    }

    fn runtime_scalar_values_from_function_key_layout(
        key_layout: &tla_jit_abi::CompoundLayout,
        pair_count: Option<usize>,
        domain_lo: Option<i64>,
    ) -> Option<Vec<tla_jit_abi::SetBitmaskElement>> {
        match key_layout {
            tla_jit_abi::CompoundLayout::Int => {
                let pair_count = pair_count?;
                let lo = domain_lo?;
                if pair_count > tla_tir::bytecode::MAX_INNER_DOMAIN_SIZE {
                    return None;
                }
                if pair_count == 0 {
                    return Some(Vec::new());
                }
                let hi = lo.checked_add(i64::try_from(pair_count).ok()?.checked_sub(1)?)?;
                runtime_typed_scalar_values_from_expansion_domain(
                    &(lo..=hi)
                        .map(tla_jit_abi::SetBitmaskElement::Int)
                        .collect::<Vec<_>>(),
                )
            }
            tla_jit_abi::CompoundLayout::ExplicitScalarDomain { keys, .. } => {
                if domain_lo.is_some() {
                    return None;
                }
                if pair_count != Some(keys.len()) {
                    return None;
                }
                runtime_typed_scalar_values_from_expansion_domain(keys)
            }
            _ => None,
        }
    }

    fn runtime_shape_from_const_value(
        value: &tla_value::Value,
    ) -> Option<RuntimeInnerExistsRegisterShape> {
        if let Some(value) = runtime_scalar_element_from_const_value(value) {
            return Some(RuntimeInnerExistsRegisterShape::Scalar { value: Some(value) });
        }
        if let Some(values) = runtime_scalar_values_from_const_finite_set(value) {
            return Some(RuntimeInnerExistsRegisterShape::SetBitmask { values });
        }
        if let tla_value::Value::Subset(subset) = value {
            let base_values = runtime_scalar_values_from_const_finite_set(subset.base())?;
            return Some(RuntimeInnerExistsRegisterShape::Powerset { base_values });
        }
        if let tla_value::Value::KSubset(ksubset) = value {
            let base_values = runtime_scalar_values_from_const_finite_set(ksubset.base())?;
            return Some(RuntimeInnerExistsRegisterShape::KSubset {
                base_values,
                k: ksubset.k(),
            });
        }
        None
    }

    fn diagnose_inner_exists_expansion_failure(
        func: &tla_tir::bytecode::BytecodeFunction,
        state_layout: Option<&tla_jit_abi::StateLayout>,
    ) -> String {
        let Some(info) = Self::single_inner_exists_info(func) else {
            return "residual inner EXISTS expansion requires exactly one supported ExistsBegin/ExistsNext pair with valid loop offsets".to_string();
        };
        let producer = Self::describe_register_producer_for_diag(
            func,
            info.begin_pc,
            info.r_domain,
            state_layout,
            0,
        );
        format!(
            "residual inner EXISTS at pc={} has domain r{} that is not an enumerable proof-backed compact set domain; domain producer: {producer}; supported guarded expansion domains are constant finite sets, KSubset over exact finite/model-value domains, compact SetBitmask FuncApply values with exact universe metadata, fixed-capacity Sequence FuncApply values with compact element-domain metadata, TaggedScalarOrSet FuncApply values with an explicit finite set universe, or integer ranges `lo..hi` with a constant `lo` and an `hi` that is constant or `Len(seq)` over a proven-capacity sequence; scalar compact FuncApply domains require an action-local set-domain proof and fail closed when no proof-backed finite-set layout is available",
            info.begin_pc, info.r_domain,
        )
    }

    /// Classify whether an action that failed both inner-EXISTS expansion
    /// paths is a *runtime-domain multi-successor loop* ("NextStateLoop") — the
    /// shape the future multi-successor native ABI
    /// ([`tla_jit_abi::NextStateLoopFn`]) targets.
    ///
    /// Returns `Some(NotYetSupported)` when the action has exactly one inner
    /// existential whose domain register is produced by an integer `Range`
    /// whose bounds are *not* both compile-time scalars (so the static and
    /// runtime-guarded expansions both fail) — e.g.
    /// `\E k \in 1 .. natMin(primer, template) : ...`, where the upper bound is
    /// a runtime `Call`. This is precisely the case that hits the
    /// single-successor ABI ceiling: it cannot be compile-time unrolled, and a
    /// single `state_out` buffer cannot hold the one-successor-per-`k` result.
    ///
    /// Returning `Some(..)` is purely a *recognition* signal. The selection
    /// site still falls back to the interpreter — the variant
    /// [`tla_jit_abi::NextStateLoopSupport::NotYetSupported`] is the explicit,
    /// fail-closed gate. Flipping it to `Supported` (once trust-codegen emits a
    /// sound `NextStateLoopFn` for this shape) is the remaining work to make
    /// `anneal`-style actions run natively.
    fn classify_runtime_domain_next_state_loop(
        func: &tla_tir::bytecode::BytecodeFunction,
    ) -> Option<tla_jit_abi::NextStateLoopSupport> {
        // Must be exactly one well-formed inner existential.
        let info = Self::single_inner_exists_info(func)?;

        // Find the producer of the domain register, scanning backwards from the
        // ExistsBegin (mirrors `describe_register_producer_for_diag`). We chase
        // `Move` aliases so a `Range -> Move -> r_domain` chain is recognized.
        let mut domain_reg = info.r_domain;
        let mut producer: Option<tla_tir::bytecode::Opcode> = None;
        let mut guard = 0usize;
        'chase: while guard < 8 {
            guard += 1;
            let scan_end = info.begin_pc.min(func.instructions.len());
            let mut found = None;
            for pc in (0..scan_end).rev() {
                let op = func.instructions[pc];
                if op.dest_register() == Some(domain_reg) {
                    found = Some(op);
                    break;
                }
            }
            match found {
                Some(tla_tir::bytecode::Opcode::Move { rs, .. }) => {
                    domain_reg = rs;
                    continue 'chase;
                }
                Some(op) => {
                    producer = Some(op);
                    break 'chase;
                }
                None => break 'chase,
            }
        }

        // Only the integer `Range { lo, hi }` domain qualifies as the
        // NextStateLoop shape today. (Other runtime domains — e.g. set-valued
        // function applications — are deliberately *not* claimed here; they
        // keep their existing fail-closed diagnostic.)
        let tla_tir::bytecode::Opcode::Range { lo, hi, .. } = producer? else {
            return None;
        };

        // If BOTH bounds are compile-time immediates the static expansion would
        // have already handled it; this classifier is specifically for the
        // *runtime* range that the static/guarded paths cannot unroll.
        let scan_end = info.begin_pc.min(func.instructions.len());
        let is_compile_time_imm = |reg: u8| -> bool {
            for pc in (0..scan_end).rev() {
                let op = func.instructions[pc];
                if op.dest_register() != Some(reg) {
                    continue;
                }
                return matches!(op, tla_tir::bytecode::Opcode::LoadImm { .. });
            }
            false
        };
        if is_compile_time_imm(lo) && is_compile_time_imm(hi) {
            return None;
        }

        Some(tla_jit_abi::NextStateLoopSupport::NotYetSupported)
    }

    /// Classify an `\E m \in <state var> : ...` action whose domain state
    /// variable carries a *proven-closed* `RecordSetBitmask` compound layout as
    /// a native multi-successor ("NextStateLoop") record-set kernel target.
    ///
    /// This is the record-set sibling of
    /// [`Self::classify_runtime_domain_next_state_loop`] (which stays
    /// `NotYetSupported` for the integer `Range` shape). When the opt-in gate
    /// `TY_RECORD_SET_NATIVE=1` is set and the shape matches, it returns
    /// [`tla_jit_abi::NextStateLoopSupport::Supported`], signalling the planner
    /// to flag the action for the `lower_next_state_loop_scaffold` kernel and
    /// the sink-call BFS dispatch convention.
    ///
    /// Default (env unset) returns `None` so the single-successor dispatch is
    /// byte-for-byte unchanged.
    fn classify_record_set_next_state_loop(
        func: &tla_tir::bytecode::BytecodeFunction,
        state_layout: Option<&tla_jit_abi::StateLayout>,
    ) -> Option<tla_jit_abi::NextStateLoopSupport> {
        // Opt-in, default-off gate. With `TY_RECORD_SET_NATIVE` unset this
        // returns before touching anything else, so the whole record-set native
        // path is inert by default.
        if std::env::var_os("TY_RECORD_SET_NATIVE").as_deref() != Some(std::ffi::OsStr::new("1")) {
            return None;
        }
        let layout = state_layout?;

        // Exactly one well-formed inner existential (rejects multi-pair pairs
        // and result/binding/domain register aliasing).
        let info = Self::single_inner_exists_info(func)?;

        // Resolve the domain register to a terminal `LoadVar { var_idx }` of a
        // state variable, chasing `Move` aliases over the pre-`ExistsBegin`
        // prefix (the exact scan used by
        // `classify_runtime_domain_next_state_loop`). Fail closed on `LoadPrime`
        // or any non-`LoadVar` producer.
        let mut domain_reg = info.r_domain;
        let mut producer: Option<tla_tir::bytecode::Opcode> = None;
        let mut guard = 0usize;
        'chase: while guard < 8 {
            guard += 1;
            let scan_end = info.begin_pc.min(func.instructions.len());
            let mut found = None;
            for pc in (0..scan_end).rev() {
                let op = func.instructions[pc];
                if op.dest_register() == Some(domain_reg) {
                    found = Some(op);
                    break;
                }
            }
            match found {
                Some(tla_tir::bytecode::Opcode::Move { rs, .. }) => {
                    domain_reg = rs;
                    continue 'chase;
                }
                Some(op) => {
                    producer = Some(op);
                    break 'chase;
                }
                None => break 'chase,
            }
        }
        let tla_tir::bytecode::Opcode::LoadVar { var_idx, .. } = producer? else {
            return None;
        };

        // Reject the disjunctive dropped-sibling shape (`\/` short-circuit whose
        // skipped arm would produce a successor the expansion silently loses).
        if tla_tir::bytecode::static_expansion_drops_sibling_successor(
            func,
            std::slice::from_ref(&info),
        ) {
            return None;
        }

        // The domain state var must carry a proven-closed RecordSetBitmask
        // layout; only then is the per-bit native enumeration sound.
        match layout.var_layout(usize::from(var_idx)) {
            Some(tla_jit_abi::VarLayout::Compound(
                tla_jit_abi::CompoundLayout::RecordSetBitmask {
                    is_proven_closed: true,
                    ..
                },
            )) => Some(tla_jit_abi::NextStateLoopSupport::Supported),
            _ => None,
        }
    }

    fn describe_register_producer_for_diag(
        func: &tla_tir::bytecode::BytecodeFunction,
        before_pc: usize,
        reg: u8,
        state_layout: Option<&tla_jit_abi::StateLayout>,
        depth: usize,
    ) -> String {
        if depth >= 4 {
            return format!("r{reg} producer recursion limit reached");
        }
        let scan_end = before_pc.min(func.instructions.len());
        for pc in (0..scan_end).rev() {
            let op = func.instructions[pc];
            if op.dest_register() != Some(reg) {
                continue;
            }
            return Self::describe_register_write_for_diag(func, pc, op, state_layout, depth);
        }
        format!("r{reg} has no prior write before pc {scan_end}")
    }

    fn describe_register_write_for_diag(
        func: &tla_tir::bytecode::BytecodeFunction,
        pc: usize,
        op: tla_tir::bytecode::Opcode,
        state_layout: Option<&tla_jit_abi::StateLayout>,
        depth: usize,
    ) -> String {
        use tla_tir::bytecode::Opcode;

        match op {
            Opcode::LoadVar { rd, var_idx } | Opcode::LoadPrime { rd, var_idx } => {
                let layout = state_layout
                    .and_then(|layout| layout.var_layout(usize::from(var_idx)))
                    .map(Self::describe_var_layout_for_diag)
                    .unwrap_or_else(|| "layout=unavailable".to_string());
                let kind = if matches!(op, Opcode::LoadPrime { .. }) {
                    "LoadPrime"
                } else {
                    "LoadVar"
                };
                format!("{kind} pc={pc} r{rd} <- var[{var_idx}] {layout}")
            }
            Opcode::FuncApply { rd, func: rf, arg } => {
                let func_producer = Self::describe_register_producer_for_diag(
                    func,
                    pc,
                    rf,
                    state_layout,
                    depth + 1,
                );
                let arg_producer = Self::describe_register_producer_for_diag(
                    func,
                    pc,
                    arg,
                    state_layout,
                    depth + 1,
                );
                format!(
                    "FuncApply pc={pc} r{rd} <- r{rf}[r{arg}] (function: {func_producer}; arg: {arg_producer})"
                )
            }
            Opcode::Move { rd, rs } => {
                let source = Self::describe_register_producer_for_diag(
                    func,
                    pc,
                    rs,
                    state_layout,
                    depth + 1,
                );
                format!("Move pc={pc} r{rd} <- r{rs} ({source})")
            }
            Opcode::LoadImm { rd, value } => {
                format!("LoadImm pc={pc} r{rd} <- {value}")
            }
            Opcode::LoadBool { rd, value } => {
                format!("LoadBool pc={pc} r{rd} <- {value}")
            }
            Opcode::LoadConst { rd, idx } => {
                format!("LoadConst pc={pc} r{rd} <- const[{idx}]")
            }
            Opcode::SetEnum { rd, count, .. } => {
                format!("SetEnum pc={pc} r{rd} count={count}")
            }
            Opcode::SetDiff { rd, r1, r2 } => {
                format!("SetDiff pc={pc} r{rd} <- r{r1} \\ r{r2}")
            }
            Opcode::SetIntersect { rd, r1, r2 } => {
                format!("SetIntersect pc={pc} r{rd} <- r{r1} \\cap r{r2}")
            }
            Opcode::SetUnion { rd, r1, r2 } => {
                format!("SetUnion pc={pc} r{rd} <- r{r1} \\cup r{r2}")
            }
            Opcode::KSubset { rd, base, k } => {
                let base_producer = Self::describe_register_producer_for_diag(
                    func,
                    pc,
                    base,
                    state_layout,
                    depth + 1,
                );
                let k_producer =
                    Self::describe_register_producer_for_diag(func, pc, k, state_layout, depth + 1);
                format!(
                    "KSubset pc={pc} r{rd} <- KSubset(r{base}, r{k}) (base: {base_producer}; k: {k_producer})"
                )
            }
            other => format!("{other:?} at pc={pc}"),
        }
    }

    fn describe_var_layout_for_diag(layout: &tla_jit_abi::VarLayout) -> String {
        match layout {
            tla_jit_abi::VarLayout::ScalarInt => "layout=ScalarInt".to_string(),
            tla_jit_abi::VarLayout::ScalarBool => "layout=ScalarBool".to_string(),
            tla_jit_abi::VarLayout::Compound(compound) => {
                format!(
                    "layout=Compound({})",
                    Self::describe_compound_layout_for_diag(compound)
                )
            }
            other => format!("layout={other:?}"),
        }
    }

    fn describe_compound_layout_for_diag(layout: &tla_jit_abi::CompoundLayout) -> String {
        match layout {
            tla_jit_abi::CompoundLayout::Function {
                key_layout,
                value_layout,
                pair_count,
                domain_lo,
            } => format!(
                "Function(key_layout={}, value_layout={}, pair_count={pair_count:?}, domain_lo={domain_lo:?})",
                Self::describe_compound_layout_for_diag(key_layout),
                Self::describe_compound_layout_for_diag(value_layout),
            ),
            tla_jit_abi::CompoundLayout::ExplicitScalarDomain { key_layout, keys } => {
                format!(
                    "ExplicitScalarDomain(key_layout={}, keys={})",
                    Self::describe_compound_layout_for_diag(key_layout),
                    keys.len(),
                )
            }
            tla_jit_abi::CompoundLayout::SetBitmask { universe, .. } => {
                format!("SetBitmask(universe={})", universe.len())
            }
            tla_jit_abi::CompoundLayout::TaggedScalarOrSet {
                scalar_kind,
                set_universe,
                proof_source,
            } => format!(
                "TaggedScalarOrSet(scalar_kind={scalar_kind:?}, set_universe={}, proof_source={proof_source:?})",
                set_universe.len(),
            ),
            tla_jit_abi::CompoundLayout::Set {
                element_layout,
                element_count,
            } => format!(
                "Set(element_layout={}, element_count={element_count:?})",
                Self::describe_compound_layout_for_diag(element_layout),
            ),
            tla_jit_abi::CompoundLayout::Sequence {
                element_layout,
                element_count,
                capacity_proven,
            } => format!(
                "Sequence(element_layout={}, element_count={element_count:?}, capacity_proven={capacity_proven})",
                Self::describe_compound_layout_for_diag(element_layout),
            ),
            tla_jit_abi::CompoundLayout::Tuple { element_layouts } => {
                format!("Tuple(elements={})", element_layouts.len())
            }
            tla_jit_abi::CompoundLayout::Record { fields } => {
                format!("Record(fields={})", fields.len())
            }
            tla_jit_abi::CompoundLayout::Int => "Int".to_string(),
            tla_jit_abi::CompoundLayout::Bool => "Bool".to_string(),
            tla_jit_abi::CompoundLayout::String => "String".to_string(),
            other => format!("{other:?}"),
        }
    }

    /// Rewrite one action bytecode function into its per-witness specialized
    /// form for the runtime-guarded inner-EXISTS expansion.
    ///
    /// `gate_witness_participation` (WP-16, transitions accounting): when the
    /// action body can reach an enabling verdict WITHOUT executing the exists
    /// region at all (e.g. Bakery `e2`/`w1`: `IF unchecked # {} THEN \E i \in
    /// unchecked[self]: .. ELSE ..` — the ELSE arm is witness-independent),
    /// EVERY per-witness kernel re-emits that witness-independent successor,
    /// while the interpreter's enumeration emits it exactly once (the `\E`
    /// binding never happens on that path). Those duplicate emissions are
    /// byte-identical, so the successor SET (and thus the state count) is
    /// unaffected, but the fused-loop `generated` counter and per-action
    /// coverage counts drift above the interpreter's transition accounting
    /// (MCBakery: 11150 native vs 10658 interpreter).
    ///
    /// The gate makes the specialized kernels partition the interpreter's
    /// enumeration exactly: only the CANONICAL witness kernel (the first in
    /// sorted binding order, `gate_witness_participation == false`) may report
    /// the action enabled via a path that bypassed the exists region. Every
    /// NON-canonical kernel tracks a participation flag — initialized false at
    /// entry, set true only when its runtime witness-membership guard passes
    /// (control actually entered the exists body) — and ANDs it into the
    /// function's returned enabled verdict. Witness-SCOPED firings are
    /// unchanged (each witness kernel still emits its own, exactly like the
    /// interpreter's per-witness enumeration, including byte-identical
    /// collisions such as `\E i \in {1,2}: x' = 1` which the interpreter
    /// counts twice); witness-INDEPENDENT firings survive only in the
    /// canonical kernel. Soundness: participation can be false only on an
    /// evaluation path that never read the witness binding, and such a path is
    /// identical across all sibling kernels, so the canonical kernel emits the
    /// suppressed successor — the union of per-witness successor sets is
    /// unchanged.
    fn rewrite_runtime_guarded_inner_exists(
        func: &tla_tir::bytecode::BytecodeFunction,
        info: &tla_tir::bytecode::InnerExistsInfo,
        binding_value: RuntimeInnerExistsBindingValue,
        constants: &mut tla_tir::bytecode::ConstantPool,
        gate_witness_participation: bool,
    ) -> Option<tla_tir::bytecode::BytecodeFunction> {
        let guard_register = func.max_register.checked_add(1)?;
        let participation_register = if gate_witness_participation {
            Some(guard_register.checked_add(1)?)
        } else {
            None
        };
        // The participation gate rewrites the single terminal `Ret` into
        // `And; Ret`. The bytecode compiler emits exactly one terminal `Ret`
        // (result moved to r0 first); fail closed on any other shape so a
        // mid-function return can never bypass the gate.
        if gate_witness_participation {
            let terminal_ret_ok = matches!(
                func.instructions.last(),
                Some(tla_tir::bytecode::Opcode::Ret { .. })
            ) && func
                .instructions
                .iter()
                .filter(|op| matches!(op, tla_tir::bytecode::Opcode::Ret { .. }))
                .count()
                == 1;
            if !terminal_ret_ok {
                return None;
            }
        }
        let mut new_func = tla_tir::bytecode::BytecodeFunction::new(func.name.clone(), func.arity);
        new_func.max_register = participation_register.unwrap_or(guard_register);
        let after_next = info.next_pc.checked_add(1)?;
        // New-coordinate pc of the guard JumpFalse: the begin block starts at
        // pc_map(begin_pc) (identity ungated; +1 past the participation
        // prelude when gated) and JumpFalse is always its 4th op
        // (LoadBool rd, binding load, SetIn, JumpFalse).
        let begin_block_start = Self::runtime_guarded_pc_map_shifted(
            func.instructions.len(),
            info,
            info.begin_pc,
            gate_witness_participation,
        )?;
        let guard_jump_pc = begin_block_start.checked_add(3)?;
        let guard_target_pc = Self::runtime_guarded_pc_map_shifted(
            func.instructions.len(),
            info,
            after_next,
            gate_witness_participation,
        )?;
        let guard_offset = i32::try_from(
            i64::try_from(guard_target_pc).ok()? - i64::try_from(guard_jump_pc).ok()?,
        )
        .ok()?;

        if let Some(participation) = participation_register {
            new_func.emit(tla_tir::bytecode::Opcode::LoadBool {
                rd: participation,
                value: false,
            });
        }
        for (old_pc, op) in func.instructions.iter().copied().enumerate() {
            if old_pc == info.begin_pc {
                new_func.emit(tla_tir::bytecode::Opcode::LoadBool {
                    rd: info.rd,
                    value: false,
                });
                Self::emit_runtime_guarded_inner_binding_load(
                    &mut new_func,
                    constants,
                    info.r_binding,
                    &binding_value,
                );
                new_func.emit(tla_tir::bytecode::Opcode::SetIn {
                    rd: guard_register,
                    elem: info.r_binding,
                    set: info.r_domain,
                });
                new_func.emit(tla_tir::bytecode::Opcode::JumpFalse {
                    rs: guard_register,
                    offset: guard_offset,
                });
                if let Some(participation) = participation_register {
                    // Fall-through of the guard JumpFalse: the witness is in
                    // the runtime domain and the exists body executes.
                    new_func.emit(tla_tir::bytecode::Opcode::LoadBool {
                        rd: participation,
                        value: true,
                    });
                }
                continue;
            }
            if old_pc == info.next_pc {
                let tla_tir::bytecode::Opcode::ExistsNext { rd, r_body, .. } = op else {
                    return None;
                };
                new_func.emit(tla_tir::bytecode::Opcode::Move { rd, rs: r_body });
                continue;
            }
            if let tla_tir::bytecode::Opcode::LoadConst { rd, idx } = op {
                if rd == info.r_domain
                    && matches!(binding_value, RuntimeInnerExistsBindingValue::FiniteSet(_))
                {
                    if let Some(domain_value) =
                        Self::materialized_runtime_ksubset_domain_value(constants.get_value(idx))
                    {
                        let idx = constants.add_value(domain_value);
                        new_func.emit(tla_tir::bytecode::Opcode::LoadConst { rd, idx });
                        continue;
                    }
                }
            }
            if let tla_tir::bytecode::Opcode::KSubset { rd, base, .. } = op {
                if rd == info.r_domain
                    && matches!(binding_value, RuntimeInnerExistsBindingValue::FiniteSet(_))
                {
                    if !Self::runtime_guarded_ksubset_domain_can_widen_to_powerset(
                        func, info, old_pc,
                    ) {
                        return None;
                    }
                    new_func.emit(tla_tir::bytecode::Opcode::Powerset { rd, rs: base });
                    continue;
                }
            }
            if matches!(
                op,
                tla_tir::bytecode::Opcode::ExistsBegin { .. }
                    | tla_tir::bytecode::Opcode::ExistsNext { .. }
            ) {
                return None;
            }
            if let (Some(participation), tla_tir::bytecode::Opcode::Ret { rs }) =
                (participation_register, op)
            {
                // Terminal-Ret uniqueness was verified above; gate the enabled
                // verdict on witness participation. Jumps that targeted the
                // old terminal `Ret` land on the `And` (see the pc map), so
                // every return path is gated.
                debug_assert_eq!(old_pc, func.instructions.len() - 1);
                new_func.emit(tla_tir::bytecode::Opcode::And {
                    rd: rs,
                    r1: rs,
                    r2: participation,
                });
                new_func.emit(tla_tir::bytecode::Opcode::Ret { rs });
                continue;
            }
            let mapped = Self::remap_runtime_guarded_opcode(
                func.instructions.len(),
                info,
                old_pc,
                op,
                gate_witness_participation,
            )?;
            new_func.emit(mapped);
        }
        Some(new_func)
    }

    fn materialized_runtime_ksubset_domain_value(
        value: &tla_value::Value,
    ) -> Option<tla_value::Value> {
        let tla_value::Value::KSubset(ksubset) = value else {
            return None;
        };
        let base_values = runtime_scalar_values_from_const_finite_set(ksubset.base())?;
        let domain_values = runtime_ksubset_binding_values(&base_values, ksubset.k())?;
        Some(tla_value::Value::set(
            domain_values.iter().map(runtime_binding_value_literal),
        ))
    }

    fn runtime_guarded_ksubset_domain_can_widen_to_powerset(
        func: &tla_tir::bytecode::BytecodeFunction,
        info: &tla_tir::bytecode::InnerExistsInfo,
        producer_pc: usize,
    ) -> bool {
        let Some(tla_tir::bytecode::Opcode::KSubset { rd, .. }) =
            func.instructions.get(producer_pc)
        else {
            return false;
        };
        if *rd != info.r_domain || producer_pc >= info.begin_pc {
            return false;
        }

        func.instructions
            .iter()
            .enumerate()
            .skip(producer_pc + 1)
            .all(|(pc, op)| {
                pc == info.begin_pc || !Self::runtime_opcode_reads_register(op, info.r_domain)
            })
    }

    fn runtime_opcode_reads_register(op: &tla_tir::bytecode::Opcode, reg: u8) -> bool {
        use tla_tir::bytecode::Opcode;

        fn range_reads_register(start: u8, count: u8, reg: u8) -> bool {
            (0..count).any(|offset| start.checked_add(offset).is_some_and(|r| r == reg))
        }

        match op {
            Opcode::LoadImm { .. }
            | Opcode::LoadBool { .. }
            | Opcode::LoadConst { .. }
            | Opcode::LoadVar { .. }
            | Opcode::LoadPrime { .. }
            | Opcode::Jump { .. }
            | Opcode::SetPrimeMode { .. }
            | Opcode::Nop
            | Opcode::Halt
            | Opcode::Unchanged { .. } => false,
            Opcode::StoreVar { rs, .. }
            | Opcode::Move { rs, .. }
            | Opcode::NegInt { rs, .. }
            | Opcode::Not { rs, .. }
            | Opcode::Powerset { rs, .. }
            | Opcode::BigUnion { rs, .. }
            | Opcode::Domain { rs, .. }
            | Opcode::RecordGet { rs, .. }
            | Opcode::TupleGet { rs, .. }
            | Opcode::Tuple2SelfEq { value: rs, .. }
            | Opcode::Tuple2SelfSubseteq { value: rs, .. }
            | Opcode::Ret { rs }
            | Opcode::JumpTrue { rs, .. }
            | Opcode::JumpFalse { rs, .. } => *rs == reg,
            Opcode::AddInt { r1, r2, .. }
            | Opcode::SubInt { r1, r2, .. }
            | Opcode::MulInt { r1, r2, .. }
            | Opcode::DivInt { r1, r2, .. }
            | Opcode::IntDiv { r1, r2, .. }
            | Opcode::ModInt { r1, r2, .. }
            | Opcode::PowInt { r1, r2, .. }
            | Opcode::Eq { r1, r2, .. }
            | Opcode::Neq { r1, r2, .. }
            | Opcode::LtInt { r1, r2, .. }
            | Opcode::LeInt { r1, r2, .. }
            | Opcode::GtInt { r1, r2, .. }
            | Opcode::GeInt { r1, r2, .. }
            | Opcode::And { r1, r2, .. }
            | Opcode::Or { r1, r2, .. }
            | Opcode::Implies { r1, r2, .. }
            | Opcode::Equiv { r1, r2, .. }
            | Opcode::SetUnion { r1, r2, .. }
            | Opcode::SetIntersect { r1, r2, .. }
            | Opcode::SetDiff { r1, r2, .. }
            | Opcode::Subseteq { r1, r2, .. }
            | Opcode::StrConcat { r1, r2, .. }
            | Opcode::Concat { r1, r2, .. } => *r1 == reg || *r2 == reg,
            Opcode::Range { lo, hi, .. } => *lo == reg || *hi == reg,
            Opcode::KSubset { base, k, .. } => *base == reg || *k == reg,
            Opcode::SetIn { elem, set, .. } => *elem == reg || *set == reg,
            Opcode::Tuple2SetIn {
                first, second, set, ..
            } => *first == reg || *second == reg || *set == reg,
            Opcode::SetEnumSubseteq {
                start, count, set, ..
            } => *set == reg || (0..*count).any(|offset| start.saturating_add(offset) == reg),
            Opcode::RoundStepEq { child, parent, .. } => *child == reg || *parent == reg,
            Opcode::FuncApply { func, arg, .. } => *func == reg || *arg == reg,
            Opcode::FuncSet { domain, range, .. } => *domain == reg || *range == reg,
            Opcode::FuncExcept {
                func, path, val, ..
            } => *func == reg || *path == reg || *val == reg,
            // Fused Eq superinstructions (implied-action eval-VM compile
            // only; never present in action bytecode this analysis sees).
            Opcode::EqFuncExcept {
                lhs,
                func,
                path,
                val,
                ..
            } => *lhs == reg || *func == reg || *path == reg || *val == reg,
            Opcode::EqRecordNew {
                lhs,
                values_start,
                count,
                ..
            } => *lhs == reg || range_reads_register(*values_start, *count, reg),
            Opcode::CondMove { cond, rs, .. } => *cond == reg || *rs == reg,
            Opcode::SetEnum { start, count, .. }
            | Opcode::TupleNew { start, count, .. }
            | Opcode::SeqNew { start, count, .. }
            | Opcode::Times { start, count, .. } => range_reads_register(*start, *count, reg),
            Opcode::RecordNew {
                values_start,
                count,
                ..
            }
            | Opcode::RecordSet {
                values_start,
                count,
                ..
            } => range_reads_register(*values_start, *count, reg),
            Opcode::FuncDef {
                r_domain,
                r_binding,
                ..
            } => *r_domain == reg || *r_binding == reg,
            Opcode::Call {
                args_start, argc, ..
            }
            | Opcode::CallExternal {
                args_start, argc, ..
            }
            | Opcode::CallBuiltin {
                args_start, argc, ..
            } => range_reads_register(*args_start, *argc, reg),
            Opcode::ValueApply {
                func,
                args_start,
                argc,
                ..
            } => *func == reg || range_reads_register(*args_start, *argc, reg),
            Opcode::MakeClosure {
                captures_start,
                capture_count,
                ..
            } => range_reads_register(*captures_start, *capture_count, reg),
            Opcode::ForallBegin {
                r_binding,
                r_domain,
                ..
            }
            | Opcode::ExistsBegin {
                r_binding,
                r_domain,
                ..
            }
            | Opcode::ChooseBegin {
                r_binding,
                r_domain,
                ..
            }
            | Opcode::SetFilterBegin {
                r_binding,
                r_domain,
                ..
            }
            | Opcode::SetBuilderBegin {
                r_binding,
                r_domain,
                ..
            }
            | Opcode::FuncDefBegin {
                r_binding,
                r_domain,
                ..
            } => *r_binding == reg || *r_domain == reg,
            Opcode::ForallNext {
                r_binding, r_body, ..
            }
            | Opcode::ExistsNext {
                r_binding, r_body, ..
            }
            | Opcode::ChooseNext {
                r_binding, r_body, ..
            }
            | Opcode::LoopNext {
                r_binding, r_body, ..
            } => *r_binding == reg || *r_body == reg,
        }
    }

    fn emit_runtime_guarded_inner_binding_load(
        func: &mut tla_tir::bytecode::BytecodeFunction,
        constants: &mut tla_tir::bytecode::ConstantPool,
        rd: u8,
        binding_value: &RuntimeInnerExistsBindingValue,
    ) {
        match binding_value {
            RuntimeInnerExistsBindingValue::Scalar(tla_jit_abi::SetBitmaskElement::Int(value)) => {
                func.emit(tla_tir::bytecode::Opcode::LoadImm { rd, value: *value });
            }
            RuntimeInnerExistsBindingValue::Scalar(tla_jit_abi::SetBitmaskElement::Bool(value)) => {
                func.emit(tla_tir::bytecode::Opcode::LoadBool { rd, value: *value });
            }
            RuntimeInnerExistsBindingValue::Scalar(tla_jit_abi::SetBitmaskElement::String(
                name,
            )) => {
                let idx =
                    constants.add_value(tla_value::Value::String(tla_core::resolve_name_id(*name).into()));
                func.emit(tla_tir::bytecode::Opcode::LoadConst { rd, idx });
            }
            RuntimeInnerExistsBindingValue::Scalar(tla_jit_abi::SetBitmaskElement::ModelValue(
                name,
            )) => {
                let idx = constants.add_value(tla_value::Value::ModelValue(
                    tla_core::resolve_name_id(*name).into(),
                ));
                func.emit(tla_tir::bytecode::Opcode::LoadConst { rd, idx });
            }
            RuntimeInnerExistsBindingValue::FiniteSet(elements) => {
                let idx = constants.add_value(tla_jit_abi::set_bitmask_elements_to_value(elements));
                func.emit(tla_tir::bytecode::Opcode::LoadConst { rd, idx });
            }
        }
    }

    fn remap_runtime_guarded_opcode(
        instruction_len: usize,
        info: &tla_tir::bytecode::InnerExistsInfo,
        old_pc: usize,
        op: tla_tir::bytecode::Opcode,
        gate_witness_participation: bool,
    ) -> Option<tla_tir::bytecode::Opcode> {
        use tla_tir::bytecode::Opcode;

        let map_offset = |offset| {
            Self::remap_runtime_guarded_offset(
                instruction_len,
                info,
                old_pc,
                offset,
                gate_witness_participation,
            )
        };
        Some(match op {
            Opcode::Jump { offset } => Opcode::Jump {
                offset: map_offset(offset)?,
            },
            Opcode::JumpTrue { rs, offset } => Opcode::JumpTrue {
                rs,
                offset: map_offset(offset)?,
            },
            Opcode::JumpFalse { rs, offset } => Opcode::JumpFalse {
                rs,
                offset: map_offset(offset)?,
            },
            Opcode::ForallBegin {
                rd,
                r_binding,
                r_domain,
                loop_end,
            } => Opcode::ForallBegin {
                rd,
                r_binding,
                r_domain,
                loop_end: map_offset(loop_end)?,
            },
            Opcode::ForallNext {
                rd,
                r_binding,
                r_body,
                loop_begin,
            } => Opcode::ForallNext {
                rd,
                r_binding,
                r_body,
                loop_begin: map_offset(loop_begin)?,
            },
            Opcode::ChooseBegin {
                rd,
                r_binding,
                r_domain,
                loop_end,
            } => Opcode::ChooseBegin {
                rd,
                r_binding,
                r_domain,
                loop_end: map_offset(loop_end)?,
            },
            Opcode::ChooseNext {
                rd,
                r_binding,
                r_body,
                loop_begin,
            } => Opcode::ChooseNext {
                rd,
                r_binding,
                r_body,
                loop_begin: map_offset(loop_begin)?,
            },
            Opcode::SetBuilderBegin {
                rd,
                r_binding,
                r_domain,
                loop_end,
            } => Opcode::SetBuilderBegin {
                rd,
                r_binding,
                r_domain,
                loop_end: map_offset(loop_end)?,
            },
            Opcode::SetFilterBegin {
                rd,
                r_binding,
                r_domain,
                loop_end,
            } => Opcode::SetFilterBegin {
                rd,
                r_binding,
                r_domain,
                loop_end: map_offset(loop_end)?,
            },
            Opcode::FuncDefBegin {
                rd,
                r_binding,
                r_domain,
                loop_end,
            } => Opcode::FuncDefBegin {
                rd,
                r_binding,
                r_domain,
                loop_end: map_offset(loop_end)?,
            },
            Opcode::LoopNext {
                r_binding,
                r_body,
                loop_begin,
            } => Opcode::LoopNext {
                r_binding,
                r_body,
                loop_begin: map_offset(loop_begin)?,
            },
            other => other,
        })
    }

    fn remap_runtime_guarded_offset(
        instruction_len: usize,
        info: &tla_tir::bytecode::InnerExistsInfo,
        old_pc: usize,
        old_offset: i32,
        gate_witness_participation: bool,
    ) -> Option<i32> {
        let old_target = Self::runtime_exists_target_pc(instruction_len, old_pc, old_offset)?;
        let new_pc = Self::runtime_guarded_pc_map_shifted(
            instruction_len,
            info,
            old_pc,
            gate_witness_participation,
        )?;
        let new_target = Self::runtime_guarded_pc_map_shifted(
            instruction_len,
            info,
            old_target,
            gate_witness_participation,
        )?;
        i32::try_from(i64::try_from(new_target).ok()? - i64::try_from(new_pc).ok()?).ok()
    }

    fn runtime_guarded_pc_map(
        instruction_len: usize,
        info: &tla_tir::bytecode::InnerExistsInfo,
        old_pc: usize,
    ) -> Option<usize> {
        Self::runtime_guarded_pc_map_shifted(instruction_len, info, old_pc, false)
    }

    /// Old-pc -> new-pc map for the runtime-guarded rewrite.
    ///
    /// Ungated layout: the 1-op `ExistsBegin` becomes a 4-op guard block, so
    /// pcs after `begin_pc` shift by +3. Gated layout (WP-16 witness
    /// participation, non-canonical kernels): a 1-op prelude precedes the
    /// function (+1 everywhere) and the guard block grows to 5 ops (+4 after
    /// `begin_pc`); the terminal `Ret` additionally becomes `And; Ret`, so the
    /// one-past-end pc shifts by one more (+6) — a jump that targets the
    /// terminal `Ret` itself lands on the `And`, which is exactly the gated
    /// return sequence.
    fn runtime_guarded_pc_map_shifted(
        instruction_len: usize,
        info: &tla_tir::bytecode::InnerExistsInfo,
        old_pc: usize,
        gate_witness_participation: bool,
    ) -> Option<usize> {
        if old_pc > instruction_len {
            return None;
        }
        if !gate_witness_participation {
            if old_pc <= info.begin_pc {
                return Some(old_pc);
            }
            return old_pc.checked_add(3);
        }
        if old_pc <= info.begin_pc {
            return old_pc.checked_add(1);
        }
        if old_pc == instruction_len {
            return old_pc.checked_add(6);
        }
        old_pc.checked_add(5)
    }

    /// Declared per-action state-variable footprint from bytecode.
    ///
    /// Scans the entry function AND every chunk callee transitively reachable
    /// through `Call` (item 4 M0-G4): callees receive `state_in_ptr` (and the
    /// entry's `state_out`) and can `LoadVar`/`StoreVar`, so an entry-only scan
    /// under-reports for helper-calling actions. The result feeds
    /// `TrustCgNativeActionEntry::{read_vars,write_vars}` /
    /// `tla_jit_abi::ActionDescriptor`, and the hybrid per-action admission
    /// dual-gates it against ty's AST footprint (any mismatch = decline).
    ///
    /// A `Call` to an unresolvable callee (no chunk / out-of-range `op_idx`)
    /// contributes nothing here; such an action cannot lower to native either
    /// (the chunk-aware lowering fails), so it never reaches dispatch with an
    /// incomplete declared footprint.
    fn action_var_access_sets(
        func: &tla_tir::bytecode::BytecodeFunction,
        chunk: Option<&tla_tir::bytecode::BytecodeChunk>,
    ) -> (Vec<u16>, Vec<u16>) {
        let mut read_vars = FxHashSet::default();
        let mut write_vars = FxHashSet::default();
        let mut visited_callees: FxHashSet<u16> = FxHashSet::default();
        let mut worklist: Vec<&tla_tir::bytecode::BytecodeFunction> = vec![func];
        while let Some(current) = worklist.pop() {
            for op in &current.instructions {
                match *op {
                    tla_tir::bytecode::Opcode::LoadVar { var_idx, .. } => {
                        read_vars.insert(var_idx);
                    }
                    tla_tir::bytecode::Opcode::StoreVar { var_idx, .. } => {
                        write_vars.insert(var_idx);
                    }
                    tla_tir::bytecode::Opcode::Call { op_idx, .. } => {
                        if let Some(chunk) = chunk {
                            if visited_callees.insert(op_idx) {
                                if let Some(callee) = chunk.functions.get(usize::from(op_idx)) {
                                    worklist.push(callee);
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        let mut read_vars: Vec<u16> = read_vars.into_iter().collect();
        let mut write_vars: Vec<u16> = write_vars.into_iter().collect();
        read_vars.sort_unstable();
        write_vars.sort_unstable();
        (read_vars, write_vars)
    }

    #[allow(clippy::too_many_arguments)]
    fn plan_next_state_action_exact(
        action_name: &str,
        func: &tla_tir::bytecode::BytecodeFunction,
        state_layout: Option<Arc<tla_jit_abi::StateLayout>>,
        opt_level: tla_trust_cg::OptLevel,
        const_pool: Option<Arc<tla_tir::bytecode::ConstantPool>>,
        chunk: Option<Arc<tla_tir::bytecode::BytecodeChunk>>,
        chunk_callee_shapes: Option<&tla_ir::lower::ChunkCalleeReturnShapes>,
        action_local_set_domain_proof: Option<tla_ir::lower::ActionLocalSetDomainProof>,
        action_compile_tasks: &mut Vec<TrustCgActionCompileTask>,
        binding_values: &[i64],
        formal_values: &[i64],
    ) {
        // Plan-time recursive-Sum-fold scalarization (Piece A): rewrite any
        // helper callee in the chunk whose body is a proven-injective
        // constant-cardinality set-builder folded by a comm-assoc recursive
        // `Sum` into an unrolled straight-line sum (GameOfLife `score`). This
        // rewrites ONLY the chunk that feeds this native compile task — a fresh
        // clone — so the interpreter oracle (and the compiled-BFS interpreter
        // crosscheck, which re-evaluates the ORIGINAL AST) is unaffected and
        // genuinely validates the rewrite. Fail-closed: returns `None` (no
        // clone) when nothing matches, leaving the action untouched.
        let chunk = match chunk {
            Some(chunk_arc) => {
                match sum_fold_scalarize::rewrite_chunk_injective_sum_folds(&chunk_arc) {
                    Some(rewritten) => Some(Arc::new(rewritten)),
                    None => Some(chunk_arc),
                }
            }
            None => None,
        };
        let (read_vars, write_vars) = Self::action_var_access_sets(func, chunk.as_deref());
        // Item 4 M1: the declared compound-read footprint. Recomputing it here
        // from the same analysis the lowering runs — rather than plumbing it
        // back out of the lowered module — is what keeps ty's admission gate
        // and the emitted code in lockstep: a var can only be declared if the
        // pre-scan admitted it, and the lowering emits a callout for exactly
        // the vars the pre-scan admitted.
        let compound_read_vars = tla_ir::lower::compound_read_callout_vars(
            &func.instructions,
            state_layout.as_deref(),
            const_pool.as_deref(),
        );
        action_compile_tasks.push(TrustCgActionCompileTask {
            action_name: action_name.to_string(),
            func: func.clone(),
            state_layout,
            opt_level,
            const_pool,
            chunk,
            chunk_callee_shapes: chunk_callee_shapes.cloned(),
            action_local_set_domain_proof,
            binding_values: binding_values.to_vec(),
            formal_values: formal_values.to_vec(),
            read_vars,
            write_vars,
            compound_read_vars,
            next_state_loop: false,
        });
    }

    /// Wall 3: sound top-level action-disjunction split.
    ///
    /// When an action `guard /\ (D1 \/ ... \/ Dn)` carries two or more inner
    /// `\E` pairs (so the single-pair expansion fails and the sibling-drop guard
    /// fails closed), split it into per-disjunct sub-actions `guard /\ Di`. Each
    /// sub-action has at most one inner `\E` pair and is planned via the normal
    /// `plan_next_state_action_entry`. The BFS engine already unions the
    /// successors of distinct native action functions, and the split is
    /// successor-exact (see `tla_tir::bytecode::split_top_level_disjunction`), so
    /// the union of the sub-actions equals the original action exactly.
    ///
    /// Returns `true` iff EVERY sub-action lowered to native AND each is
    /// native-fused-safe (so the combined action can join the compiled BFS
    /// loop). On any sub-action failure this rolls back all sub-action planning
    /// and returns `false`, leaving the action on its existing fail-closed path
    /// (interpreter). Rolling back keeps the all-or-nothing coverage invariant.
    #[allow(clippy::too_many_arguments)]
    /// Register the union of run_prepare-level disjunction-split arms
    /// (`X#d0..X#dn` bytecode entries) under their base action name `X`.
    ///
    /// SOUNDNESS: `run_prepare`'s split is union-exact (see
    /// `tla_tir::bytecode::disjunction_split` module docs): the successor set
    /// of `X` equals the union of the arms' successor sets, and the BFS
    /// dispatch invokes EVERY expansion key of a base action and unions the
    /// results. Registration fails closed unless (a) the base has no own
    /// bytecode entry, (b) the arm indices form the complete contiguous group
    /// `0..n` (run_prepare registers arms all-or-nothing, so a partial group
    /// means name collision — reject), and (c) every arm either produced a
    /// native compile task or was itself expanded with a fused-safe proof.
    /// Arms that fail native COMPILATION later are caught by the existing
    /// `contains_action` checks in coverage/eligibility/dispatch (dispatch
    /// returns interpreter fallback for the whole parent state, dropping no
    /// successors).
    fn register_run_prepare_split_arm_unions(
        action_bytecodes: &FxHashMap<String, &tla_tir::bytecode::BytecodeFunction>,
        action_compile_tasks: &[TrustCgActionCompileTask],
        inner_exists_expansion_keys: &mut FxHashMap<String, Vec<String>>,
        inner_exists_expansion_proofs: &mut FxHashMap<String, TrustCgInnerExistsExpansionProof>,
    ) {
        let mut arm_groups: FxHashMap<String, Vec<(usize, String)>> = FxHashMap::default();
        for action_name in action_bytecodes.keys() {
            let Some(pos) = action_name.rfind("#d") else {
                continue;
            };
            let digits = &action_name[pos + 2..];
            if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_digit()) {
                continue;
            }
            let base = &action_name[..pos];
            if base.is_empty() || action_bytecodes.contains_key(base) {
                continue;
            }
            let Ok(idx) = digits.parse::<usize>() else {
                continue;
            };
            arm_groups
                .entry(base.to_string())
                .or_default()
                .push((idx, action_name.clone()));
        }

        let mut bases: Vec<String> = arm_groups.keys().cloned().collect();
        bases.sort();
        for base in bases {
            let mut arms = arm_groups.remove(&base).unwrap_or_default();
            if inner_exists_expansion_keys.contains_key(&base) {
                continue;
            }
            arms.sort();
            // Complete contiguous 0..n group only.
            if arms.len() < 2 || !arms.iter().enumerate().all(|(i, (k, _))| *k == i) {
                continue;
            }

            let mut final_keys: Vec<String> = Vec::new();
            let mut all_planned = true;
            for (_, arm_key) in &arms {
                match inner_exists_expansion_keys.get(arm_key) {
                    Some(keys) if !keys.is_empty() => {
                        let safe = inner_exists_expansion_proofs
                            .get(arm_key)
                            .is_some_and(|proof| proof.native_fused_safe(keys.len()));
                        if !safe {
                            all_planned = false;
                            break;
                        }
                        final_keys.extend(keys.iter().cloned());
                    }
                    _ => {
                        if !action_compile_tasks
                            .iter()
                            .any(|task| &task.action_name == arm_key)
                        {
                            all_planned = false;
                            break;
                        }
                        final_keys.push(arm_key.clone());
                    }
                }
            }
            if !all_planned {
                continue;
            }
            // Dispatch keys must be unique so each resolves to a distinct
            // native fn.
            let mut seen = FxHashSet::default();
            if !final_keys.iter().all(|key| seen.insert(key.clone())) {
                continue;
            }

            // Per-arm expansion entries become internal scaffolding; the base
            // entry is what per-instance dispatch looks up.
            for (_, arm_key) in &arms {
                inner_exists_expansion_keys.remove(arm_key);
                inner_exists_expansion_proofs.remove(arm_key);
            }
            let total_keys = final_keys.len();
            inner_exists_expansion_keys.insert(base.clone(), final_keys);
            inner_exists_expansion_proofs.insert(
                base.clone(),
                TrustCgInnerExistsExpansionProof {
                    expansion_count: total_keys,
                    kind: TrustCgInnerExistsExpansionProofKind::SplitDisjunction { total_keys },
                },
            );
            eprintln!(
                "[trust-cg] action '{base}': {} bytecode disjunction arm(s) registered as {total_keys} native action key(s) under the base action",
                arms.len(),
            );
        }
    }

    fn try_plan_split_disjunction_action_entry(
        action_name: &str,
        func: &tla_tir::bytecode::BytecodeFunction,
        state_layout: Option<Arc<tla_jit_abi::StateLayout>>,
        opt_level: tla_trust_cg::OptLevel,
        const_pool: Option<Arc<tla_tir::bytecode::ConstantPool>>,
        chunk: Option<Arc<tla_tir::bytecode::BytecodeChunk>>,
        chunk_callee_shapes: Option<&tla_ir::lower::ChunkCalleeReturnShapes>,
        action_compile_tasks: &mut Vec<TrustCgActionCompileTask>,
        inner_exists_expansion_keys: &mut FxHashMap<String, Vec<String>>,
        inner_exists_expansion_proofs: &mut FxHashMap<String, TrustCgInnerExistsExpansionProof>,
        stats: &mut TrustCgBuildStats,
        binding_values: &[i64],
        formal_values: &[i64],
        scalarize_env: Option<&RecordSetScalarizeEnv<'_>>,
    ) -> bool {
        let Some(subs) = tla_tir::bytecode::split_top_level_disjunction(func) else {
            return false;
        };
        if subs.is_empty() {
            return false;
        }

        // Snapshot rollback points so a partial failure leaves NO trace.
        let tasks_checkpoint = action_compile_tasks.len();
        let stats_checkpoint = (stats.actions_compiled, stats.actions_failed);
        let mut sub_keys_registered: Vec<String> = Vec::new();

        let rollback = |action_compile_tasks: &mut Vec<TrustCgActionCompileTask>,
                        inner_exists_expansion_keys: &mut FxHashMap<String, Vec<String>>,
                        inner_exists_expansion_proofs: &mut FxHashMap<
            String,
            TrustCgInnerExistsExpansionProof,
        >,
                        stats: &mut TrustCgBuildStats,
                        sub_keys_registered: &[String]| {
            action_compile_tasks.truncate(tasks_checkpoint);
            stats.actions_compiled = stats_checkpoint.0;
            stats.actions_failed = stats_checkpoint.1;
            for sub_name in sub_keys_registered {
                inner_exists_expansion_keys.remove(sub_name);
                inner_exists_expansion_proofs.remove(sub_name);
            }
        };

        let mut final_keys: Vec<String> = Vec::new();
        for sub in &subs {
            let tasks_before = action_compile_tasks.len();
            // Each sub-action is planned independently. It may compile directly
            // (no residual `\E`) or be inner-EXISTS-expanded into its own keys.
            Self::plan_next_state_action_entry(
                &sub.name,
                sub,
                state_layout.clone(),
                opt_level,
                const_pool.clone(),
                chunk.clone(),
                chunk_callee_shapes,
                action_compile_tasks,
                inner_exists_expansion_keys,
                inner_exists_expansion_proofs,
                stats,
                binding_values,
                formal_values,
                scalarize_env,
            );
            sub_keys_registered.push(sub.name.clone());

            // A sub-action that produced NO compile task failed to plan -> the
            // whole split must fail closed (else we drop that disjunct's
            // successors).
            if action_compile_tasks.len() == tasks_before {
                rollback(
                    action_compile_tasks,
                    inner_exists_expansion_keys,
                    inner_exists_expansion_proofs,
                    stats,
                    &sub_keys_registered,
                );
                return false;
            }

            // Resolve the sub-action's final native keys. If it was
            // inner-EXISTS-expanded, those are its expansion keys (and they must
            // be native-fused-safe for the combined action to join the fused
            // loop). Otherwise it compiled directly and its own name is the key.
            let sub_keys = inner_exists_expansion_keys.get(&sub.name).cloned();
            match sub_keys {
                Some(keys) if !keys.is_empty() => {
                    let safe = inner_exists_expansion_proofs
                        .get(&sub.name)
                        .is_some_and(|proof| proof.native_fused_safe(keys.len()));
                    if !safe {
                        rollback(
                            action_compile_tasks,
                            inner_exists_expansion_keys,
                            inner_exists_expansion_proofs,
                            stats,
                            &sub_keys_registered,
                        );
                        return false;
                    }
                    final_keys.extend(keys);
                }
                _ => {
                    final_keys.push(sub.name.clone());
                }
            }
        }

        // Dispatch keys must be unique so each resolves to a distinct native fn.
        let mut seen = FxHashSet::default();
        if !final_keys.iter().all(|key| seen.insert(key.clone())) {
            rollback(
                action_compile_tasks,
                inner_exists_expansion_keys,
                inner_exists_expansion_proofs,
                stats,
                &sub_keys_registered,
            );
            return false;
        }

        // The sub-action expansion-key entries are internal scaffolding; the
        // base action's own entry is what the BFS dispatch looks up. Remove the
        // per-sub entries and install the union under `action_name`.
        for sub_name in &sub_keys_registered {
            inner_exists_expansion_keys.remove(sub_name);
            inner_exists_expansion_proofs.remove(sub_name);
        }
        let total_keys = final_keys.len();
        inner_exists_expansion_keys.insert(action_name.to_string(), final_keys);
        inner_exists_expansion_proofs.insert(
            action_name.to_string(),
            TrustCgInnerExistsExpansionProof {
                expansion_count: total_keys,
                kind: TrustCgInnerExistsExpansionProofKind::SplitDisjunction { total_keys },
            },
        );

        eprintln!(
            "[trust-cg] action '{action_name}': top-level disjunction split into {} sub-action(s) -> {total_keys} native action key(s)",
            subs.len(),
        );
        true
    }

    /// Register the per-witness native keys for a record-set aggregate
    /// scalarization (Route C) and push their single-successor compile tasks.
    ///
    /// Mirrors the runtime-guarded inner-EXISTS expansion registration: one
    /// native key per witness under the base action, plus a
    /// `RuntimeGuardedFiniteDomain` proof whose distinct binding values gate
    /// fused-level admission. Returns `false` (registering nothing) when any
    /// witness has no key/proof encoding — the caller then falls through to
    /// the existing fail-closed diagnostics.
    #[allow(clippy::too_many_arguments)]
    fn plan_record_set_scalarized_expansions(
        action_name: &str,
        outcome: record_set_scalarize::ScalarizeOutcome,
        state_layout: Option<Arc<tla_jit_abi::StateLayout>>,
        opt_level: tla_trust_cg::OptLevel,
        chunk: Option<Arc<tla_tir::bytecode::BytecodeChunk>>,
        chunk_callee_shapes: Option<&tla_ir::lower::ChunkCalleeReturnShapes>,
        action_compile_tasks: &mut Vec<TrustCgActionCompileTask>,
        inner_exists_expansion_keys: &mut FxHashMap<String, Vec<String>>,
        inner_exists_expansion_proofs: &mut FxHashMap<String, TrustCgInnerExistsExpansionProof>,
        binding_values: &[i64],
        formal_values: &[i64],
    ) -> bool {
        let record_set_scalarize::ScalarizeOutcome { expansions, pool } = outcome;
        if expansions.is_empty() {
            return false;
        }
        // Per-witness dispatch keys + proof binding values (both fail-closed).
        let mut keys: Vec<String> = Vec::with_capacity(expansions.len());
        let mut proof_values: Vec<RuntimeInnerExistsBindingValue> =
            Vec::with_capacity(expansions.len());
        let mut seen = FxHashSet::default();
        for expansion in &expansions {
            let Some(key) = tla_jit_abi::binding_key_for_values(
                action_name,
                std::slice::from_ref(&expansion.witness),
            ) else {
                return false;
            };
            if !seen.insert(key.clone()) {
                return false;
            }
            keys.push(key);
            let value = if let Some(element) =
                runtime_scalar_element_from_const_value(&expansion.witness)
            {
                RuntimeInnerExistsBindingValue::Scalar(element)
            } else if let Some(elements) =
                runtime_scalar_values_from_const_finite_set(&expansion.witness)
            {
                RuntimeInnerExistsBindingValue::FiniteSet(elements)
            } else {
                return false;
            };
            proof_values.push(value);
        }
        {
            let mut distinct = FxHashSet::default();
            if !proof_values.iter().all(|value| distinct.insert(value)) {
                return false;
            }
        }

        // All witnesses share one appended pool (append-only, so the source
        // chunk's precomputed callee shapes stay exact); rebuild the chunk to
        // carry it, mirroring the runtime-guarded expansion path.
        let compile_const_pool = Some(Arc::new(pool));
        let compile_chunk = match (chunk.as_deref(), compile_const_pool.as_deref()) {
            (Some(source_chunk), Some(pool)) => {
                let mut chunk = source_chunk.clone();
                chunk.constants = pool.clone();
                Some(Arc::new(chunk))
            }
            _ => chunk.clone(),
        };

        inner_exists_expansion_keys.insert(action_name.to_string(), keys.clone());
        inner_exists_expansion_proofs.insert(
            action_name.to_string(),
            TrustCgInnerExistsExpansionProof {
                expansion_count: keys.len(),
                kind: TrustCgInnerExistsExpansionProofKind::RuntimeGuardedFiniteDomain {
                    binding_values: proof_values,
                },
            },
        );
        eprintln!(
            "[trust-cg] action '{action_name}': record-set aggregate scalarization expanded into {} native action function(s) (TY_RECORD_SET_NATIVE=1)",
            keys.len(),
        );
        for (key, expansion) in keys.iter().zip(expansions) {
            debug_assert!(
                !expansion.func.instructions.iter().any(|op| matches!(
                    op,
                    tla_tir::bytecode::Opcode::ExistsBegin { .. }
                        | tla_tir::bytecode::Opcode::ExistsNext { .. }
                )),
                "scalarized expansion must be EXISTS-free"
            );
            Self::plan_next_state_action_exact(
                key,
                &expansion.func,
                state_layout.clone(),
                opt_level,
                compile_const_pool.clone(),
                compile_chunk.clone(),
                chunk_callee_shapes,
                None,
                action_compile_tasks,
                binding_values,
                formal_values,
            );
        }
        true
    }

    #[allow(clippy::too_many_arguments)]
    fn plan_next_state_action_entry(
        action_name: &str,
        func: &tla_tir::bytecode::BytecodeFunction,
        state_layout: Option<Arc<tla_jit_abi::StateLayout>>,
        opt_level: tla_trust_cg::OptLevel,
        const_pool: Option<Arc<tla_tir::bytecode::ConstantPool>>,
        chunk: Option<Arc<tla_tir::bytecode::BytecodeChunk>>,
        chunk_callee_shapes: Option<&tla_ir::lower::ChunkCalleeReturnShapes>,
        action_compile_tasks: &mut Vec<TrustCgActionCompileTask>,
        inner_exists_expansion_keys: &mut FxHashMap<String, Vec<String>>,
        inner_exists_expansion_proofs: &mut FxHashMap<String, TrustCgInnerExistsExpansionProof>,
        stats: &mut TrustCgBuildStats,
        binding_values: &[i64],
        formal_values: &[i64],
        scalarize_env: Option<&RecordSetScalarizeEnv<'_>>,
    ) {
        if !Self::has_residual_exists_opcode(func) {
            Self::plan_next_state_action_exact(
                action_name,
                func,
                state_layout,
                opt_level,
                const_pool,
                chunk,
                chunk_callee_shapes,
                None,
                action_compile_tasks,
                binding_values,
                formal_values,
            );
            return;
        }

        if trust_cg_dump_filter_matches(TRUST_CG_DUMP_ACTION_BYTECODE_ENV, action_name) {
            eprintln!("[trust-cg] action {action_name} bytecode: {func:#?}");
            if let Some(layout) = state_layout.as_deref() {
                eprintln!("[trust-cg] action {action_name} state layout: {layout:#?}");
            }
        }

        let expansion_const_pool = const_pool
            .as_deref()
            .or_else(|| chunk.as_ref().map(|chunk| &chunk.constants));
        let expansion_mode;
        let expanded: Vec<RuntimeGuardedInnerExistsExpansion> = if let Some(expanded) =
            Self::sorted_inner_exists_expansions(func, expansion_const_pool)
        {
            expansion_mode = "statically";
            let native_fused_proof = Self::static_inner_exists_native_fused_proof(&expanded);
            expanded
                .into_iter()
                .map(|action| RuntimeGuardedInnerExistsExpansion {
                    action,
                    const_pool: None,
                    action_local_set_domain_proof: None,
                    native_fused_proof: native_fused_proof.clone(),
                    inner_binding_literals: None,
                })
                .collect()
        } else if let Some(expanded) = Self::sorted_runtime_guarded_inner_exists_expansion_plans(
            func,
            state_layout.as_deref(),
            expansion_const_pool,
        ) {
            expansion_mode = "with runtime membership guards";
            expanded
        } else if Self::try_plan_split_disjunction_action_entry(
            action_name,
            func,
            state_layout.clone(),
            opt_level,
            const_pool.clone(),
            chunk.clone(),
            chunk_callee_shapes,
            action_compile_tasks,
            inner_exists_expansion_keys,
            inner_exists_expansion_proofs,
            stats,
            binding_values,
            formal_values,
            scalarize_env,
        ) {
            // Wall 3: the action was a top-level disjunction whose per-disjunct
            // sub-actions each lowered to native (handled entirely inside the
            // helper, which registered the union of expansion keys + a
            // SplitDisjunction native-fused proof under `action_name`).
            return;
        } else {
            stats.actions_failed += 1;
            // Route B: an `\E m \in <state var>` action whose domain carries a
            // proven-closed RecordSetBitmask layout can execute natively as a
            // multi-successor record-set kernel (gated on TY_RECORD_SET_NATIVE=1;
            // classify returns None otherwise so the default path is untouched).
            // Push a normal compile task flagged `next_state_loop: true` so it
            // lowers via `lower_next_state_loop_scaffold` and dispatches through
            // the sink call convention.
            if let Some(support) =
                Self::classify_record_set_next_state_loop(func, state_layout.as_deref())
            {
                if support.is_supported() {
                    let (read_vars, write_vars) =
                        Self::action_var_access_sets(func, chunk.as_deref());
                    action_compile_tasks.push(TrustCgActionCompileTask {
                        action_name: action_name.to_string(),
                        func: func.clone(),
                        state_layout: state_layout.clone(),
                        opt_level,
                        const_pool: const_pool.clone(),
                        chunk: chunk.clone(),
                        chunk_callee_shapes: chunk_callee_shapes.cloned(),
                        action_local_set_domain_proof: None,
                        binding_values: binding_values.to_vec(),
                        formal_values: formal_values.to_vec(),
                        read_vars,
                        write_vars,
                        // The record-set loop kernel is a separate lowering
                        // path that never emits the callout.
                        compound_read_vars: Vec::new(),
                        next_state_loop: true,
                    });
                    // Undo the pessimistic failure bump above: this action is
                    // now planned as a native record-set loop kernel.
                    stats.actions_failed = stats.actions_failed.saturating_sub(1);
                    eprintln!(
                        "[trust-cg] action '{action_name}': recognized proven-closed record-set multi-successor inner EXISTS; planning native NextStateLoop record-set kernel (TY_RECORD_SET_NATIVE=1)",
                    );
                    return;
                }
            }
            // Route C: record-set AGGREGATE scalarization (gated on
            // TY_RECORD_SET_NATIVE=1 inside the same env check as Route B).
            // `\E w \in <constant finite set> : <aggregates over a
            // proven-closed record-set var>` (PaxosCommit Phase2a) rewrites
            // per witness into EXISTS-free scalar functions; each registers as
            // its own single-successor native key and the BFS unions their
            // successors, exactly like the runtime-guarded expansion path.
            if std::env::var_os("TY_RECORD_SET_NATIVE").as_deref()
                == Some(std::ffi::OsStr::new("1"))
            {
                if let (Some(env), Some(layout), Some(pool)) =
                    (scalarize_env, state_layout.as_deref(), expansion_const_pool)
                {
                    if let Some(outcome) =
                        record_set_scalarize::scalarize_record_set_aggregate_action(
                            func, pool, layout, env,
                        )
                    {
                        if Self::plan_record_set_scalarized_expansions(
                            action_name,
                            outcome,
                            state_layout.clone(),
                            opt_level,
                            chunk.clone(),
                            chunk_callee_shapes,
                            action_compile_tasks,
                            inner_exists_expansion_keys,
                            inner_exists_expansion_proofs,
                            binding_values,
                            formal_values,
                        ) {
                            // Undo the pessimistic failure bump above.
                            stats.actions_failed = stats.actions_failed.saturating_sub(1);
                            return;
                        }
                    }
                }
            }
            // Recognize the runtime-domain multi-successor ("NextStateLoop")
            // shape that the single-successor native ABI cannot express (one
            // parent -> N successors, where N depends on a runtime domain such
            // as `1 .. natMin(primer, template)`). This shape is the target of
            // the future `tla_jit_abi::NextStateLoopFn` ABI. Today the support
            // gate is `NotYetSupported`, so we still fall back to the
            // interpreter (fail-closed) — emitting a partial/wrong successor
            // set would silently drop or fabricate states. The dedicated
            // diagnostic makes the selection observable and de-risks the later
            // codegen wiring.
            if let Some(support) = Self::classify_runtime_domain_next_state_loop(func) {
                debug_assert!(
                    !support.is_supported(),
                    "NextStateLoop scaffold must remain fail-closed until sound codegen lands"
                );
                stats.next_state_loop_recognized_unsupported += 1;
                let reason =
                    Self::diagnose_inner_exists_expansion_failure(func, state_layout.as_deref());
                stats.record_first_action_failure(action_name, &reason);
                eprintln!(
                    "[trust-cg] action '{action_name}': recognized runtime-domain multi-successor inner EXISTS (NextStateLoop ABI target); native multi-successor support gate={} -- falling back to interpreter (fail-closed). {reason}",
                    support.code(),
                );
                return;
            }
            let reason =
                Self::diagnose_inner_exists_expansion_failure(func, state_layout.as_deref());
            stats.record_first_action_failure(action_name, &reason);
            eprintln!(
                "[trust-cg] skipping action '{action_name}': {reason} (interpreter fallback permanent for this run)",
            );
            return;
        };

        eprintln!(
            "[trust-cg] action '{action_name}': inner EXISTS expanded {expansion_mode} into {} native action function(s)",
            expanded.len(),
        );

        let expanded_keys: Vec<String> = expanded
            .iter()
            .map(|expansion| {
                expansion
                    .inner_binding_literals
                    .as_ref()
                    .and_then(|values| tla_jit_abi::binding_key_for_values(action_name, values))
                    .unwrap_or_else(|| {
                        tla_jit_abi::specialized_key(
                            action_name,
                            &expansion.action.inner_binding_values,
                        )
                    })
            })
            .collect();
        inner_exists_expansion_keys.insert(action_name.to_string(), expanded_keys.clone());
        let native_fused_proof = expanded
            .first()
            .and_then(|expansion| expansion.native_fused_proof.clone())
            .filter(|first| {
                expanded
                    .iter()
                    .all(|expansion| expansion.native_fused_proof.as_ref() == Some(first))
            });
        if let Some(kind) = native_fused_proof {
            inner_exists_expansion_proofs.insert(
                action_name.to_string(),
                TrustCgInnerExistsExpansionProof {
                    expansion_count: expanded_keys.len(),
                    kind,
                },
            );
        }

        for (expanded_key, expansion) in expanded_keys.into_iter().zip(expanded) {
            let RuntimeGuardedInnerExistsExpansion {
                action,
                const_pool: expanded_const_pool,
                action_local_set_domain_proof,
                ..
            } = expansion;
            if Self::has_residual_exists_opcode(&action.func) {
                stats.actions_failed += 1;
                let reason = "inner EXISTS expansion left residual EXISTS opcodes";
                stats.record_first_action_failure(&expanded_key, reason);
                eprintln!("[trust-cg] failed to expand action '{expanded_key}': {reason}");
                continue;
            }
            let mut expanded_binding_values =
                Vec::with_capacity(binding_values.len() + action.inner_binding_values.len());
            expanded_binding_values.extend_from_slice(binding_values);
            expanded_binding_values.extend_from_slice(&action.inner_binding_values);
            let compile_const_pool = expanded_const_pool
                .map(Arc::new)
                .or_else(|| const_pool.clone());
            let compile_chunk = match (chunk.as_deref(), compile_const_pool.as_deref()) {
                (Some(source_chunk), Some(pool)) => {
                    let mut chunk = source_chunk.clone();
                    chunk.constants = pool.clone();
                    Some(Arc::new(chunk))
                }
                _ => chunk.clone(),
            };
            // The expanded chunk shares the source chunk's functions; its
            // constant pool only differs by appended entries, so the
            // precomputed callee shapes remain valid (see reuse contract).
            Self::plan_next_state_action_exact(
                &expanded_key,
                &action.func,
                state_layout.clone(),
                opt_level,
                compile_const_pool,
                compile_chunk,
                chunk_callee_shapes,
                action_local_set_domain_proof,
                action_compile_tasks,
                &expanded_binding_values,
                formal_values,
            );
        }
    }

    fn static_inner_exists_native_fused_proof(
        expanded: &[tla_tir::bytecode::ExpandedAction],
    ) -> Option<TrustCgInnerExistsExpansionProofKind> {
        if expanded.is_empty() {
            return None;
        }
        let binding_values = expanded
            .iter()
            .map(|expansion| expansion.inner_binding_values.clone())
            .collect::<Vec<_>>();
        let mut seen = FxHashSet::default();
        if !binding_values.iter().all(|values| seen.insert(values)) {
            return None;
        }
        Some(TrustCgInnerExistsExpansionProofKind::StaticFiniteDomain { binding_values })
    }

    fn lower_next_state_action_task(
        task: TrustCgActionCompileTask,
    ) -> Result<TrustCgLoweredActionCompileTask, TrustCgActionCompileOutcome> {
        match Self::lower_next_state_action_with_trust_ir_proof_facts_and_callee_shapes(
            &task.action_name,
            &task.func,
            task.state_layout.as_deref(),
            task.const_pool.as_deref(),
            task.chunk.as_deref(),
            task.chunk_callee_shapes.as_ref(),
            task.action_local_set_domain_proof.as_ref(),
            task.next_state_loop,
        ) {
            Ok((trust_ir_module, symbol_name, trust_ir_proof_facts)) => {
                Ok(TrustCgLoweredActionCompileTask {
                    action_name: task.action_name,
                    opt_level: task.opt_level,
                    trust_ir_module,
                    symbol_name,
                    binding_values: task.binding_values,
                    formal_values: task.formal_values,
                    read_vars: task.read_vars,
                    write_vars: task.write_vars,
                    compound_read_vars: task.compound_read_vars,
                    trust_ir_proof_facts,
                    next_state_loop: task.next_state_loop,
                })
            }
            Err(e) => Err(TrustCgActionCompileOutcome::Failed {
                action_name: task.action_name,
                message: e.to_string(),
            }),
        }
    }

    fn compile_lowered_next_state_action_task(
        lowered: TrustCgLoweredActionCompileTask,
    ) -> TrustCgActionCompileOutcome {
        match Self::compile_lowered_next_state_action_task_inner(lowered) {
            Ok(outcome) => outcome,
            Err((action_name, message)) => TrustCgActionCompileOutcome::Failed {
                action_name,
                message,
            },
        }
    }

    fn compile_lowered_next_state_action_task_inner(
        lowered: TrustCgLoweredActionCompileTask,
    ) -> Result<TrustCgActionCompileOutcome, (String, String)> {
        let lib = tla_trust_cg::compile_module_native(&lowered.trust_ir_module, lowered.opt_level)
            .map_err(|e| (lowered.action_name.clone(), e.to_string()))?;
        let fn_ptr = unsafe {
            let raw = lib
                .get_symbol(&lowered.symbol_name)
                .map_err(|e| (lowered.action_name.clone(), e.to_string()))?;
            std::mem::transmute::<*mut std::ffi::c_void, NativeNextStateFn>(raw)
        };

        Ok(TrustCgActionCompileOutcome::Compiled {
            action_name: lowered.action_name,
            fn_ptr,
            library: lib,
            symbol_name: lowered.symbol_name,
            binding_values: lowered.binding_values,
            formal_values: lowered.formal_values,
            read_vars: lowered.read_vars,
            write_vars: lowered.write_vars,
            compound_read_vars: lowered.compound_read_vars,
            trust_ir_proof_facts: lowered.trust_ir_proof_facts,
            batch_shard: None,
            next_state_loop: lowered.next_state_loop,
        })
    }

    fn compile_next_state_action_task(
        task: TrustCgActionCompileTask,
    ) -> TrustCgActionCompileOutcome {
        match Self::lower_next_state_action_task(task) {
            Ok(lowered) => Self::compile_lowered_next_state_action_task(lowered),
            Err(outcome) => outcome,
        }
    }

    fn compile_lowered_next_state_action_tasks(
        tasks: Vec<TrustCgLoweredActionCompileTask>,
        jobs: usize,
    ) -> Vec<TrustCgActionCompileOutcome> {
        // Force atomic `Rp` refcounts while this batch may spawn+join compile
        // worker threads that touch shared `Value`s. No-op unless the sequential
        // BFS enabled the non-atomic fast path; the guard restores it on return.
        let _rp_pause = tla_value::rp::pause_single_threaded();
        if jobs <= 1 || tasks.len() <= 1 {
            return tasks
                .into_iter()
                .map(Self::compile_lowered_next_state_action_task)
                .collect();
        }

        let mut indexed_tasks = tasks.into_iter().enumerate();
        let mut indexed_results = Vec::new();
        loop {
            let mut handles = Vec::new();
            for _ in 0..jobs {
                let Some((idx, task)) = indexed_tasks.next() else {
                    break;
                };
                handles.push((
                    idx,
                    std::thread::spawn(move || Self::compile_lowered_next_state_action_task(task)),
                ));
            }
            if handles.is_empty() {
                break;
            }
            let mut first_panic = None;
            for (idx, handle) in handles {
                match handle.join() {
                    Ok(outcome) => indexed_results.push((idx, outcome)),
                    Err(payload) => {
                        if first_panic.is_none() {
                            first_panic = Some(payload);
                        }
                    }
                }
            }
            if let Some(payload) = first_panic {
                std::panic::resume_unwind(payload);
            }
        }

        indexed_results.sort_by_key(|(idx, _)| *idx);
        indexed_results
            .into_iter()
            .map(|(_, outcome)| outcome)
            .collect()
    }

    #[cfg(test)]
    fn compile_next_state_action_tasks_as_batch(
        tasks: Vec<TrustCgActionCompileTask>,
        jobs: usize,
    ) -> (
        Vec<TrustCgActionCompileOutcome>,
        TrustCgNativeActionCalloutBatchStats,
    ) {
        let caller_identity = tla_trust_cg::compile::BatchJitCallerIdentity::empty();
        Self::compile_next_state_action_tasks_as_batch_with_caller_identity(
            tasks,
            jobs,
            &caller_identity,
        )
    }

    fn compile_next_state_action_tasks_as_batch_with_caller_identity(
        tasks: Vec<TrustCgActionCompileTask>,
        jobs: usize,
        caller_identity: &tla_trust_cg::compile::BatchJitCallerIdentity,
    ) -> (
        Vec<TrustCgActionCompileOutcome>,
        TrustCgNativeActionCalloutBatchStats,
    ) {
        let setup_start = std::time::Instant::now();
        let task_count = tasks.len();
        let mut batch_stats = TrustCgNativeActionCalloutBatchStats::attempted(task_count);
        let lowering_start = std::time::Instant::now();
        let mut indexed_results = Vec::new();
        let mut lowered_tasks = Vec::new();
        for (idx, task) in tasks.into_iter().enumerate() {
            match Self::lower_next_state_action_task(task) {
                Ok(lowered) => lowered_tasks.push((idx, lowered)),
                Err(outcome) => indexed_results.push((idx, outcome)),
            }
        }
        batch_stats.lowered_tasks = lowered_tasks.len();
        batch_stats.lowering_failed = indexed_results.len();
        batch_stats.lowering_ms = trust_cg_elapsed_ms(lowering_start);
        if trust_cg_setup_timing_enabled() {
            eprintln!(
                "[trust_cg-timing] native_action_callout_lowering_ms={} tasks={} lowered={} lowering_failed={} batch_enabled=1",
                batch_stats.lowering_ms,
                task_count,
                lowered_tasks.len(),
                indexed_results.len(),
            );
        }

        if lowered_tasks.len() <= 1 {
            batch_stats.fallback_reason = if lowered_tasks.is_empty() {
                TrustCgActionCalloutBatchFallbackReason::NoLoweredTasks
            } else {
                TrustCgActionCalloutBatchFallbackReason::SingleLoweredTask
            };
            batch_stats.fallback_per_action_tasks = lowered_tasks.len();
            let fallback_compile_start = std::time::Instant::now();
            for (idx, outcome) in lowered_tasks
                .into_iter()
                .map(|(idx, lowered)| (idx, Self::compile_lowered_next_state_action_task(lowered)))
            {
                indexed_results.push((idx, outcome));
            }
            batch_stats.fallback_per_action_compile_ms =
                trust_cg_elapsed_ms(fallback_compile_start);
        } else {
            let batch_result = match Self::plan_native_action_callout_batch_shards(&lowered_tasks) {
                Ok(shard_plan) => {
                    batch_stats.record_shard_plan(&shard_plan);
                    Self::compile_lowered_next_state_action_tasks_as_shards_with_stats(
                        &lowered_tasks,
                        &shard_plan,
                        &mut batch_stats,
                        caller_identity,
                        jobs,
                    )
                }
                Err(fallback) => Err(fallback),
            };
            match batch_result {
                Ok(batch_outcomes) => {
                    batch_stats.batch_compiled = batch_outcomes.len();
                    batch_stats.fallback_reason =
                        TrustCgActionCalloutBatchFallbackReason::NoFallback;
                    if batch_stats.sharding_policy_selected {
                        eprintln!(
                            "[trust-cg] compiled {} native action callout(s) in {} batch shard artifact(s)",
                            batch_outcomes.len(),
                            batch_stats.shard_count,
                        );
                    } else {
                        eprintln!(
                            "[trust-cg] compiled {} native action callout(s) in one batch artifact",
                            batch_outcomes.len(),
                        );
                    }
                    for ((idx, _), outcome) in lowered_tasks.into_iter().zip(batch_outcomes) {
                        indexed_results.push((idx, outcome));
                    }
                }
                Err(fallback) => {
                    batch_stats.record_batch_fallback(&fallback);
                    eprintln!(
                        "[trust-cg] native action callout batch unavailable: {}; fallback_reason={}; using per-action compilation",
                        fallback.message,
                        fallback.reason.code(),
                    );
                    let indices = lowered_tasks
                        .iter()
                        .map(|(idx, _)| *idx)
                        .collect::<Vec<_>>();
                    let lowered_only = lowered_tasks
                        .into_iter()
                        .map(|(_, lowered)| lowered)
                        .collect::<Vec<_>>();
                    batch_stats.fallback_per_action_tasks = lowered_only.len();
                    let fallback_compile_start = std::time::Instant::now();
                    let outcomes =
                        Self::compile_lowered_next_state_action_tasks(lowered_only, jobs);
                    batch_stats.fallback_per_action_compile_ms =
                        trust_cg_elapsed_ms(fallback_compile_start);
                    if trust_cg_setup_timing_enabled() {
                        eprintln!(
                            "[trust_cg-timing] native_action_callout_batch_fallback_per_action_compile_ms={} lowered_tasks={} jobs={}",
                            batch_stats.fallback_per_action_compile_ms,
                            indices.len(),
                            jobs,
                        );
                    }
                    for (idx, outcome) in indices.into_iter().zip(outcomes) {
                        indexed_results.push((idx, outcome));
                    }
                }
            }
        }

        indexed_results.sort_by_key(|(idx, _)| *idx);
        let outcomes = indexed_results
            .into_iter()
            .map(|(_, outcome)| outcome)
            .collect();
        batch_stats.setup_ms = trust_cg_elapsed_ms(setup_start);
        (outcomes, batch_stats)
    }

    fn lowered_next_state_action_tasks_common_opt_level(
        lowered_tasks: &[(usize, TrustCgLoweredActionCompileTask)],
    ) -> Result<Option<tla_trust_cg::OptLevel>, TrustCgActionCalloutBatchFallback> {
        let lowered_task_refs = lowered_tasks
            .iter()
            .map(|(idx, task)| (*idx, task))
            .collect::<Vec<_>>();
        Self::lowered_next_state_action_task_refs_common_opt_level(&lowered_task_refs)
    }

    fn lowered_next_state_action_task_refs_common_opt_level(
        lowered_tasks: &[(usize, &TrustCgLoweredActionCompileTask)],
    ) -> Result<Option<tla_trust_cg::OptLevel>, TrustCgActionCalloutBatchFallback> {
        let Some((_, first)) = lowered_tasks.first() else {
            return Ok(None);
        };
        let opt_level = first.opt_level;
        if lowered_tasks
            .iter()
            .any(|(_, task)| task.opt_level != opt_level)
        {
            return Err(TrustCgActionCalloutBatchFallback::new(
                TrustCgActionCalloutBatchFallbackReason::MixedOptLevels,
                "mixed opt levels in native action callout batch",
            ));
        }
        Ok(Some(opt_level))
    }

    fn estimate_native_action_callout_ir_nodes(task: &TrustCgLoweredActionCompileTask) -> usize {
        let module = &task.trust_ir_module;
        let function_count = module.functions.len();
        let block_count: usize = module
            .functions
            .iter()
            .map(|function| function.blocks.len())
            .sum();
        let instruction_count: usize = module
            .functions
            .iter()
            .flat_map(|function| function.blocks.iter())
            .map(|block| block.body.len())
            .sum();
        instruction_count
            + block_count.saturating_mul(2)
            + function_count.saturating_mul(4)
            + module.func_types.len()
            + module.types.len()
            + module.structs.len().saturating_mul(2)
            + module.enums.len().saturating_mul(2)
            + module.records.len().saturating_mul(2)
            + module.closure_types.len().saturating_mul(2)
            + module.globals.len().saturating_mul(2)
            + module.proof_obligations.len()
            + module.proof_certificates.len()
    }

    fn plan_native_action_callout_batch_shards(
        lowered_tasks: &[(usize, TrustCgLoweredActionCompileTask)],
    ) -> Result<TrustCgActionCalloutShardPlan, TrustCgActionCalloutBatchFallback> {
        let lowered_task_refs = lowered_tasks
            .iter()
            .map(|(idx, task)| (*idx, task))
            .collect::<Vec<_>>();
        Self::plan_native_action_callout_batch_shards_from_refs(&lowered_task_refs)
    }

    fn plan_native_action_callout_batch_shards_from_refs(
        lowered_tasks: &[(usize, &TrustCgLoweredActionCompileTask)],
    ) -> Result<TrustCgActionCalloutShardPlan, TrustCgActionCalloutBatchFallback> {
        let estimates = lowered_tasks
            .iter()
            .map(|(_, task)| Self::estimate_native_action_callout_ir_nodes(task))
            .collect::<Vec<_>>();
        let estimated_ir_nodes = estimates.iter().sum();
        let frontend_neutral_module_digests = lowered_tasks
            .iter()
            .map(|(_, task)| {
                let stats = tla_trust_cg::BatchJitStats::from_module(
                    &task.trust_ir_module,
                    tla_trust_cg::BatchJitOptions {
                        opt_level: task.opt_level,
                    },
                );
                stats.artifact_identity().semantic_digest.clone()
            })
            .collect::<Vec<_>>();

        let inputs = lowered_tasks
            .iter()
            .enumerate()
            .map(|(input_index, (_, task))| {
                BatchPlanningInput::new(
                    &frontend_neutral_module_digests[input_index],
                    &task.trust_ir_module,
                    estimates[input_index] as u64,
                )
                .with_evidence_id(&task.action_name)
            });
        let partition_plan = plan_frontend_neutral_module_batch_partitions(
            inputs,
            BatchPartitionOptions::new(TRUST_CG_NATIVE_ACTION_CALLOUT_SHARD_MAX_ACTIONS)
                .with_max_estimated_ir_size_per_shard(
                    TRUST_CG_NATIVE_ACTION_CALLOUT_SHARD_MAX_ESTIMATED_IR_NODES as u64,
                ),
        )
        .map_err(|err| {
            TrustCgActionCalloutBatchFallback::new(
                TrustCgActionCalloutBatchFallbackReason::TrustIrBatchAssembly,
                err,
            )
        })?;

        let mut planned_shards: Vec<(TrustCgActionCalloutShard, String)> = Vec::new();
        for batch_shard in partition_plan.shards {
            for input_index in batch_shard.input_indices.iter().copied() {
                if input_index >= estimates.len() {
                    return Err(TrustCgActionCalloutBatchFallback::new(
                        TrustCgActionCalloutBatchFallbackReason::TrustIrBatchAssembly,
                        format!(
                            "trust-ir batch partition produced out-of-range input index {input_index}"
                        ),
                    ));
                }
            }

            let mut input_indices = batch_shard.input_indices.clone();
            input_indices.sort_by(|left, right| {
                frontend_neutral_module_digests[*left]
                    .cmp(&frontend_neutral_module_digests[*right])
                    .then_with(|| estimates[*left].cmp(&estimates[*right]))
                    .then_with(|| left.cmp(right))
            });
            let shard_nodes = usize::try_from(batch_shard.estimated_ir_size).unwrap_or(usize::MAX);
            planned_shards.push((
                TrustCgActionCalloutShard {
                    input_indices,
                    estimated_ir_nodes: shard_nodes,
                    stable_id: batch_shard.stable_id,
                    shared_shape_id: batch_shard.shared_shape_id,
                    frontend_neutral_reuse_id: batch_shard.frontend_neutral_reuse_id,
                    digest_input_sha256: String::new(),
                },
                batch_shard.digest_input,
            ));
        }

        let mut digest_base_counts = std::collections::BTreeMap::new();
        for (_, shard_digest_input) in &planned_shards {
            *digest_base_counts
                .entry(shard_digest_input.clone())
                .or_insert(0usize) += 1;
        }
        let mut digest_base_seen = std::collections::BTreeMap::new();
        let mut shards = Vec::with_capacity(planned_shards.len());
        for (mut shard, mut shard_digest_input) in planned_shards {
            let digest_base_count = digest_base_counts
                .get(&shard_digest_input)
                .copied()
                .unwrap_or(1);
            if digest_base_count > 1 {
                let digest_base_ordinal = {
                    let seen = digest_base_seen
                        .entry(shard_digest_input.clone())
                        .or_insert(0usize);
                    *seen += 1;
                    *seen
                };
                shard_digest_input.push_str("identical_shard_digest_base_count=");
                shard_digest_input.push_str(&digest_base_count.to_string());
                shard_digest_input.push('\n');
                shard_digest_input.push_str("identical_shard_digest_base_ordinal=");
                shard_digest_input.push_str(&digest_base_ordinal.to_string());
                shard_digest_input.push('\n');
            }
            shard.digest_input_sha256 = trust_cg_native_admission_sha256(
                "native_action_callout_batch_shard_digest_input",
                &shard_digest_input,
            );
            shards.push(shard);
        }

        Ok(TrustCgActionCalloutShardPlan {
            policy_selected: shards.len() > 1,
            shards,
            estimated_ir_nodes,
            trust_ir_batch_partition_plan_reuse_manifest_id: partition_plan
                .reuse_manifest
                .manifest_id,
        })
    }

    #[cfg(test)]
    fn compile_lowered_next_state_action_tasks_as_batch_with_stats(
        lowered_tasks: &[(usize, TrustCgLoweredActionCompileTask)],
        batch_stats: Option<&mut TrustCgNativeActionCalloutBatchStats>,
    ) -> Result<Vec<TrustCgActionCompileOutcome>, TrustCgActionCalloutBatchFallback> {
        let caller_identity = tla_trust_cg::compile::BatchJitCallerIdentity::empty();
        Self::compile_lowered_next_state_action_tasks_as_batch_with_caller_identity(
            lowered_tasks,
            batch_stats,
            &caller_identity,
        )
    }

    #[allow(dead_code)] // JIT batch-compile / trust-cg-cache machinery, currently unwired (TY_JIT off by default, #4035)
    fn compile_lowered_next_state_action_tasks_as_batch_with_caller_identity(
        lowered_tasks: &[(usize, TrustCgLoweredActionCompileTask)],
        batch_stats: Option<&mut TrustCgNativeActionCalloutBatchStats>,
        caller_identity: &tla_trust_cg::compile::BatchJitCallerIdentity,
    ) -> Result<Vec<TrustCgActionCompileOutcome>, TrustCgActionCalloutBatchFallback> {
        let lowered_task_refs = lowered_tasks
            .iter()
            .map(|(idx, task)| (*idx, task))
            .collect::<Vec<_>>();
        Self::compile_lowered_next_state_action_tasks_as_batch_named_with_stats(
            "ty_native_action_callout_batch",
            &lowered_task_refs,
            None,
            batch_stats,
            caller_identity,
        )
    }

    fn compile_lowered_next_state_action_tasks_as_batch_named_with_stats(
        batch_module_name: &str,
        lowered_tasks: &[(usize, &TrustCgLoweredActionCompileTask)],
        shard_metadata: Option<TrustCgNativeActionBatchShardCompileMetadata>,
        mut batch_stats: Option<&mut TrustCgNativeActionCalloutBatchStats>,
        caller_identity: &tla_trust_cg::compile::BatchJitCallerIdentity,
    ) -> Result<Vec<TrustCgActionCompileOutcome>, TrustCgActionCalloutBatchFallback> {
        let opt_level =
            match Self::lowered_next_state_action_task_refs_common_opt_level(lowered_tasks) {
                Ok(Some(opt_level)) => opt_level,
                Ok(None) => return Ok(Vec::new()),
                Err(fallback) => {
                    if let Some(stats) = batch_stats.as_deref_mut() {
                        stats.record_batch_fallback(&fallback);
                    }
                    return Err(fallback);
                }
            };
        if let Some(stats) = batch_stats.as_deref_mut() {
            if stats.shard_count == 0 && !lowered_tasks.is_empty() {
                match Self::plan_native_action_callout_batch_shards_from_refs(lowered_tasks) {
                    Ok(plan) => stats.record_shard_plan(&plan),
                    Err(fallback) => {
                        stats.record_batch_fallback(&fallback);
                        return Err(fallback);
                    }
                }
            }
        }
        let modules = lowered_tasks
            .iter()
            .map(|(_, task)| &task.trust_ir_module)
            .collect::<Vec<_>>();
        if let Some(stats) = batch_stats.as_deref_mut() {
            stats.batch_assembly_attempted = true;
        }
        let batch_assembly_start = std::time::Instant::now();
        let batch_module_result =
            tla_ir::module_batch::assemble_module_batch(batch_module_name, modules);
        let batch_assembly_ms = trust_cg_elapsed_ms(batch_assembly_start);
        if let Some(stats) = batch_stats.as_deref_mut() {
            stats.batch_assembly_ms = stats.batch_assembly_ms.saturating_add(batch_assembly_ms);
            stats.shard_assembly_ms.push(batch_assembly_ms);
            stats.batch_assembly_failed |= batch_module_result.is_err();
        }
        if trust_cg_setup_timing_enabled() {
            eprintln!(
                "[trust_cg-timing] native_action_callout_batch_assembly_ms={} lowered_tasks={} opt_level={} batch_module={} result={}",
                batch_assembly_ms,
                lowered_tasks.len(),
                opt_level.as_str(),
                batch_module_name,
                if batch_module_result.is_ok() { "ok" } else { "err" },
            );
        }
        let batch_module = match batch_module_result {
            Ok(batch_module) => batch_module,
            Err(err) => {
                let fallback = TrustCgActionCalloutBatchFallback::new(
                    TrustCgActionCalloutBatchFallbackReason::TrustIrBatchAssembly,
                    err,
                );
                if let Some(stats) = batch_stats.as_deref_mut() {
                    stats.record_batch_fallback(&fallback);
                }
                return Err(fallback);
            }
        };
        let exports = lowered_tasks
            .iter()
            .map(|(_, task)| task.symbol_name.clone())
            .collect::<Vec<_>>();
        let symbols = match tla_trust_cg::BatchJitSymbolContract::empty().with_exports(exports) {
            Ok(symbols) => symbols,
            Err(err) => {
                let fallback = TrustCgActionCalloutBatchFallback::new(
                    TrustCgActionCalloutBatchFallbackReason::SymbolContract,
                    err,
                );
                if let Some(stats) = batch_stats.as_deref_mut() {
                    stats.record_batch_fallback(&fallback);
                }
                return Err(fallback);
            }
        };
        let batch_options = tla_trust_cg::BatchJitOptions { opt_level };
        let batch_caller_identity =
            Self::native_batch_caller_identity_for_shard(caller_identity, shard_metadata.as_ref());
        // Prepare the assembled batch module exactly once. `candidate_stats`
        // below (used for the warm-cache lookup) and the on-miss compile call
        // both reuse this single frontend-neutral preparation instead of each
        // re-running `BatchJitPreparedManifest::from_module` over the same
        // module. The results are byte-identical to the previous two-build flow:
        // `candidate_stats` matches
        // `BatchJitStats::from_module_with_symbols_and_caller_identity` and
        // `prepared_batch.compile` matches
        // `compile_batch_with_symbols_and_caller_identity`.
        let prepared_batch = tla_trust_cg::prepare_batch(&batch_module);
        let candidate_stats =
            prepared_batch.candidate_stats(batch_options, &symbols, &batch_caller_identity);
        if let Some(stats) = batch_stats.as_deref_mut() {
            stats.record_batch_jit_stats(&candidate_stats, "trust_cg_batch_candidate_stats", false);
        }
        let warm_lookup_start = std::time::Instant::now();
        let warm_lookup =
            lookup_trust_cg_native_batch_warm_artifact(candidate_stats.artifact_identity());
        let warm_lookup_ms = trust_cg_elapsed_ms(warm_lookup_start);
        if let Some(stats) = batch_stats.as_deref_mut() {
            stats.record_warm_cache_lookup(&warm_lookup.key, warm_lookup.status, warm_lookup_ms);
        }
        let warm_cache_status = warm_lookup.status;
        let (library, artifact_stats, warm_cache_status) = if let Some(artifact) =
            warm_lookup.artifact
        {
            if let Some(stats) = batch_stats.as_deref_mut() {
                stats.shard_compile_ms.push(0);
                stats.record_batch_jit_stats(
                    &artifact.stats,
                    "trust_cg_warm_batch_artifact_stats",
                    true,
                );
            }
            (artifact.library, artifact.stats, warm_cache_status)
        } else {
            if let Some(stats) = batch_stats.as_deref_mut() {
                stats.batch_compile_attempted = true;
            }
            let batch_compile_start = std::time::Instant::now();
            let batch_result =
                prepared_batch.compile(batch_options, &symbols, &batch_caller_identity);
            let batch_compile_ms = trust_cg_elapsed_ms(batch_compile_start);
            if let Some(stats) = batch_stats.as_deref_mut() {
                stats.batch_compile_ms = stats.batch_compile_ms.saturating_add(batch_compile_ms);
                stats.shard_compile_ms.push(batch_compile_ms);
                stats.batch_compile_failed |= batch_result.is_err();
            }
            if trust_cg_setup_timing_enabled() {
                eprintln!(
                        "[trust_cg-timing] native_action_callout_batch_compile_ms={} lowered_tasks={} exports={} opt_level={} batch_module={} warm_cache_status={} result={}",
                        batch_compile_ms,
                        lowered_tasks.len(),
                        symbols.exports().len(),
                        opt_level.as_str(),
                        batch_module_name,
                        warm_cache_status,
                        if batch_result.is_ok() { "ok" } else { "err" },
                    );
            }
            let batch = match batch_result {
                Ok(batch) => batch,
                Err(err) => {
                    let fallback = TrustCgActionCalloutBatchFallback::new(
                        TrustCgActionCalloutBatchFallbackReason::BatchCompile,
                        err,
                    );
                    if let Some(stats) = batch_stats.as_deref_mut() {
                        stats.record_batch_fallback(&fallback);
                    }
                    return Err(fallback);
                }
            };
            let artifact_stats = batch.stats.clone();
            let library = batch.library().clone();
            if let Some(stats) = batch_stats.as_deref_mut() {
                stats.record_batch_jit_stats(
                    &artifact_stats,
                    "trust_cg_compiled_batch_stats",
                    true,
                );
                if store_trust_cg_native_batch_warm_artifact(&artifact_stats, &library) {
                    stats.record_warm_cache_store();
                }
            }
            (batch.into_library(), artifact_stats, warm_cache_status)
        };
        let artifact_materialization_start = std::time::Instant::now();
        let artifact_identity = artifact_stats.artifact_identity();
        let artifact_shared_identity = artifact_identity.shared_engine_identity();
        let artifact_cache_digest = artifact_identity.cache_digest.clone();
        let batch_shard = shard_metadata.as_ref().map(|metadata| {
            metadata.with_artifact(
                artifact_shared_identity.clone(),
                artifact_cache_digest.clone(),
                warm_cache_status,
            )
        });
        let mut outcomes = Vec::with_capacity(lowered_tasks.len());
        for (_, task) in lowered_tasks {
            let fn_ptr = unsafe {
                let raw = match library.get_symbol(&task.symbol_name) {
                    Ok(raw) => raw,
                    Err(err) => {
                        let fallback = TrustCgActionCalloutBatchFallback::new(
                            TrustCgActionCalloutBatchFallbackReason::BatchSymbolLookup,
                            err,
                        );
                        if let Some(stats) = batch_stats.as_deref_mut() {
                            stats.record_artifact_materialization(trust_cg_elapsed_ms(
                                artifact_materialization_start,
                            ));
                            stats.record_batch_fallback(&fallback);
                        }
                        return Err(fallback);
                    }
                };
                std::mem::transmute::<*mut std::ffi::c_void, NativeNextStateFn>(raw)
            };
            outcomes.push(TrustCgActionCompileOutcome::Compiled {
                action_name: task.action_name.clone(),
                fn_ptr,
                library: library.clone(),
                symbol_name: task.symbol_name.clone(),
                binding_values: task.binding_values.clone(),
                formal_values: task.formal_values.clone(),
                read_vars: task.read_vars.clone(),
                write_vars: task.write_vars.clone(),
                compound_read_vars: task.compound_read_vars.clone(),
                trust_ir_proof_facts: task.trust_ir_proof_facts,
                batch_shard: batch_shard.clone(),
                next_state_loop: task.next_state_loop,
            });
        }
        if let Some(stats) = batch_stats {
            stats.record_artifact_materialization(trust_cg_elapsed_ms(
                artifact_materialization_start,
            ));
        }
        Ok(outcomes)
    }

    fn compile_lowered_next_state_action_tasks_as_shards_with_stats(
        lowered_tasks: &[(usize, TrustCgLoweredActionCompileTask)],
        plan: &TrustCgActionCalloutShardPlan,
        batch_stats: &mut TrustCgNativeActionCalloutBatchStats,
        caller_identity: &tla_trust_cg::compile::BatchJitCallerIdentity,
        jobs: usize,
    ) -> Result<Vec<TrustCgActionCompileOutcome>, TrustCgActionCalloutBatchFallback> {
        // Force atomic `Rp` refcounts across the sharded compile (may spawn+join
        // worker threads touching shared `Value`s). See sibling batch fn.
        let _rp_pause = tla_value::rp::pause_single_threaded();
        if let Err(fallback) = Self::lowered_next_state_action_tasks_common_opt_level(lowered_tasks)
        {
            batch_stats.record_batch_fallback(&fallback);
            return Err(fallback);
        }

        let shard_inputs =
            Self::native_action_callout_shard_inputs(lowered_tasks, plan).map_err(|fallback| {
                batch_stats.record_batch_fallback(&fallback);
                fallback
            })?;
        let shard_results = if jobs <= 1 || shard_inputs.len() <= 1 {
            shard_inputs
                .iter()
                .map(|input| Self::compile_native_action_callout_shard(input, caller_identity))
                .collect::<Vec<_>>()
        } else {
            std::thread::scope(|scope| {
                let handles = shard_inputs
                    .iter()
                    .map(|input| {
                        let identity = caller_identity.clone();
                        scope.spawn(move || {
                            Self::compile_native_action_callout_shard(input, &identity)
                        })
                    })
                    .collect::<Vec<_>>();
                handles
                    .into_iter()
                    .map(|handle| match handle.join() {
                        Ok(result) => result,
                        Err(payload) => std::panic::resume_unwind(payload),
                    })
                    .collect::<Vec<_>>()
            })
        };

        let mut indexed_outcomes = Vec::with_capacity(lowered_tasks.len());
        for shard_result in shard_results {
            let shard_result = match shard_result {
                Ok(result) => result,
                Err((fallback, shard_stats)) => {
                    batch_stats.merge_compiled_shard_stats(shard_stats);
                    batch_stats.record_batch_fallback(&fallback);
                    return Err(fallback);
                }
            };
            batch_stats.merge_compiled_shard_stats(shard_result.stats);
            indexed_outcomes.extend(shard_result.indexed_outcomes);
        }
        indexed_outcomes.sort_by_key(|(input_index, _)| *input_index);
        let outcomes = indexed_outcomes
            .into_iter()
            .map(|(_, outcome)| outcome)
            .collect();
        batch_stats.refresh_compiled_artifact_aggregate();
        Ok(outcomes)
    }

    fn native_action_callout_shard_inputs<'a>(
        lowered_tasks: &'a [(usize, TrustCgLoweredActionCompileTask)],
        plan: &TrustCgActionCalloutShardPlan,
    ) -> Result<Vec<TrustCgNativeActionCalloutShardInput<'a>>, TrustCgActionCalloutBatchFallback>
    {
        let mut shard_inputs = Vec::with_capacity(plan.shards.len());
        for (shard_idx, shard) in plan.shards.iter().enumerate() {
            let mut shard_tasks = Vec::with_capacity(shard.input_indices.len());
            for input_index in shard.input_indices.iter().copied() {
                let Some((_, task)) = lowered_tasks.get(input_index) else {
                    return Err(TrustCgActionCalloutBatchFallback::new(
                        TrustCgActionCalloutBatchFallbackReason::TrustIrBatchAssembly,
                        format!(
                            "native action callout shard references missing lowered task index {input_index}"
                        ),
                    ));
                };
                shard_tasks.push((input_index, task));
            }
            let batch_module_name = if plan.shards.len() == 1 {
                "ty_native_action_callout_batch".to_string()
            } else {
                format!(
                    "ty_native_action_callout_batch_shard_{:04}_of_{:04}",
                    shard_idx + 1,
                    plan.shards.len()
                )
            };
            shard_inputs.push(TrustCgNativeActionCalloutShardInput {
                batch_module_name,
                tasks: shard_tasks,
                metadata: TrustCgNativeActionBatchShardCompileMetadata::from_plan_shard(
                    shard_idx,
                    plan.shards.len(),
                    shard,
                    &plan.trust_ir_batch_partition_plan_reuse_manifest_id,
                ),
            });
        }
        Ok(shard_inputs)
    }

    fn compile_native_action_callout_shard(
        input: &TrustCgNativeActionCalloutShardInput<'_>,
        caller_identity: &tla_trust_cg::compile::BatchJitCallerIdentity,
    ) -> Result<
        TrustCgNativeActionCalloutShardResult,
        (
            TrustCgActionCalloutBatchFallback,
            TrustCgNativeActionCalloutBatchStats,
        ),
    > {
        let mut shard_stats = TrustCgNativeActionCalloutBatchStats::attempted(input.tasks.len());
        let result = Self::compile_lowered_next_state_action_tasks_as_batch_named_with_stats(
            &input.batch_module_name,
            &input.tasks,
            Some(input.metadata.clone()),
            Some(&mut shard_stats),
            caller_identity,
        );
        match result {
            Ok(outcomes) => {
                let indexed_outcomes = input
                    .tasks
                    .iter()
                    .map(|(input_index, _)| *input_index)
                    .zip(outcomes)
                    .collect();
                Ok(TrustCgNativeActionCalloutShardResult {
                    indexed_outcomes,
                    stats: shard_stats,
                })
            }
            Err(fallback) => Err((fallback, shard_stats)),
        }
    }

    #[cfg(test)]
    fn compile_next_state_action_tasks(
        tasks: Vec<TrustCgActionCompileTask>,
        jobs: usize,
    ) -> (
        Vec<TrustCgActionCompileOutcome>,
        TrustCgNativeActionCalloutBatchStats,
    ) {
        let caller_identity = tla_trust_cg::compile::BatchJitCallerIdentity::empty();
        Self::compile_next_state_action_tasks_with_caller_identity(tasks, jobs, &caller_identity)
    }

    fn compile_next_state_action_tasks_with_caller_identity(
        tasks: Vec<TrustCgActionCompileTask>,
        jobs: usize,
        caller_identity: &tla_trust_cg::compile::BatchJitCallerIdentity,
    ) -> (
        Vec<TrustCgActionCompileOutcome>,
        TrustCgNativeActionCalloutBatchStats,
    ) {
        // Force atomic `Rp` refcounts across this compile batch (may spawn+join
        // worker threads touching shared `Value`s). See sibling batch fn.
        let _rp_pause = tla_value::rp::pause_single_threaded();
        if trust_cg_native_callout_batch_enabled() && tasks.len() > 1 {
            return Self::compile_next_state_action_tasks_as_batch_with_caller_identity(
                tasks,
                jobs,
                caller_identity,
            );
        }

        if jobs <= 1 || tasks.len() <= 1 {
            let outcomes = tasks
                .into_iter()
                .map(Self::compile_next_state_action_task)
                .collect();
            return (outcomes, TrustCgNativeActionCalloutBatchStats::default());
        }

        let mut indexed_tasks = tasks.into_iter().enumerate();
        let mut indexed_results = Vec::new();
        loop {
            let mut handles = Vec::new();
            for _ in 0..jobs {
                let Some((idx, task)) = indexed_tasks.next() else {
                    break;
                };
                handles.push((
                    idx,
                    std::thread::spawn(move || Self::compile_next_state_action_task(task)),
                ));
            }
            if handles.is_empty() {
                break;
            }
            let mut first_panic = None;
            for (idx, handle) in handles {
                match handle.join() {
                    Ok(outcome) => indexed_results.push((idx, outcome)),
                    Err(payload) => {
                        if first_panic.is_none() {
                            first_panic = Some(payload);
                        }
                    }
                }
            }
            if let Some(payload) = first_panic {
                std::panic::resume_unwind(payload);
            }
        }

        indexed_results.sort_by_key(|(idx, _)| *idx);
        let outcomes = indexed_results
            .into_iter()
            .map(|(_, outcome)| outcome)
            .collect();
        (outcomes, TrustCgNativeActionCalloutBatchStats::default())
    }

    fn native_batch_caller_identity_for_shard(
        caller_identity: &tla_trust_cg::compile::BatchJitCallerIdentity,
        shard_metadata: Option<&TrustCgNativeActionBatchShardCompileMetadata>,
    ) -> tla_trust_cg::compile::BatchJitCallerIdentity {
        let mut identity = caller_identity.clone();
        if identity.plan_reuse_manifest_id.is_none() {
            if let Some(metadata) = shard_metadata {
                identity = identity
                    .with_plan_reuse_manifest_id(metadata.frontend_neutral_reuse_id.as_str());
            }
        }
        identity
    }

    fn dedupe_action_compile_outcomes(
        outcomes: Vec<TrustCgActionCompileOutcome>,
    ) -> Vec<TrustCgActionCompileOutcome> {
        let mut key_index = FxHashMap::default();
        let mut deduped: Vec<TrustCgActionCompileOutcome> = Vec::new();

        for outcome in outcomes {
            let action_name = outcome.action_name().to_string();
            let Some(existing_idx) = key_index.get(&action_name).copied() else {
                key_index.insert(action_name, deduped.len());
                deduped.push(outcome);
                continue;
            };

            if !deduped[existing_idx].is_compiled() && outcome.is_compiled() {
                deduped[existing_idx] = outcome;
            }
        }

        deduped
    }

    fn record_action_compile_outcome(
        outcome: TrustCgActionCompileOutcome,
        next_state_fns: &mut FxHashMap<String, NativeNextStateFn>,
        next_state_loop_fns: &mut FxHashMap<String, tla_jit_abi::NextStateLoopFn>,
        native_action_entries: &mut FxHashMap<String, TrustCgNativeActionEntry>,
        libraries: &mut Vec<tla_trust_cg::NativeLibrary>,
        stats: &mut TrustCgBuildStats,
    ) {
        match outcome {
            TrustCgActionCompileOutcome::Compiled {
                action_name,
                fn_ptr,
                library,
                symbol_name,
                binding_values,
                formal_values,
                read_vars,
                write_vars,
                compound_read_vars,
                trust_ir_proof_facts,
                batch_shard,
                next_state_loop,
            } => {
                if next_state_loop {
                    // `fn_ptr` was resolved from the compiled symbol as a
                    // `NativeNextStateFn`, but the record-set kernel actually
                    // implements the `NextStateLoopFn` ABI (param#2 is a
                    // `*mut NextStateLoopSink`). Both are pointer-sized
                    // `unsafe extern "C" fn`; reinterpret and route it to the
                    // loop map ONLY. It must never enter `next_state_fns`, or a
                    // fused single-successor call would hand it a plain state_out
                    // buffer as a sink pointer and corrupt memory.
                    let loop_fn = unsafe {
                        std::mem::transmute::<NativeNextStateFn, tla_jit_abi::NextStateLoopFn>(
                            fn_ptr,
                        )
                    };
                    next_state_loop_fns.insert(action_name.clone(), loop_fn);
                } else {
                    next_state_fns.insert(action_name.clone(), fn_ptr);
                }
                native_action_entries.insert(
                    action_name.clone(),
                    TrustCgNativeActionEntry {
                        library: library.clone(),
                        symbol_name,
                        binding_values,
                        formal_values,
                        read_vars,
                        write_vars,
                        // Item 4 M1: non-empty only when the lowering emitted a
                        // compound-read callout for the action. Empty keeps the
                        // hybrid admission gate at its strict M0 behaviour.
                        compound_read_vars,
                        batch_shard,
                    },
                );
                libraries.push(library);
                stats.actions_compiled += 1;
                stats.record_trust_ir_proof_facts(trust_ir_proof_facts);
                telemetry_eprintln!(
                    "[trust-cg] compiled next-state for action '{}'",
                    action_name
                );
            }
            TrustCgActionCompileOutcome::Failed {
                action_name,
                message,
            } => {
                // Log-once policy (#4251 Stream 6 requirement c): failed
                // compile marks this action as permanently interpreter-only
                // for the remainder of the run.
                stats.actions_failed += 1;
                stats.record_first_action_failure(&action_name, &message);
                eprintln!(
                    "[trust-cg] failed to compile action '{}': {} (interpreter fallback permanent for this run)",
                    action_name, message,
                );
            }
        }
    }

    /// Build an trust-codegen native cache from bytecode functions.
    ///
    /// For each action, compiles the next-state bytecode function through the
    /// trust-codegen pipeline (BytecodeFunction -> trust-ir -> trust-codegen native code).
    /// Actions that fail compilation are silently skipped (interpreter handles them).
    ///
    /// # Arguments
    ///
    /// * `action_bytecodes` - Map from action name to bytecode function.
    /// * `invariant_bytecodes` - Bytecode functions for each invariant.
    /// * `state_constraint_bytecodes` - Bytecode functions for each state constraint.
    /// * `state_var_count` - Number of state variables in the model.
    /// * `opt_level` - LLVM optimization level.
    /// * `const_pool` - Optional constant pool for resolving `LoadConst`/`Unchanged` opcodes.
    /// * `specializations` - `BindingSpec` entries (one per EXISTS-bound split
    ///   action instance) whose scalar binding values should be baked into a
    ///   specialized bytecode function. Only honored when
    ///   [`TrustCgNativeCache::exists_enabled`] returns true. Part of #4270.
    /// * `chunk` - Optional source bytecode chunk. When provided, action /
    ///   invariant compilation routes through the chunk-aware trust-ir lowering
    ///   path so user-defined operator `Call` opcodes in the entry body
    ///   resolve to fully-defined callee functions instead of leaving
    ///   `__func_N` unresolved-symbol references in the output LLVM module.
    ///   Required to compile any action that calls into another operator
    ///   (DieHard, AsyncTerminationDetection, etc.). Part of #4280 Gap C.
    #[allow(dead_code)] // JIT batch-compile / trust-cg-cache machinery, currently unwired (TY_JIT off by default, #4035)
    pub(in crate::check) fn build(
        action_bytecodes: &FxHashMap<String, &tla_tir::bytecode::BytecodeFunction>,
        invariant_bytecodes: &[Option<&tla_tir::bytecode::BytecodeFunction>],
        state_constraint_bytecodes: &[Option<&tla_tir::bytecode::BytecodeFunction>],
        state_var_count: usize,
        state_layout: Option<&tla_jit_abi::StateLayout>,
        opt_level: tla_trust_cg::OptLevel,
        const_pool: Option<&tla_tir::bytecode::ConstantPool>,
        invariant_const_pool: Option<&tla_tir::bytecode::ConstantPool>,
        state_constraint_const_pool: Option<&tla_tir::bytecode::ConstantPool>,
        specializations: &[BindingSpec],
        chunk: Option<&tla_tir::bytecode::BytecodeChunk>,
        invariant_chunk: Option<&tla_tir::bytecode::BytecodeChunk>,
        state_constraint_chunk: Option<&tla_tir::bytecode::BytecodeChunk>,
    ) -> (Self, TrustCgBuildStats) {
        let raw_action_compile_skip_keys = FxHashMap::default();
        Self::build_with_shadowed_raw_action_keys(
            action_bytecodes,
            invariant_bytecodes,
            state_constraint_bytecodes,
            state_var_count,
            state_layout,
            opt_level,
            const_pool,
            invariant_const_pool,
            state_constraint_const_pool,
            specializations,
            chunk,
            invariant_chunk,
            state_constraint_chunk,
            &raw_action_compile_skip_keys,
        )
    }

    /// Build an trust-codegen native cache while skipping raw split-action bytecodes
    /// that are shadowed by executable BindingSpec alias keys.
    #[allow(dead_code)] // JIT batch-compile / trust-cg-cache machinery, currently unwired (TY_JIT off by default, #4035)
    pub(in crate::check) fn build_with_shadowed_raw_action_keys(
        action_bytecodes: &FxHashMap<String, &tla_tir::bytecode::BytecodeFunction>,
        invariant_bytecodes: &[Option<&tla_tir::bytecode::BytecodeFunction>],
        state_constraint_bytecodes: &[Option<&tla_tir::bytecode::BytecodeFunction>],
        state_var_count: usize,
        state_layout: Option<&tla_jit_abi::StateLayout>,
        opt_level: tla_trust_cg::OptLevel,
        const_pool: Option<&tla_tir::bytecode::ConstantPool>,
        invariant_const_pool: Option<&tla_tir::bytecode::ConstantPool>,
        state_constraint_const_pool: Option<&tla_tir::bytecode::ConstantPool>,
        specializations: &[BindingSpec],
        chunk: Option<&tla_tir::bytecode::BytecodeChunk>,
        invariant_chunk: Option<&tla_tir::bytecode::BytecodeChunk>,
        state_constraint_chunk: Option<&tla_tir::bytecode::BytecodeChunk>,
        raw_action_compile_skip_keys: &FxHashMap<String, String>,
    ) -> (Self, TrustCgBuildStats) {
        let caller_identity = Self::ty_cache_build_caller_identity(
            action_bytecodes,
            invariant_bytecodes,
            state_constraint_bytecodes,
            state_var_count,
            state_layout,
            specializations,
            raw_action_compile_skip_keys,
        );
        Self::build_with_shadowed_raw_action_keys_and_caller_identity(
            action_bytecodes,
            invariant_bytecodes,
            state_constraint_bytecodes,
            state_var_count,
            state_layout,
            opt_level,
            const_pool,
            invariant_const_pool,
            state_constraint_const_pool,
            specializations,
            chunk,
            invariant_chunk,
            state_constraint_chunk,
            raw_action_compile_skip_keys,
            &caller_identity,
            None,
        )
    }

    fn ty_cache_build_caller_identity(
        action_bytecodes: &FxHashMap<String, &tla_tir::bytecode::BytecodeFunction>,
        invariant_bytecodes: &[Option<&tla_tir::bytecode::BytecodeFunction>],
        state_constraint_bytecodes: &[Option<&tla_tir::bytecode::BytecodeFunction>],
        state_var_count: usize,
        state_layout: Option<&tla_jit_abi::StateLayout>,
        specializations: &[BindingSpec],
        raw_action_compile_skip_keys: &FxHashMap<String, String>,
    ) -> tla_trust_cg::compile::BatchJitCallerIdentity {
        use std::fmt::Write as _;

        let mut action_facts: Vec<String> = action_bytecodes
            .iter()
            .map(|(name, func)| {
                format!(
                    "{name}:func_name={}:arity={}:max_register={}:instructions={}",
                    func.name,
                    func.arity,
                    func.max_register,
                    func.instructions.len()
                )
            })
            .collect();
        action_facts.sort_unstable();

        let mut specialization_facts: Vec<String> = specializations
            .iter()
            .map(|spec| {
                format!(
                    "action={}:key={}:binding_values={:?}:binding_literals={:?}:formal_values={:?}:formal_literals={:?}",
                    spec.action_name,
                    spec.binding_key,
                    spec.binding_values,
                    spec.binding_value_literals,
                    spec.formal_values,
                    spec.formal_value_literals
                )
            })
            .collect();
        specialization_facts.sort_unstable();

        let mut raw_skip_keys: Vec<String> = raw_action_compile_skip_keys
            .iter()
            .map(|(raw, alias)| format!("{raw}=>{alias}"))
            .collect();
        raw_skip_keys.sort_unstable();

        let layout_fact = match state_layout {
            Some(layout) => format!(
                "layout=compact:v1:vars={}:slots={}:debug={layout:?}",
                layout.var_count(),
                layout.compact_slot_count()
            ),
            None => format!("layout=absent:v1:state_var_count={state_var_count}"),
        };
        let fingerprint_domain = match state_layout {
            Some(layout) => format!(
                "ty:tla-plus:flat-state:v1:vars={}:slots={}:layout_sha256={}",
                layout.var_count(),
                layout.compact_slot_count(),
                trust_cg_native_admission_sha256("ty_cache_build_state_layout", &layout_fact)
            ),
            None => format!("ty:tla-plus:flat-state:v1:vars={state_var_count}:layout=absent"),
        };

        let invariant_slots = invariant_bytecodes
            .iter()
            .filter(|func| func.is_some())
            .count();
        let state_constraint_slots = state_constraint_bytecodes
            .iter()
            .filter(|func| func.is_some())
            .count();
        let mut source_facts = String::new();
        let _ = write!(
            source_facts,
            "ty_trust_cg_cache_build:v1;state_var_count={state_var_count};{layout_fact};actions={};invariant_slots={}/{};state_constraint_slots={}/{};specializations={};raw_action_compile_skip_keys={}",
            action_facts.join("|"),
            invariant_slots,
            invariant_bytecodes.len(),
            state_constraint_slots,
            state_constraint_bytecodes.len(),
            specialization_facts.join("|"),
            raw_skip_keys.join("|"),
        );

        tla_trust_cg::compile::BatchJitCallerIdentity::empty()
            .with_source_fingerprint(trust_cg_native_admission_sha256(
                "ty_cache_build_source",
                &source_facts,
            ))
            .with_fingerprint_domain_identity(fingerprint_domain)
            .with_cache_namespace_identity("ty:tla-check:trust-cg:native-cache:v1")
    }

    /// Caller identity for the HYBRID flat-view action cache build (item 4
    /// M0-G1).
    ///
    /// Reuses [`Self::ty_cache_build_caller_identity`]'s source facts — the
    /// hybrid layout's `Debug` string carries its `hybrid_flat_view` marker, so
    /// every layout-derived fact/digest already diverges from the whole-state
    /// build — and additionally pins a dedicated cache namespace so hybrid and
    /// whole-state artifacts can never share a warm-cache entry even under
    /// future identity-scheme drift.
    pub(in crate::check) fn ty_hybrid_cache_build_caller_identity(
        action_bytecodes: &FxHashMap<String, &tla_tir::bytecode::BytecodeFunction>,
        state_var_count: usize,
        state_layout: Option<&tla_jit_abi::StateLayout>,
        specializations: &[BindingSpec],
        raw_action_compile_skip_keys: &FxHashMap<String, String>,
    ) -> tla_trust_cg::compile::BatchJitCallerIdentity {
        debug_assert!(
            state_layout.is_none_or(tla_jit_abi::StateLayout::is_hybrid_flat_view),
            "hybrid cache build identity requires a hybrid-marked layout"
        );
        Self::ty_cache_build_caller_identity(
            action_bytecodes,
            &[],
            &[],
            state_var_count,
            state_layout,
            specializations,
            raw_action_compile_skip_keys,
        )
        .with_cache_namespace_identity("ty:tla-check:trust-cg:native-cache:hybrid-flat-view:v1")
    }

    /// Build an trust-codegen native cache with a caller-supplied batch identity.
    ///
    /// The identity is threaded only into the trust-codegen batch callout artifact path;
    /// per-action fallback compilation keeps the historical frontend-neutral
    /// module-native cache behavior.
    pub(in crate::check) fn build_with_shadowed_raw_action_keys_and_caller_identity(
        action_bytecodes: &FxHashMap<String, &tla_tir::bytecode::BytecodeFunction>,
        invariant_bytecodes: &[Option<&tla_tir::bytecode::BytecodeFunction>],
        state_constraint_bytecodes: &[Option<&tla_tir::bytecode::BytecodeFunction>],
        state_var_count: usize,
        state_layout: Option<&tla_jit_abi::StateLayout>,
        opt_level: tla_trust_cg::OptLevel,
        const_pool: Option<&tla_tir::bytecode::ConstantPool>,
        invariant_const_pool: Option<&tla_tir::bytecode::ConstantPool>,
        state_constraint_const_pool: Option<&tla_tir::bytecode::ConstantPool>,
        specializations: &[BindingSpec],
        chunk: Option<&tla_tir::bytecode::BytecodeChunk>,
        invariant_chunk: Option<&tla_tir::bytecode::BytecodeChunk>,
        state_constraint_chunk: Option<&tla_tir::bytecode::BytecodeChunk>,
        raw_action_compile_skip_keys: &FxHashMap<String, String>,
        caller_identity: &tla_trust_cg::compile::BatchJitCallerIdentity,
        scalarize_env: Option<&RecordSetScalarizeEnv<'_>>,
    ) -> (Self, TrustCgBuildStats) {
        let start = std::time::Instant::now();
        let mut stats = TrustCgBuildStats::default();
        let mut next_state_fns = FxHashMap::default();
        let mut next_state_loop_fns = FxHashMap::default();
        let mut inner_exists_expansion_keys = FxHashMap::default();
        let mut inner_exists_expansion_proofs = FxHashMap::default();
        let mut invariant_fns = Vec::new();
        let mut state_constraint_fns = Vec::new();
        let implied_action_fns = Vec::new();
        let mut native_action_entries = FxHashMap::default();
        let mut native_invariant_entries = Vec::new();
        let mut native_state_constraint_entries = Vec::new();
        let native_implied_action_entries = Vec::new();
        let mut libraries = Vec::new();
        let state_layout_shared = state_layout.cloned().map(Arc::new);
        let const_pool_shared = const_pool.cloned().map(Arc::new);
        let chunk_shared = chunk.cloned().map(Arc::new);
        let callout_opt_level = opt_level;

        let exists_enabled = Self::exists_enabled();
        if exists_enabled {
            telemetry_eprintln!(
                "[trust-cg] TY_TRUST_CG_EXISTS=1: specializing {} binding entries (#4270)",
                specializations.len(),
            );
        }
        let specialized_action_names: FxHashSet<&str> = specializations
            .iter()
            .map(|spec| spec.action_name.as_str())
            .collect();
        let mut action_compile_tasks = Vec::new();
        // Shadowed raw split-action forms deferred for possible direct compile.
        // Each entry is (raw_key, func, alias_key). If the alias specialization
        // fails to plan, the raw arity-0 form is compiled under its own key as a
        // sound fallback (its bindings were already substituted as typed literal
        // expressions, so model-value/string provenance is preserved). Compiling
        // both is harmless: dispatch prefers the alias key when present, so the
        // raw entry is only ever reached when the alias was not produced.
        let mut deferred_shadowed_raw: Vec<(
            &String,
            &tla_tir::bytecode::BytecodeFunction,
            &String,
        )> = Vec::new();
        let action_planning_start = std::time::Instant::now();

        // Chunk-wide callee return-shape inference is pure in the chunk
        // functions + the pool entries they reference + the layout, so run it
        // ONCE here and share it across every action compile task instead of
        // re-running it inside `tla_ir::lower` per task. Specialized /
        // expanded chunks derived below only append constant-pool entries
        // (`ConstantPool::add_value` is append-only) and share the same
        // functions, so the precomputed shapes stay exact for them too.
        // Lazily initialized so chunk-less or task-less builds pay nothing.
        let chunk_callee_shapes_cell: std::cell::OnceCell<tla_ir::lower::ChunkCalleeReturnShapes> =
            std::cell::OnceCell::new();
        let chunk_callee_shapes = |chunk: &tla_tir::bytecode::BytecodeChunk| {
            chunk_callee_shapes_cell
                .get_or_init(|| tla_ir::lower::ChunkCalleeReturnShapes::infer(chunk, state_layout))
        };

        // Compile each action's next-state function.
        for (action_name, func) in action_bytecodes {
            // Skip actions with arity > 0 (they have EXISTS binding parameters
            // and cannot be lowered to the next-state ABI which expects arity 0).
            // When `TY_TRUST_CG_EXISTS=1`, the binding specialization loop below
            // handles arity-positive actions by prepending `LoadImm` for each
            // binding register and compiling the resulting arity-0 function.
            // Without specializations, EXISTS-bound actions stay interpreter-only.
            if func.arity != 0 {
                if !count_arity_positive_action_failure(
                    exists_enabled,
                    &specialized_action_names,
                    action_name,
                ) {
                    eprintln!(
                        "[trust-cg] skipping action '{}' as arity-positive wrapper {}; executable BindingSpec specializations are counted separately",
                        action_name, func.arity,
                    );
                    continue;
                }
                stats.actions_failed += 1;
                stats.record_first_action_failure(
                    action_name,
                    format!(
                        "arity-positive {} requires BindingSpec metadata",
                        func.arity
                    ),
                );
                eprintln!(
                    "[trust-cg] skipping action '{}' as arity-positive {} (requires BindingSpec metadata/specialization via split_action_meta before trust-codegen dispatch)",
                    action_name, func.arity,
                );
                continue;
            }

            if exists_enabled {
                if let Some(alias_key) = raw_action_compile_skip_keys.get(action_name.as_str()) {
                    stats.native_action_callouts_skipped_shadowed += 1;
                    eprintln!(
                        "[trust-cg] skipping action '{action_name}' as shadowed raw split action; executable BindingSpec alias '{alias_key}' is counted separately",
                    );
                    // Defer rather than drop: if the alias specialization fails
                    // to plan, the raw arity-0 form is compiled directly below.
                    deferred_shadowed_raw.push((action_name, func, alias_key));
                    continue;
                }
            }

            Self::plan_next_state_action_entry(
                action_name,
                func,
                state_layout_shared.clone(),
                callout_opt_level,
                const_pool_shared.clone(),
                chunk_shared.clone(),
                chunk_shared.as_deref().map(&chunk_callee_shapes),
                &mut action_compile_tasks,
                &mut inner_exists_expansion_keys,
                &mut inner_exists_expansion_proofs,
                &mut stats,
                &[],
                &[],
                scalarize_env,
            );
        }

        // Bytecode-level disjunction-split arms: the next-state transform in
        // `run_prepare` splits a disjunctive action `X` into per-arm
        // generators `X#d0..X#dn` and registers ONLY the arms in the action
        // bytecode map (the monolithic `X` has no single-successor next-state
        // form). Each arm was planned independently above. Register the
        // arm-union under the base name `X` — mirroring
        // `try_plan_split_disjunction_action_entry`'s union-of-sub-actions
        // registration — so per-instance dispatch, coverage accounting, and
        // fused-level eligibility all see `X` as natively covered with
        // union-of-arms successor semantics (the BFS invokes every expansion
        // key and unions the successors; union exactness is established by
        // the split itself). Fail-closed: registered only when every arm of a
        // complete `0..n` group planned successfully.
        Self::register_run_prepare_split_arm_unions(
            action_bytecodes,
            &action_compile_tasks,
            &mut inner_exists_expansion_keys,
            &mut inner_exists_expansion_proofs,
        );

        // Part of #4270: compile one specialized function per BindingSpec.
        // The base bytecode is fetched from `action_bytecodes`; the binding
        // formal values are baked with typed scalar constants so model values
        // and strings keep their provenance for trust-ir/trust_cg proof paths.
        // Specialized functions are
        // inserted under `tla_jit_abi::specialized_key(name, &values)` so the
        // dispatcher can look them up with the same key it uses for other
        // native dispatch paths. Specializations are enabled by default; users can
        // set `TY_TRUST_CG_EXISTS=0` to force interpreter fallback for this path.
        if exists_enabled {
            let key_collisions = Self::typed_specialization_raw_key_collisions(specializations);
            // Alias keys that successfully reached planning. A shadowed raw split
            // form whose alias is absent here is "orphaned" and gets compiled
            // directly below so the action keeps native coverage instead of
            // falling back to the interpreter on a single alias-plan failure.
            let mut planned_specialization_keys: FxHashSet<String> = FxHashSet::default();
            for spec in specializations {
                let key = spec.binding_key.clone();
                if key_collisions.contains(&key) {
                    eprintln!(
                        "[trust-cg] specialization '{key}': dispatch key collides across distinct typed BindingSpec literals, skipping",
                    );
                    stats.actions_failed += 1;
                    continue;
                }

                let base = match action_bytecodes.get(spec.action_name.as_str()) {
                    Some(f) => *f,
                    None => {
                        eprintln!(
                            "[trust-cg] specialization '{key}': base action '{}' not in bytecode map, skipping",
                            spec.action_name,
                        );
                        stats.actions_failed += 1;
                        continue;
                    }
                };

                let source_const_pool = chunk_shared
                    .as_deref()
                    .map(|chunk| &chunk.constants)
                    .or(const_pool_shared.as_deref());
                if source_const_pool.is_none() && Self::bytecode_function_uses_const_pool(base) {
                    eprintln!(
                        "[trust-cg] specialization '{key}': base action '{}' uses constant-pool operands but no source constant pool/chunk was provided, skipping",
                        spec.action_name,
                    );
                    stats.actions_failed += 1;
                    continue;
                }

                let formal_value_literals =
                    match Self::specialization_formal_value_literals(spec, base.arity, &key) {
                        Ok(values) => values,
                        Err(message) => {
                            eprintln!("{message}, skipping");
                            stats.actions_failed += 1;
                            continue;
                        }
                    };

                let mut specialized_constants = source_const_pool.cloned().unwrap_or_default();
                let specialized = match tla_tir::bytecode::specialize_bytecode_function_with_values(
                    base,
                    formal_value_literals,
                    &key,
                    &mut specialized_constants,
                ) {
                    Ok(values) => values,
                    Err(message) => {
                        eprintln!("{message}, skipping");
                        stats.actions_failed += 1;
                        continue;
                    }
                };
                let specialized_const_pool = Arc::new(specialized_constants.clone());
                let specialized_chunk = chunk_shared.as_deref().map(|chunk| {
                    let mut chunk = chunk.clone();
                    chunk.constants = specialized_constants;
                    Arc::new(chunk)
                });

                if specialized.arity != 0 {
                    // Defensive: specialize_bytecode_function is required to
                    // return an arity-0 function. Any drift is a contract bug.
                    eprintln!(
                        "[trust-cg] specialization '{key}': unexpected arity {}, skipping",
                        specialized.arity,
                    );
                    stats.actions_failed += 1;
                    continue;
                }

                let tasks_before = action_compile_tasks.len();
                // The specialized chunk shares the source chunk's functions and
                // its constant pool only differs by appended entries, so the
                // shared precomputed callee shapes remain exact for it.
                let specialized_chunk_shapes = specialized_chunk
                    .is_some()
                    .then(|| chunk_shared.as_deref().map(&chunk_callee_shapes))
                    .flatten();
                Self::plan_next_state_action_entry(
                    &key,
                    &specialized,
                    state_layout_shared.clone(),
                    callout_opt_level,
                    Some(specialized_const_pool),
                    specialized_chunk,
                    specialized_chunk_shapes,
                    &mut action_compile_tasks,
                    &mut inner_exists_expansion_keys,
                    &mut inner_exists_expansion_proofs,
                    &mut stats,
                    &spec.binding_values,
                    &spec.formal_values,
                    scalarize_env,
                );
                // The alias produced native dispatch (a direct callout or
                // inner-EXISTS expanded callouts) only when planning emitted at
                // least one compile task. A no-op means the alias failed closed,
                // leaving the shadowed raw form as the only native candidate.
                if action_compile_tasks.len() > tasks_before {
                    planned_specialization_keys.insert(key);
                }
            }

            // Orphaned shadowed raw split forms: the alias specialization that
            // shadowed them failed to plan, so compile the raw arity-0 form
            // directly. This is sound — the raw synthetic op already has its
            // bindings baked as typed literal expressions (model values stay
            // identifiers, strings stay strings), and dispatch only consults the
            // raw key when the alias key is absent from the cache.
            for (raw_key, func, alias_key) in &deferred_shadowed_raw {
                if planned_specialization_keys.contains(alias_key.as_str()) {
                    continue;
                }
                if func.arity != 0 {
                    continue;
                }
                eprintln!(
                    "[trust-cg] alias '{alias_key}' failed to plan; compiling shadowed raw split action '{raw_key}' directly as fallback",
                );
                stats.native_action_callouts_skipped_shadowed = stats
                    .native_action_callouts_skipped_shadowed
                    .saturating_sub(1);
                Self::plan_next_state_action_entry(
                    raw_key,
                    func,
                    state_layout_shared.clone(),
                    callout_opt_level,
                    const_pool_shared.clone(),
                    chunk_shared.clone(),
                    chunk_shared.as_deref().map(&chunk_callee_shapes),
                    &mut action_compile_tasks,
                    &mut inner_exists_expansion_keys,
                    &mut inner_exists_expansion_proofs,
                    &mut stats,
                    &[],
                    &[],
                    scalarize_env,
                );
            }
        }
        stats.native_action_callout_planning_ms =
            action_planning_start.elapsed().as_millis() as u64;

        let action_task_count = action_compile_tasks.len();
        stats.native_action_callouts_planned = action_task_count;
        let action_compile_jobs = trust_cg_native_callout_compile_jobs(action_task_count);
        if action_task_count > 1 && action_compile_jobs > 1 {
            eprintln!(
                "[trust-cg] compiling {action_task_count} native action callout(s) with {action_compile_jobs} job(s)",
            );
        }
        let action_compile_start = std::time::Instant::now();
        let (action_compile_outcomes, action_batch_stats) =
            Self::compile_next_state_action_tasks_with_caller_identity(
                action_compile_tasks,
                action_compile_jobs,
                caller_identity,
            );
        stats.native_action_callout_compile_ms = action_compile_start.elapsed().as_millis() as u64;
        stats.native_action_callout_batch = action_batch_stats;
        stats.native_action_callouts_compiled = action_compile_outcomes
            .iter()
            .filter(|outcome| outcome.is_compiled())
            .count();
        for outcome in Self::dedupe_action_compile_outcomes(action_compile_outcomes) {
            Self::record_action_compile_outcome(
                outcome,
                &mut next_state_fns,
                &mut next_state_loop_fns,
                &mut native_action_entries,
                &mut libraries,
                &mut stats,
            );
        }
        telemetry_eprintln!(
            "[trust-cg] native action callouts: planned={} compiled={} skipped_shadowed={}",
            stats.native_action_callouts_planned,
            stats.native_action_callouts_compiled,
            stats.native_action_callouts_skipped_shadowed,
        );

        if !action_bytecodes.is_empty() && next_state_fns.is_empty() {
            let first_failure = stats
                .first_action_failure
                .as_deref()
                .unwrap_or("no native action failure recorded");
            eprintln!(
                "[trust-cg] zero native action coverage after action compilation; skipping invariant/state-constraint native compilation and compiled-BFS/native-fused setup for this run (first failure: {first_failure})",
            );
            stats.total_compile_ms = start.elapsed().as_millis() as u64;
            stats.maybe_log_native_cache_build_profile();
            stats.dump_native_admission_failures();
            let cache = TrustCgNativeCache {
                next_state_fns,
                next_state_loop_fns,
                inner_exists_expansion_keys,
                inner_exists_expansion_proofs,
                invariant_fns,
                state_constraint_fns,
                implied_action_fns,
                native_action_entries,
                native_invariant_entries,
                native_state_constraint_entries,
                native_implied_action_entries,
                state_var_count,
                opt_level: callout_opt_level,
                _libraries: libraries,
            };
            TrustCgNativeCalloutSelftest::log_cache_build_without_sample(&cache);
            return (cache, stats);
        }

        // Compile each invariant function.
        // Use Option to maintain index alignment: invariant_fns[i] always
        // corresponds to spec invariant i, even when compilation fails.
        // This fixes #4197 where failed compilations caused index misalignment.
        let invariant_compile_start = std::time::Instant::now();
        for (idx, func) in invariant_bytecodes.iter().enumerate() {
            let Some(func) = *func else {
                stats.invariants_failed += 1;
                eprintln!(
                    "[trust-cg] missing bytecode for invariant {idx}; compiled BFS will be ineligible"
                );
                invariant_fns.push(None);
                native_invariant_entries.push(None);
                continue;
            };
            let inv_name = format!("trust_cg_inv_{idx}");
            match Self::compile_invariant_func_with_trust_ir_proof_facts(
                NativeEntrypointRole::Invariant,
                &inv_name,
                func,
                state_layout,
                callout_opt_level,
                invariant_const_pool,
                invariant_chunk,
            ) {
                Ok((fn_ptr, lib, symbol_name, trust_ir_proof_facts)) => {
                    invariant_fns.push(Some(fn_ptr));
                    native_invariant_entries.push(Some(TrustCgNativeInvariantEntry {
                        library: lib.clone(),
                        symbol_name: symbol_name.clone(),
                    }));
                    libraries.push(lib);
                    stats.invariants_compiled += 1;
                    stats.record_trust_ir_proof_facts(trust_ir_proof_facts);
                    telemetry_eprintln!(
                        "[trust-cg] compiled invariant {idx} ({symbol_name}); eligible for invariant-checking native fused level",
                    );
                }
                Err(e) => {
                    stats.invariants_failed += 1;
                    eprintln!("[trust-cg] failed to compile invariant {idx}: {}", e);
                    invariant_fns.push(None);
                    native_invariant_entries.push(None);
                }
            }
        }
        stats.native_invariant_callout_compile_ms =
            invariant_compile_start.elapsed().as_millis() as u64;

        // Compile each state-constraint predicate separately from invariants.
        // The native ABI is identical to invariants, but the fused BFS loop
        // must use these as successor-pruning predicates, not safety checks.
        let state_constraint_compile_start = std::time::Instant::now();
        for (idx, func) in state_constraint_bytecodes.iter().enumerate() {
            let Some(func) = *func else {
                stats.state_constraints_failed += 1;
                eprintln!(
                    "[trust-cg] missing bytecode for state constraint {idx}; constrained native fused BFS will be ineligible"
                );
                state_constraint_fns.push(None);
                native_state_constraint_entries.push(None);
                continue;
            };
            let constraint_name = format!("trust_cg_state_constraint_{idx}");
            match Self::compile_invariant_func_with_trust_ir_proof_facts(
                NativeEntrypointRole::StateConstraint,
                &constraint_name,
                func,
                state_layout,
                callout_opt_level,
                state_constraint_const_pool,
                state_constraint_chunk,
            ) {
                Ok((fn_ptr, lib, symbol_name, trust_ir_proof_facts)) => {
                    state_constraint_fns.push(Some(fn_ptr));
                    native_state_constraint_entries.push(Some(TrustCgNativeInvariantEntry {
                        library: lib.clone(),
                        symbol_name: symbol_name.clone(),
                    }));
                    libraries.push(lib);
                    stats.state_constraints_compiled += 1;
                    stats.record_trust_ir_proof_facts(trust_ir_proof_facts);
                    telemetry_eprintln!(
                        "[trust-cg] compiled state constraint {idx} ({symbol_name}); eligible for native fused constraint pruning when backend hook is available",
                    );
                }
                Err(e) => {
                    stats.state_constraints_failed += 1;
                    eprintln!("[trust-cg] failed to compile state constraint {idx}: {}", e);
                    state_constraint_fns.push(None);
                    native_state_constraint_entries.push(None);
                }
            }
        }
        stats.native_state_constraint_callout_compile_ms =
            state_constraint_compile_start.elapsed().as_millis() as u64;

        stats.total_compile_ms = start.elapsed().as_millis() as u64;
        stats.maybe_log_native_cache_build_profile();
        stats.dump_native_admission_failures();

        let cache = TrustCgNativeCache {
            next_state_fns,
            next_state_loop_fns,
            inner_exists_expansion_keys,
            inner_exists_expansion_proofs,
            invariant_fns,
            state_constraint_fns,
            implied_action_fns,
            native_action_entries,
            native_invariant_entries,
            native_state_constraint_entries,
            native_implied_action_entries,
            state_var_count,
            opt_level: callout_opt_level,
            _libraries: libraries,
        };

        TrustCgNativeCalloutSelftest::log_cache_build_without_sample(&cache);

        (cache, stats)
    }

    /// Compile a single next-state action through the trust-codegen native pipeline.
    ///
    /// Pipeline: BytecodeFunction -> trust-ir -> trust-codegen JIT -> NativeLibrary.
    ///
    /// When `chunk` is `Some`, compilation routes through
    /// `compile_entry_next_state_native_with_chunk`, which drains pending
    /// `Call` opcodes so callee function bodies are emitted alongside the
    /// entry. This is required for any action that invokes another user-
    /// defined operator; without it, trust-codegen emits unresolved `__func_N`
    /// symbols. When `chunk` is `None`, falls back to the single-function
    /// path (legacy callers + JIT specialization of arity-0 functions that
    /// do not appear in the chunk's `functions` table).
    ///
    /// Part of #4280 Gap C.
    #[allow(dead_code)] // off-by-default JIT compile/eval machinery (TY_JIT, #4035); kept, currently unwired
    fn compile_next_state_action(
        action_name: &str,
        func: &tla_tir::bytecode::BytecodeFunction,
        state_layout: Option<&tla_jit_abi::StateLayout>,
        opt_level: tla_trust_cg::OptLevel,
        const_pool: Option<&tla_tir::bytecode::ConstantPool>,
        chunk: Option<&tla_tir::bytecode::BytecodeChunk>,
    ) -> Result<(NativeNextStateFn, tla_trust_cg::NativeLibrary, String), tla_trust_cg::TrustCgError>
    {
        let (fn_ptr, library, symbol_name, _) =
            Self::compile_next_state_action_with_trust_ir_proof_facts(
                action_name,
                func,
                state_layout,
                opt_level,
                const_pool,
                chunk,
                None,
            )?;
        Ok((fn_ptr, library, symbol_name))
    }

    #[allow(dead_code)] // off-by-default JIT compile/eval machinery (TY_JIT, #4035); kept, currently unwired
    fn compile_next_state_action_with_action_local_set_domain_proof(
        action_name: &str,
        func: &tla_tir::bytecode::BytecodeFunction,
        state_layout: Option<&tla_jit_abi::StateLayout>,
        opt_level: tla_trust_cg::OptLevel,
        const_pool: Option<&tla_tir::bytecode::ConstantPool>,
        chunk: Option<&tla_tir::bytecode::BytecodeChunk>,
        action_local_set_domain_proof: Option<&tla_ir::lower::ActionLocalSetDomainProof>,
    ) -> Result<(NativeNextStateFn, tla_trust_cg::NativeLibrary, String), tla_trust_cg::TrustCgError>
    {
        let (fn_ptr, library, symbol_name, _) =
            Self::compile_next_state_action_with_trust_ir_proof_facts(
                action_name,
                func,
                state_layout,
                opt_level,
                const_pool,
                chunk,
                action_local_set_domain_proof,
            )?;
        Ok((fn_ptr, library, symbol_name))
    }

    fn lower_next_state_action_with_trust_ir_proof_facts(
        action_name: &str,
        func: &tla_tir::bytecode::BytecodeFunction,
        state_layout: Option<&tla_jit_abi::StateLayout>,
        const_pool: Option<&tla_tir::bytecode::ConstantPool>,
        chunk: Option<&tla_tir::bytecode::BytecodeChunk>,
        action_local_set_domain_proof: Option<&tla_ir::lower::ActionLocalSetDomainProof>,
    ) -> Result<
        (
            trust_ir::Module,
            String,
            tla_ir::annotations::NativeProofAnnotationSummary,
        ),
        tla_trust_cg::TrustCgError,
    > {
        Self::lower_next_state_action_with_trust_ir_proof_facts_and_callee_shapes(
            action_name,
            func,
            state_layout,
            const_pool,
            chunk,
            None,
            action_local_set_domain_proof,
            false,
        )
    }

    /// Same as [`Self::lower_next_state_action_with_trust_ir_proof_facts`],
    /// reusing precomputed chunk-wide callee return shapes when provided
    /// (identical lowering decisions; skips the per-task chunk inference).
    #[allow(clippy::too_many_arguments)]
    fn lower_next_state_action_with_trust_ir_proof_facts_and_callee_shapes(
        action_name: &str,
        func: &tla_tir::bytecode::BytecodeFunction,
        state_layout: Option<&tla_jit_abi::StateLayout>,
        const_pool: Option<&tla_tir::bytecode::ConstantPool>,
        chunk: Option<&tla_tir::bytecode::BytecodeChunk>,
        chunk_callee_shapes: Option<&tla_ir::lower::ChunkCalleeReturnShapes>,
        action_local_set_domain_proof: Option<&tla_ir::lower::ActionLocalSetDomainProof>,
        next_state_loop: bool,
    ) -> Result<
        (
            trust_ir::Module,
            String,
            tla_ir::annotations::NativeProofAnnotationSummary,
        ),
        tla_trust_cg::TrustCgError,
    > {
        // Build a stable, collision-free LLVM function name for the native entrypoint.
        let safe_name = native_entrypoint_symbol_name(NativeEntrypointRole::Action, action_name);
        if trust_cg_dump_filter_matches(TRUST_CG_DUMP_ACTION_BYTECODE_ENV, action_name)
            || trust_cg_dump_filter_matches(TRUST_CG_DUMP_ACTION_BYTECODE_ENV, &safe_name)
        {
            eprintln!("[trust-cg] action {action_name} bytecode: {func:#?}");
        }

        // Route B: multi-successor record-set kernel. This emits a real
        // `tla_jit_abi::NextStateLoopFn` module (param#2 is a
        // `*mut NextStateLoopSink`), distinct from the single-successor
        // `lower_next_state*` lowering below.
        if next_state_loop {
            let trust_ir_module = tla_ir::lower::lower_next_state_loop_scaffold(
                func,
                &safe_name,
                const_pool,
                state_layout,
            )?;
            let trust_ir_proof_facts =
                tla_ir::annotations::summarize_native_proof_annotations(&trust_ir_module);
            audit_lowered_action_tla_externs(action_name, &trust_ir_module);
            return Ok((trust_ir_module, safe_name, trust_ir_proof_facts));
        }

        // Compile BytecodeFunction -> trust-ir -> native code via trust-codegen JIT pipeline.
        let trust_ir_module = if let Some(proof) = action_local_set_domain_proof {
            let proofs = std::slice::from_ref(proof);
            if let Some(((chunk, layout), shapes)) =
                chunk.zip(state_layout).zip(chunk_callee_shapes)
            {
                tla_ir::lower::lower_entry_next_state_with_chunk(
                    func,
                    chunk,
                    &safe_name,
                    tla_ir::lower::LoweringOptions::new()
                        .with_layout(layout)
                        .with_action_local_set_domain_proofs(proofs)
                        .with_callee_shapes(shapes),
                )?
            } else if let Some((chunk, layout)) = chunk.zip(state_layout) {
                tla_ir::lower::lower_entry_next_state_with_chunk(
                    func,
                    chunk,
                    &safe_name,
                    tla_ir::lower::LoweringOptions::new()
                        .with_layout(layout)
                        .with_action_local_set_domain_proofs(proofs),
                )?
            } else if let Some((pool, layout)) = const_pool.zip(state_layout) {
                tla_ir::lower::lower_next_state(
                    func,
                    &safe_name,
                    tla_ir::lower::LoweringOptions::new()
                        .with_constants(pool)
                        .with_layout(layout)
                        .with_action_local_set_domain_proofs(proofs),
                )?
            } else {
                return Err(tla_trust_cg::TrustCgError::from(tla_ir::TrustIrError::UnsupportedOpcode(
                    "action-local set-domain proof requires constant-pool and state-layout lowering"
                        .to_owned(),
                )));
            }
        } else if let Some(((chunk, layout), shapes)) =
            chunk.zip(state_layout).zip(chunk_callee_shapes)
        {
            tla_ir::lower::lower_entry_next_state_with_chunk(
                func,
                chunk,
                &safe_name,
                tla_ir::lower::LoweringOptions::new()
                    .with_layout(layout)
                    .with_action_local_set_domain_proofs(&[])
                    .with_callee_shapes(shapes),
            )?
        } else if let Some((chunk, layout)) = chunk.zip(state_layout) {
            tla_ir::lower::lower_entry_next_state_with_chunk(
                func,
                chunk,
                &safe_name,
                tla_ir::lower::LoweringOptions::new().with_layout(layout),
            )?
        } else if let Some(chunk) = chunk {
            // Chunk-aware path: drains `Call` callees from chunk.functions so
            // they are emitted alongside the entry. Fixes #4280 Gap C.
            tla_ir::lower::lower_entry_next_state_with_chunk(
                func,
                chunk,
                &safe_name,
                tla_ir::lower::LoweringOptions::new(),
            )?
        } else if let Some((pool, layout)) = const_pool.zip(state_layout) {
            tla_ir::lower::lower_next_state(
                func,
                &safe_name,
                tla_ir::lower::LoweringOptions::new()
                    .with_constants(pool)
                    .with_layout(layout),
            )?
        } else if let Some(pool) = const_pool {
            tla_ir::lower::lower_next_state(
                func,
                &safe_name,
                tla_ir::lower::LoweringOptions::new().with_constants(pool),
            )?
        } else {
            tla_ir::lower::lower_next_state(
                func,
                &safe_name,
                tla_ir::lower::LoweringOptions::new(),
            )?
        };
        let trust_ir_proof_facts =
            tla_ir::annotations::summarize_native_proof_annotations(&trust_ir_module);
        audit_lowered_action_tla_externs(action_name, &trust_ir_module);
        Ok((trust_ir_module, safe_name, trust_ir_proof_facts))
    }

    fn compile_next_state_action_with_trust_ir_proof_facts(
        action_name: &str,
        func: &tla_tir::bytecode::BytecodeFunction,
        state_layout: Option<&tla_jit_abi::StateLayout>,
        opt_level: tla_trust_cg::OptLevel,
        const_pool: Option<&tla_tir::bytecode::ConstantPool>,
        chunk: Option<&tla_tir::bytecode::BytecodeChunk>,
        action_local_set_domain_proof: Option<&tla_ir::lower::ActionLocalSetDomainProof>,
    ) -> Result<
        (
            NativeNextStateFn,
            tla_trust_cg::NativeLibrary,
            String,
            tla_ir::annotations::NativeProofAnnotationSummary,
        ),
        tla_trust_cg::TrustCgError,
    > {
        let (trust_ir_module, safe_name, trust_ir_proof_facts) =
            Self::lower_next_state_action_with_trust_ir_proof_facts(
                action_name,
                func,
                state_layout,
                const_pool,
                chunk,
                action_local_set_domain_proof,
            )?;
        let lib = tla_trust_cg::compile_module_native(&trust_ir_module, opt_level)?;

        // Look up the compiled function symbol.
        // SAFETY: We control the compilation pipeline and know the exact signature.
        let fn_ptr = unsafe {
            let raw = lib.get_symbol(&safe_name)?;
            std::mem::transmute::<*mut std::ffi::c_void, NativeNextStateFn>(raw)
        };

        Ok((fn_ptr, lib, safe_name, trust_ir_proof_facts))
    }

    /// Compile a single invariant function through the trust-codegen native pipeline.
    ///
    /// Pipeline: BytecodeFunction -> trust-ir -> trust-codegen JIT -> NativeLibrary.
    ///
    /// `chunk` semantics match [`compile_next_state_action`]. Part of #4280 Gap C.
    #[allow(dead_code)] // off-by-default JIT compile/eval machinery (TY_JIT, #4035); kept, currently unwired
    fn compile_invariant_func(
        inv_name: &str,
        func: &tla_tir::bytecode::BytecodeFunction,
        state_layout: Option<&tla_jit_abi::StateLayout>,
        opt_level: tla_trust_cg::OptLevel,
        const_pool: Option<&tla_tir::bytecode::ConstantPool>,
        chunk: Option<&tla_tir::bytecode::BytecodeChunk>,
    ) -> Result<(NativeInvariantFn, tla_trust_cg::NativeLibrary, String), tla_trust_cg::TrustCgError>
    {
        let (fn_ptr, library, symbol_name, _) =
            Self::compile_invariant_func_with_trust_ir_proof_facts(
                NativeEntrypointRole::Invariant,
                inv_name,
                func,
                state_layout,
                opt_level,
                const_pool,
                chunk,
            )?;
        Ok((fn_ptr, library, symbol_name))
    }

    fn compile_invariant_func_with_trust_ir_proof_facts(
        entry_role: NativeEntrypointRole,
        inv_name: &str,
        func: &tla_tir::bytecode::BytecodeFunction,
        state_layout: Option<&tla_jit_abi::StateLayout>,
        opt_level: tla_trust_cg::OptLevel,
        const_pool: Option<&tla_tir::bytecode::ConstantPool>,
        chunk: Option<&tla_tir::bytecode::BytecodeChunk>,
    ) -> Result<
        (
            NativeInvariantFn,
            tla_trust_cg::NativeLibrary,
            String,
            tla_ir::annotations::NativeProofAnnotationSummary,
        ),
        tla_trust_cg::TrustCgError,
    > {
        let safe_name = native_entrypoint_symbol_name(entry_role, inv_name);

        if std::env::var("TY_TRUST_CG_DUMP_INVARIANT_BYTECODE").as_deref() == Ok("1") {
            eprintln!("[trust-cg] invariant {inv_name} bytecode: {func:#?}");
        }

        // Compile BytecodeFunction -> trust-ir -> native code via trust-codegen JIT pipeline.
        let trust_ir_module = if let Some((chunk, layout)) = chunk.zip(state_layout) {
            tla_ir::lower::lower_entry_invariant_with_chunk(
                func,
                chunk,
                &safe_name,
                tla_ir::lower::LoweringOptions::new().with_layout(layout),
            )?
        } else if let Some(chunk) = chunk {
            // Chunk-aware path: emit callee bodies alongside the entry. #4280 Gap C.
            tla_ir::lower::lower_entry_invariant_with_chunk(
                func,
                chunk,
                &safe_name,
                tla_ir::lower::LoweringOptions::new(),
            )?
        } else if let Some((pool, layout)) = const_pool.zip(state_layout) {
            tla_ir::lower::lower_invariant(
                func,
                &safe_name,
                tla_ir::lower::LoweringOptions::new()
                    .with_constants(pool)
                    .with_layout(layout),
            )?
        } else if let Some(pool) = const_pool {
            tla_ir::lower::lower_invariant(
                func,
                &safe_name,
                tla_ir::lower::LoweringOptions::new().with_constants(pool),
            )?
        } else {
            tla_ir::lower::lower_invariant(func, &safe_name, tla_ir::lower::LoweringOptions::new())?
        };
        let trust_ir_proof_facts =
            tla_ir::annotations::summarize_native_proof_annotations(&trust_ir_module);
        let lib = tla_trust_cg::compile_module_native(&trust_ir_module, opt_level)?;

        // SAFETY: We control the compilation pipeline and know the exact signature.
        let fn_ptr = unsafe {
            let raw = lib.get_symbol(&safe_name)?;
            std::mem::transmute::<*mut std::ffi::c_void, NativeInvariantFn>(raw)
        };

        Ok((fn_ptr, lib, safe_name, trust_ir_proof_facts))
    }

    /// Check whether trust-codegen native compilation is available.
    ///
    /// Returns `true` when the `native` feature is compiled into `tla-trust_cg`.
    /// When `false`, [`TrustCgNativeCache::build`] will fail to compile any functions.
    pub(in crate::check) fn is_available() -> bool {
        tla_trust_cg::is_native_available()
    }

    /// Whether the trust-codegen BFS path is active for this run.
    ///
    /// OFF by default: the interpreter is the default execution engine. The
    /// native compiled-flat BFS path is an in-flight opcode-rewrite checkpoint
    /// that can under-generate successors and trip the canonical-payload
    /// soundness guard (`canonical_payload_mismatch`), so it stays opt-in until
    /// that rewrite is sound. Because a wrong reachable-state count is the
    /// dominant scoring risk, the default must be the engine we trust. Opt in
    /// explicitly with any engine var set to a truthy value:
    /// - `TY_TRUST_CG=1` — enable trust-cg native.
    /// - `TY_trust_cg=1` — legacy alias, same effect.
    /// - `TY_TRUST_CG_BFS=1` — BFS-scoped alias (epic #4251 Stream 6).
    ///
    /// An explicit falsey value (`0`/`false`/`off`/`no`) on any alias forces the
    /// interpreter and overrides a truthy value on another alias. The JIT is
    /// always compiled in; this is engine selection, not a backend on/off switch.
    pub(in crate::check) fn is_enabled(structurally_vetoed: bool) -> bool {
        // Runtime structural veto: the AUTO engine-selector (see
        // `trust_cg_auto_select_enabled`) may decide before paying the native
        // compile cost that native will not help this run, and latch the path
        // off for the remainder of the process. Once latched, every trust-cg
        // gate (`should_use_trust_cg`) returns false so the plain interpreter
        // runs without action splitting, native bytecode compilation, or the
        // native fused/per-action dispatch overhead.
        if structurally_vetoed {
            return false;
        }
        // The set-once process-global env snapshot (installed only by the CLI binary)
        // carries the synthesized AUTO/native decision and the exact 3-alias trichotomy.
        // Library/test callers never install it, so they fall through to the legacy env
        // path below — keeping every `EnvVarGuard` matrix byte-identical.
        tla_backend::global_overlay()
            .map(tla_backend::EngineEnvOverlay::trust_cg_enabled)
            .unwrap_or_else(|| {
                if trust_cg_env_flag_disabled("TY_TRUST_CG")
                    || trust_cg_env_flag_disabled("TY_trust_cg")
                    || trust_cg_env_flag_disabled("TY_TRUST_CG_BFS")
                {
                    return false;
                }
                trust_cg_env_flag_enabled("TY_TRUST_CG")
                    || trust_cg_env_flag_enabled("TY_trust_cg")
                    || trust_cg_env_flag_enabled("TY_TRUST_CG_BFS")
            })
    }

    /// Number of successfully compiled action functions.
    pub(in crate::check) fn action_count(&self) -> usize {
        self.next_state_fns.len()
    }

    /// Whether at least one next-state action was successfully compiled.
    ///
    /// Used by the fingerprint mixed-mode guard in
    /// `try_activate_compiled_fingerprinting`: when any compiled action can
    /// produce successors during BFS, the fingerprint pipeline must stay on a
    /// single hash domain so that compiled-emitted and interpreter-emitted
    /// successors of the same logical state dedup into the same `Fingerprint`.
    ///
    /// Part of #4319 (trust_cg fingerprint unification) Phase 0.
    #[inline]
    pub(in crate::check) fn has_any_compiled_action(&self) -> bool {
        !self.next_state_fns.is_empty()
    }

    /// Number of successfully compiled invariant functions.
    pub(in crate::check) fn invariant_count(&self) -> usize {
        self.invariant_fns.iter().filter(|f| f.is_some()).count()
    }

    /// Number of invariant slots tracked by this cache.
    ///
    /// This is the total configured-invariant count observed at build time,
    /// including `None` entries for bytecode/lowering failures. It is distinct
    /// from [`Self::invariant_count`], which only counts successful native
    /// compiles.
    pub(in crate::check) fn invariant_slot_count(&self) -> usize {
        self.invariant_fns.len()
    }

    /// Number of successfully compiled state-constraint functions.
    pub(in crate::check) fn state_constraint_count(&self) -> usize {
        self.state_constraint_fns
            .iter()
            .filter(|f| f.is_some())
            .count()
    }

    /// Number of state-constraint slots tracked by this cache.
    pub(in crate::check) fn state_constraint_slot_count(&self) -> usize {
        self.state_constraint_fns.len()
    }

    /// Whether every configured invariant has a compiled trust-codegen function.
    ///
    /// The length check prevents treating a shorter partial vector as full
    /// coverage. Each slot remains parallel to `config.invariants`, so index
    /// `i` in failures and reports maps back to invariant `i` in the spec.
    pub(in crate::check) fn has_all_invariants(&self, invariant_count: usize) -> bool {
        self.invariant_fns.len() == invariant_count
            && self.invariant_fns.iter().all(Option::is_some)
    }

    /// Names of configured invariants that lack compiled trust-codegen function pointers.
    pub(in crate::check) fn missing_invariant_names(
        &self,
        invariant_names: &[String],
    ) -> Vec<String> {
        invariant_names
            .iter()
            .enumerate()
            .filter_map(|(idx, name)| {
                let missing_fn = self.invariant_fns.get(idx).map_or(true, Option::is_none);
                missing_fn.then(|| name.clone())
            })
            .collect()
    }

    /// Whether every configured state constraint has a compiled trust-codegen function.
    pub(in crate::check) fn has_all_state_constraints(&self, constraint_count: usize) -> bool {
        self.state_constraint_fns.len() == constraint_count
            && self.state_constraint_fns.iter().all(Option::is_some)
    }

    /// Names of configured state constraints that lack compiled native entries.
    pub(in crate::check) fn missing_state_constraint_names(
        &self,
        constraint_names: &[String],
    ) -> Vec<String> {
        constraint_names
            .iter()
            .enumerate()
            .filter_map(|(idx, name)| {
                let missing_fn = self
                    .state_constraint_fns
                    .get(idx)
                    .map_or(true, Option::is_none);
                let missing_native = self
                    .native_state_constraint_entries
                    .get(idx)
                    .map_or(true, Option::is_none);
                (missing_fn || missing_native).then(|| name.clone())
            })
            .collect()
    }

    /// Number of state variables.
    pub(in crate::check) fn state_var_count(&self) -> usize {
        self.state_var_count
    }

    pub(in crate::check) fn compile_implied_actions_for_cache(
        &mut self,
        implied_action_names: &[String],
        implied_action_bytecodes: &[Option<&tla_tir::bytecode::BytecodeFunction>],
        state_layout: Option<&tla_jit_abi::StateLayout>,
        action_const_pool: Option<&tla_tir::bytecode::ConstantPool>,
        implied_const_pool: Option<&tla_tir::bytecode::ConstantPool>,
        implied_chunk: Option<&tla_tir::bytecode::BytecodeChunk>,
    ) {
        self.implied_action_fns.clear();
        self.native_implied_action_entries.clear();
        for (idx, func) in implied_action_bytecodes.iter().enumerate() {
            let Some(func) = *func else {
                self.implied_action_fns.push(None);
                self.native_implied_action_entries.push(None);
                continue;
            };
            let name = implied_action_names
                .get(idx)
                .cloned()
                .unwrap_or_else(|| format!("__ty_native_implied_action_{idx}"));
            match Self::compile_next_state_action_with_trust_ir_proof_facts(
                &name,
                func,
                state_layout,
                self.opt_level,
                implied_const_pool.or(action_const_pool),
                implied_chunk,
                None,
            ) {
                Ok((fn_ptr, lib, symbol_name, _proofs)) => {
                    self.implied_action_fns.push(Some(fn_ptr));
                    self.native_implied_action_entries
                        .push(Some(TrustCgNativeInvariantEntry {
                            library: lib.clone(),
                            symbol_name: symbol_name.clone(),
                        }));
                    self._libraries.push(lib);
                    eprintln!(
                        "[trust_cg] compiled implied action {idx} ({symbol_name}); eligible for native fused action-property checking",
                    );
                }
                Err(err) => {
                    eprintln!("[trust_cg] failed to compile implied action {idx}: {err}");
                    self.implied_action_fns.push(None);
                    self.native_implied_action_entries.push(None);
                }
            }
        }
    }

    #[allow(dead_code)] // off-by-default JIT compile/eval machinery (TY_JIT, #4035); kept, currently unwired
    pub(in crate::check) fn has_all_implied_actions(&self, implied_count: usize) -> bool {
        self.implied_action_fns.len() == implied_count
            && self.implied_action_fns.iter().all(Option::is_some)
    }

    /// Check if a specific action is compiled.
    pub(in crate::check) fn contains_action(&self, name: &str) -> bool {
        self.next_state_fns.contains_key(name)
    }

    /// Like [`Self::contains_action`] but ALSO counts multi-successor
    /// record-set kernels (`next_state_loop_fns`, the `NextStateLoopFn` sink
    /// ABI). Use ONLY where the consumer handles both ABIs — the fused-level
    /// eligibility check and coverage reporting, whose action resolution
    /// (`resolve_native_actions_ordered`) tags loop actions `with_is_loop` and
    /// dispatches them via the sink call convention. Per-action single-successor
    /// callout paths MUST keep using `contains_action`: handing a loop kernel a
    /// plain `state_out` buffer as a sink pointer would corrupt memory.
    pub(in crate::check) fn contains_action_any_abi(&self, name: &str) -> bool {
        self.next_state_fns.contains_key(name) || self.next_state_loop_fns.contains_key(name)
    }

    fn entry_counter_allows_action_dispatch(&self, action_name: &str, limit: u64) -> bool {
        self.native_action_entries
            .get(action_name)
            .map(|entry| entry.library.entry_count(&entry.symbol_name))
            .map_or(false, |entry_count| {
                entry_counter_gate_allows_dispatch(entry_count, limit)
            })
    }

    /// Find compiled native action keys produced by inner-EXISTS expansion.
    ///
    /// The returned order is deterministic so fused BFS action descriptor
    /// ordering is stable across runs.
    pub(in crate::check) fn inner_exists_expansion_keys(&self, base_name: &str) -> Vec<String> {
        self.inner_exists_expansion_keys
            .get(base_name)
            .cloned()
            .unwrap_or_default()
    }

    pub(in crate::check) fn inner_exists_expansion_native_fused_safe(
        &self,
        base_name: &str,
    ) -> bool {
        let Some(keys) = self.inner_exists_expansion_keys.get(base_name) else {
            return false;
        };
        self.inner_exists_expansion_proofs
            .get(base_name)
            .is_some_and(|proof| proof.native_fused_safe(keys.len()))
    }

    /// Resolve action keys to native function pointers in caller-provided order.
    pub(in crate::check) fn resolve_actions_ordered(
        &self,
        action_keys: &[String],
    ) -> Option<Vec<NativeNextStateFn>> {
        let mut resolved = Vec::with_capacity(action_keys.len());
        for key in action_keys {
            resolved.push(*self.next_state_fns.get(key)?);
        }
        Some(resolved)
    }

    /// Resolve action keys to host-callable compiled actions across BOTH
    /// next-state ABIs, in caller-provided order.
    ///
    /// Single-successor keys resolve from `next_state_fns` exactly like
    /// [`Self::resolve_actions_ordered`]. Multi-successor record-set kernels
    /// resolve from `next_state_loop_fns` and are built via
    /// [`tla_trust_cg::TrustCgCompiledAction::new_loop`], so the prototype
    /// parent loop dispatches them through the `NextStateLoopSink` call
    /// convention — NEVER the single-successor convention (the Route B
    /// memory-corruption hazard). Owner library + exact symbol are attached
    /// from `native_action_entries` for cached-pointer republication.
    ///
    /// Use ONLY for consumers that handle both ABIs (the fused/prototype
    /// CompiledBfsLevel). `CompiledBfsStep` and the per-action callout paths
    /// stay strict single-ABI on [`Self::resolve_actions_ordered`].
    fn resolve_compiled_actions_ordered_any_abi(
        &self,
        action_keys: &[String],
    ) -> Option<Vec<tla_trust_cg::TrustCgCompiledAction>> {
        let mut resolved = Vec::with_capacity(action_keys.len());
        for (idx, key) in action_keys.iter().enumerate() {
            let descriptor = self.action_descriptor_for_key(key, idx);
            let action = if let Some(func) = self.next_state_fns.get(key) {
                tla_trust_cg::TrustCgCompiledAction::new(descriptor, *func)
            } else if let Some(loop_fn) = self.next_state_loop_fns.get(key) {
                tla_trust_cg::TrustCgCompiledAction::new_loop(descriptor, *loop_fn)
            } else {
                return None;
            };
            let action = match self.native_action_entries.get(key) {
                Some(entry) => action.with_owner_library_and_symbol(
                    entry.library.clone(),
                    entry.symbol_name.clone(),
                ),
                None => action,
            };
            resolved.push(action);
        }
        Some(resolved)
    }

    /// True when any of `action_keys` is a Route B multi-successor
    /// (`NextStateLoopFn`) kernel.
    fn any_action_is_loop(&self, action_keys: &[String]) -> bool {
        action_keys
            .iter()
            .any(|key| self.next_state_loop_fns.contains_key(key))
    }

    fn resolve_action_owner_libraries_ordered(
        &self,
        action_keys: &[String],
    ) -> Option<Vec<Option<tla_trust_cg::NativeLibrary>>> {
        let mut resolved = Vec::with_capacity(action_keys.len());
        for key in action_keys {
            let library = self
                .native_action_entries
                .get(key)
                .map(|entry| entry.library.clone());
            resolved.push(library);
        }
        Some(resolved)
    }

    fn resolve_action_symbol_names_ordered(
        &self,
        action_keys: &[String],
    ) -> Option<Vec<Option<String>>> {
        let mut resolved = Vec::with_capacity(action_keys.len());
        for key in action_keys {
            let symbol_name = self
                .native_action_entries
                .get(key)
                .map(|entry| entry.symbol_name.clone());
            resolved.push(symbol_name);
        }
        Some(resolved)
    }

    /// Declared bytecode footprint for a compiled action key:
    /// `(read_vars, write_vars)`, sorted ascending. `None` when the key has no
    /// native entry (e.g. hand-built test caches). Computed transitively over
    /// chunk callees by [`Self::action_var_access_sets`]; consumed by the
    /// hybrid per-action admission dual gate (item 4 M0-G4).
    pub(in crate::check) fn action_declared_footprint(
        &self,
        key: &str,
    ) -> Option<(&[u16], &[u16])> {
        self.native_action_entries
            .get(key)
            .map(|entry| (entry.read_vars.as_slice(), entry.write_vars.as_slice()))
    }

    /// Declared compound-read callout footprint for a compiled action key
    /// (wishlist item 4 M1): the hybrid-placeholder vars the artifact reads
    /// through `tla_hybrid_compound_*` against the parent context ty publishes
    /// around the dispatch.
    ///
    /// Empty for every artifact compiled without the callout — which is the
    /// fail-closed default, so the M1 admission gate degrades to M0's until
    /// the lowering actually emits a callout and declares its var.
    pub(in crate::check) fn action_declared_compound_read_vars(&self, key: &str) -> &[u16] {
        self.native_action_entries
            .get(key)
            .map(|entry| entry.compound_read_vars.as_slice())
            .unwrap_or(&[])
    }

    /// Whether `key` compiled as a Route B multi-successor (`NextStateLoopFn`)
    /// kernel. The hybrid flat-view dispatch is strictly single-successor ABI,
    /// so loop kernels must decline hybrid admission.
    pub(in crate::check) fn action_is_loop_kernel(&self, key: &str) -> bool {
        self.next_state_loop_fns.contains_key(key)
    }

    fn action_descriptor_for_key(
        &self,
        key: &str,
        action_idx: usize,
    ) -> tla_trust_cg::ActionDescriptor {
        let (binding_values, formal_values, read_vars, write_vars) = self
            .native_action_entries
            .get(key)
            .map(|entry| {
                (
                    entry.binding_values.clone(),
                    entry.formal_values.clone(),
                    entry.read_vars.clone(),
                    entry.write_vars.clone(),
                )
            })
            .unwrap_or_else(|| (Vec::new(), Vec::new(), Vec::new(), Vec::new()));
        tla_trust_cg::ActionDescriptor {
            name: key.to_string(),
            action_idx: action_idx as u32,
            binding_values,
            formal_values,
            read_vars,
            write_vars,
            // The compound-read callout footprint is a property of the HYBRID
            // layout an artifact was compiled against, which this cache does
            // not carry; the hybrid dispatcher derives and declares it (see
            // `ModelChecker::hybrid_declared_compound_read_vars`). Whole-state
            // descriptors have none by construction.
            compound_read_vars: Vec::new(),
        }
    }

    /// Resolve action keys to native-library entries in caller-provided order.
    pub(in crate::check) fn resolve_native_actions_ordered(
        &self,
        action_keys: &[String],
    ) -> Option<Vec<tla_trust_cg::TrustCgBfsLevelNativeAction>> {
        let mut resolved = Vec::with_capacity(action_keys.len());
        for (idx, key) in action_keys.iter().enumerate() {
            let entry = self.native_action_entries.get(key)?;
            // Route B: an action present in `next_state_loop_fns` implements the
            // multi-successor `NextStateLoopFn` ABI and must be dispatched via
            // the sink call convention (is_loop), never the single-successor
            // path. Empty by default, so is_loop is always false unless
            // TY_RECORD_SET_NATIVE=1 selected a record-set kernel.
            let is_loop = self.next_state_loop_fns.contains_key(key);
            resolved.push(
                tla_trust_cg::TrustCgBfsLevelNativeAction::new(
                    self.action_descriptor_for_key(key, idx),
                    entry.library.clone(),
                    entry.symbol_name.clone(),
                )
                .with_is_loop(is_loop),
            );
        }
        Some(resolved)
    }

    /// Resolve native-library invariant entries in stable spec-index order.
    pub(in crate::check) fn resolve_native_invariants_ordered(
        &self,
        invariant_names: &[String],
    ) -> Option<Vec<tla_trust_cg::TrustCgBfsLevelNativeInvariant>> {
        if self.native_invariant_entries.len() != invariant_names.len()
            || self.native_invariant_entries.iter().any(Option::is_none)
        {
            return None;
        }

        let mut resolved = Vec::with_capacity(invariant_names.len());
        for (idx, name) in invariant_names.iter().enumerate() {
            let entry = self.native_invariant_entries.get(idx)?.as_ref()?;
            resolved.push(tla_trust_cg::TrustCgBfsLevelNativeInvariant::new(
                tla_trust_cg::InvariantDescriptor {
                    name: name.clone(),
                    invariant_idx: idx as u32,
                },
                entry.library.clone(),
                entry.symbol_name.clone(),
            ));
        }
        Some(resolved)
    }

    /// Resolve native-library state-constraint entries in stable config order.
    pub(in crate::check) fn resolve_native_state_constraints_ordered(
        &self,
        constraint_names: &[String],
    ) -> Option<Vec<tla_trust_cg::TrustCgBfsLevelNativeStateConstraint>> {
        if self.native_state_constraint_entries.len() != constraint_names.len()
            || self
                .native_state_constraint_entries
                .iter()
                .any(Option::is_none)
        {
            return None;
        }

        let mut resolved = Vec::with_capacity(constraint_names.len());
        for (idx, name) in constraint_names.iter().enumerate() {
            let entry = self.native_state_constraint_entries.get(idx)?.as_ref()?;
            let constraint_idx = u32::try_from(idx).ok()?;
            resolved.push(tla_trust_cg::TrustCgBfsLevelNativeStateConstraint::new(
                name.clone(),
                constraint_idx,
                entry.library.clone(),
                entry.symbol_name.clone(),
            ));
        }
        Some(resolved)
    }

    pub(in crate::check) fn resolve_native_implied_actions_ordered(
        &self,
        implied_names: &[String],
    ) -> Option<Vec<tla_trust_cg::TrustCgBfsLevelNativeImpliedAction>> {
        if self.native_implied_action_entries.len() != implied_names.len()
            || self
                .native_implied_action_entries
                .iter()
                .any(Option::is_none)
        {
            return None;
        }

        let mut resolved = Vec::with_capacity(implied_names.len());
        for (idx, name) in implied_names.iter().enumerate() {
            let entry = self.native_implied_action_entries.get(idx)?.as_ref()?;
            let implied_idx = u32::try_from(idx).ok()?;
            resolved.push(tla_trust_cg::TrustCgBfsLevelNativeImpliedAction::new(
                name.clone(),
                implied_idx,
                entry.library.clone(),
                entry.symbol_name.clone(),
            ));
        }
        Some(resolved)
    }

    /// Resolve invariant function pointers in stable spec-index order.
    pub(in crate::check) fn resolve_invariants_ordered(
        &self,
        invariant_count: usize,
    ) -> Option<Vec<NativeInvariantFn>> {
        if !self.has_all_invariants(invariant_count) {
            return None;
        }
        let mut resolved = Vec::with_capacity(invariant_count);
        for slot in &self.invariant_fns {
            resolved.push((*slot)?);
        }
        Some(resolved)
    }

    fn resolve_invariant_owner_libraries_ordered(
        &self,
        invariant_count: usize,
    ) -> Option<Vec<Option<tla_trust_cg::NativeLibrary>>> {
        if !self.has_all_invariants(invariant_count) {
            return None;
        }
        let mut resolved = Vec::with_capacity(invariant_count);
        for idx in 0..invariant_count {
            let library = self
                .native_invariant_entries
                .get(idx)
                .and_then(|entry| entry.as_ref())
                .map(|entry| entry.library.clone());
            resolved.push(library);
        }
        Some(resolved)
    }

    fn resolve_invariant_symbol_names_ordered(
        &self,
        invariant_count: usize,
    ) -> Option<Vec<Option<String>>> {
        if !self.has_all_invariants(invariant_count) {
            return None;
        }
        let mut resolved = Vec::with_capacity(invariant_count);
        for idx in 0..invariant_count {
            let symbol_name = self
                .native_invariant_entries
                .get(idx)
                .and_then(|entry| entry.as_ref())
                .map(|entry| entry.symbol_name.clone());
            resolved.push(symbol_name);
        }
        Some(resolved)
    }

    /// Evaluate a compiled next-state action on a flat i64 state buffer.
    ///
    /// Returns `Some(Ok(result))` if the action is compiled and evaluation
    /// succeeded. Returns `Some(Err(()))` on runtime error. Returns `None`
    /// if the action is not compiled.
    ///
    /// The `Enabled` result carries only the successor buffer. Compound value
    /// deserialization needs the predecessor `state_in` buffer to resolve
    /// offsets written by native FuncExcept operations (#4193); that buffer is
    /// invariant across every dispatch of one predecessor, so callers supply it
    /// directly (the shared flat-state scratch) rather than snapshotting a copy
    /// per enabled successor.
    #[allow(dead_code)] // off-by-default JIT compile/eval machinery (TY_JIT, #4035); kept, currently unwired
    pub(in crate::check) fn eval_action(
        &self,
        action_name: &str,
        state_in: &[i64],
    ) -> Option<Result<TrustCgActionResult, ()>> {
        self.eval_action_with_state_len(action_name, state_in, self.state_var_count)
    }

    /// Evaluate a compiled next-state action with an explicit state-buffer width.
    ///
    /// Most non-fused callers use the logical state-variable count. Flat-slot
    /// callers must instead pass the fully flattened slot count so native code
    /// sees the same width used by the input and output buffers.
    #[allow(dead_code)] // off-by-default JIT compile/eval machinery (TY_JIT, #4035); kept, currently unwired
    pub(in crate::check) fn eval_action_with_state_len(
        &self,
        action_name: &str,
        state_in: &[i64],
        state_len: usize,
    ) -> Option<Result<TrustCgActionResult, ()>> {
        let mut state_out = Vec::new();
        match self.eval_action_with_state_len_into(action_name, state_in, state_len, &mut state_out)
        {
            Some(Ok(true)) => Some(Ok(TrustCgActionResult::Enabled {
                successor: state_out,
            })),
            Some(Ok(false)) => Some(Ok(TrustCgActionResult::Disabled)),
            Some(Err(())) => Some(Err(())),
            None => None,
        }
    }

    /// Evaluate a compiled next-state action into caller-owned scratch.
    ///
    /// This is the hot-path form used by per-action Trust-CG dispatch. Disabled
    /// actions no longer allocate an output `Vec`; enabled actions clone only at
    /// the boundary where the successor has to escape the reusable scratch.
    pub(in crate::check) fn eval_action_with_state_len_into(
        &self,
        action_name: &str,
        state_in: &[i64],
        state_len: usize,
        state_out: &mut Vec<i64>,
    ) -> Option<Result<bool, ()>> {
        // WP-21: reset the per-thread error-kind side channel; only a
        // `JitStatus::RuntimeError` below repopulates it.
        LAST_ACTION_RUNTIME_ERROR_KIND.with(|kind| kind.set(0));
        let fn_ptr = self.next_state_fns.get(action_name)?;
        if let Some(limit) = tla_trust_cg::trust_cg_entry_counter_dispatch_gate_limit() {
            if !self.entry_counter_allows_action_dispatch(action_name, limit) {
                return None;
            }
        }
        let state_len = match u32::try_from(state_len) {
            Ok(state_len) if state_in.len() >= state_len as usize => state_len,
            _ => return Some(Err(())),
        };

        let mut out = JitCallOut::default();
        // Transformed next-state actions keep `Unchanged` checks and only write
        // primed slots they touch. Mirror the native ABI contract by seeding the
        // successor buffer from the predecessor state before native execution.
        state_out.clear();
        state_out.extend_from_slice(state_in);

        // SAFETY: fn_ptr was obtained from our compilation pipeline with the
        // correct ABI. state_in/state_out are valid i64 buffers. out is
        // caller-allocated. state_len matches the model's variable count.
        let owner_published = if let Some(entry) = self.native_action_entries.get(action_name) {
            if entry
                .library
                .ensure_published_symbol_ptr(
                    &entry.symbol_name,
                    *fn_ptr as *const () as *mut std::ffi::c_void,
                )
                .is_err()
            {
                return Some(Err(()));
            }
            true
        } else {
            // Ownerless direct-eval caches exist only in hand-built tests and
            // legacy prototypes; production trust-codegen cache construction stores a
            // NativeLibrary entry next to each raw action pointer.
            false
        };
        if owner_published {
            tla_trust_cg::ensure_jit_execute_mode();
        } else {
            if false {
                return Some(Err(()));
            }
            tla_trust_cg::ensure_jit_execute_mode();
        }
        unsafe {
            fn_ptr(
                &mut out,
                state_in.as_ptr(),
                state_out.as_mut_ptr(),
                state_len,
            );
        }

        match out.status {
            tla_jit_abi::JitStatus::Ok => {
                let enabled = match decode_strict_trust_cg_boolean(
                    out.value,
                    &format!("action {action_name}"),
                ) {
                    Ok(enabled) => enabled,
                    Err(()) => return Some(Err(())),
                };
                Some(Ok(enabled))
            }
            tla_jit_abi::JitStatus::RuntimeError => {
                // WP-21: publish the typed error kind for same-thread
                // classification (shape-guard decline vs real error).
                LAST_ACTION_RUNTIME_ERROR_KIND.with(|kind| kind.set(out.err_kind as u8));
                Some(Err(()))
            }
            // PartialPass or other status -- fall back to interpreter.
            _ => None,
        }
    }

    /// Evaluate a compiled invariant on a flat i64 state buffer.
    ///
    /// Returns `Some(Ok(true))` if the invariant holds, `Some(Ok(false))` if
    /// violated, `Some(Err(()))` on runtime error, `None` if not compiled
    /// (either index out of range or compilation failed for this invariant).
    #[allow(dead_code)] // off-by-default JIT compile/eval machinery (TY_JIT, #4035); kept, currently unwired
    pub(in crate::check) fn eval_invariant(
        &self,
        invariant_idx: usize,
        state: &[i64],
    ) -> Option<Result<bool, ()>> {
        self.eval_invariant_with_state_len(invariant_idx, state, self.state_var_count)
    }

    /// Evaluate a compiled invariant with an explicit state-buffer width.
    #[allow(dead_code)] // off-by-default JIT compile/eval machinery (TY_JIT, #4035); kept, currently unwired
    pub(in crate::check) fn eval_invariant_with_state_len(
        &self,
        invariant_idx: usize,
        state: &[i64],
        state_len: usize,
    ) -> Option<Result<bool, ()>> {
        // Double-unwrap: first get the slot (bounds check), then check if compiled.
        let fn_ptr = (*self.invariant_fns.get(invariant_idx)?)?;
        let state_len = match u32::try_from(state_len) {
            Ok(state_len) if state.len() >= state_len as usize => state_len,
            _ => return Some(Err(())),
        };

        let mut out = JitCallOut::default();

        // SAFETY: fn_ptr was obtained from our compilation pipeline with the
        // correct ABI. state is a valid i64 buffer. out is caller-allocated.
        let owner_published = if let Some(entry) = self
            .native_invariant_entries
            .get(invariant_idx)
            .and_then(Option::as_ref)
        {
            if entry
                .library
                .ensure_published_symbol_ptr(
                    &entry.symbol_name,
                    fn_ptr as *const () as *mut std::ffi::c_void,
                )
                .is_err()
            {
                return Some(Err(()));
            }
            true
        } else {
            // Ownerless direct-eval caches exist only in hand-built tests and
            // legacy prototypes; production trust-codegen cache construction stores a
            // NativeLibrary entry next to each raw invariant pointer.
            false
        };
        if owner_published {
            tla_trust_cg::ensure_jit_execute_mode();
        } else {
            if false {
                return Some(Err(()));
            }
            tla_trust_cg::ensure_jit_execute_mode();
        }
        unsafe {
            fn_ptr(&mut out, state.as_ptr(), state_len);
        }

        match out.status {
            tla_jit_abi::JitStatus::Ok => {
                match decode_strict_trust_cg_boolean(
                    out.value,
                    &format!("invariant index {invariant_idx}"),
                ) {
                    Ok(holds) => Some(Ok(holds)),
                    Err(()) => Some(Err(())),
                }
            }
            tla_jit_abi::JitStatus::RuntimeError => Some(Err(())),
            _ => None,
        }
    }
}

impl std::fmt::Debug for TrustCgNativeCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let native_action_batch_shards = self
            .native_action_entries
            .values()
            .filter(|entry| entry.batch_shard.is_some())
            .count();
        f.debug_struct("TrustCgNativeCache")
            .field("actions", &self.next_state_fns.len())
            .field("invariants", &self.invariant_fns.len())
            .field("state_constraints", &self.state_constraint_fns.len())
            .field("implied_actions", &self.implied_action_fns.len())
            .field("state_var_count", &self.state_var_count)
            .field("native_action_batch_shards", &native_action_batch_shards)
            .field("libraries", &self._libraries.len())
            .finish()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NativeEntrypointRole {
    Action,
    Invariant,
    StateConstraint,
}

impl NativeEntrypointRole {
    fn symbol_component(self) -> &'static str {
        match self {
            Self::Action => "action",
            Self::Invariant => "invariant",
            Self::StateConstraint => "state_constraint",
        }
    }
}

/// Build a stable LLVM symbol for a native user-level entrypoint.
///
/// The human-readable stem is intentionally not the uniqueness boundary:
/// the role and original UTF-8 bytes are appended as length-prefixed hex, so
/// names that sanitize to the same stem still produce distinct symbols.
fn native_entrypoint_symbol_name(role: NativeEntrypointRole, name: &str) -> String {
    use std::fmt::Write as _;

    let stem = sanitize_llvm_name(name);
    let role_component = role.symbol_component();
    let mut symbol = format!("trust_cg_{role_component}_{stem}");

    write!(
        &mut symbol,
        "_r{}x{}_n{}x",
        role_component.len(),
        hex_component(role_component.as_bytes()),
        name.len(),
    )
    .expect("writing to String cannot fail");
    symbol.push_str(&hex_component(name.as_bytes()));
    symbol
}

/// No-boxing audit (wishlist item 8) of a lowered action module: report any
/// `tla_*`-family extern outside the sanctioned handle-mode set
/// ([`tla_ir::lower::SANCTIONED_HANDLE_MODE_TLA_EXTERNS`]). Such an extern
/// means the action compiled but routes through a boxed interpreter-parity
/// kernel (flat slot -> `Value` -> flat slot) — native admission without the
/// win the wishlist's differential inner loop is measuring.
///
/// Gated by `TY_TRUST_CG_DUMP_NATIVE_ADMISSION_FAILURES` and observability-only:
/// emits a diagnostic line (plus a debug-build assert so every lowering
/// increment is verified Value-free by the standard inner loop); never changes
/// admission, dispatch, or verdicts.
fn audit_lowered_action_tla_externs(action_name: &str, module: &trust_ir::Module) {
    if !trust_cg_dump_native_admission_failures_enabled() {
        return;
    }
    let unsanctioned = tla_ir::lower::unsanctioned_tla_extern_names(module);
    if unsanctioned.is_empty() {
        return;
    }
    eprintln!(
        "[trust_cg-admission] action='{action_name}' unsanctioned_boxed_tla_externs={unsanctioned:?} \
         (compiled but Value-boxed: routes through interpreter-parity tla_* kernels)"
    );
    debug_assert!(
        unsanctioned.is_empty(),
        "action '{action_name}' lowered with unsanctioned boxed tla_* externs {unsanctioned:?} \
         (sanctioned set: tla_ir::lower::SANCTIONED_HANDLE_MODE_TLA_EXTERNS)"
    );
}

fn hex_component(bytes: &[u8]) -> String {
    use std::fmt::Write as _;

    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    encoded
}

/// Sanitize a TLA+ action name into a human-readable LLVM function stem.
///
/// LLVM identifiers must match `[a-zA-Z._][a-zA-Z._0-9]*`. Replace invalid
/// characters with underscores.
fn sanitize_llvm_name(name: &str) -> String {
    let mut result = String::with_capacity(name.len());
    for (i, ch) in name.chars().enumerate() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '.' {
            result.push(ch);
        } else if i == 0 {
            result.push('_');
        } else {
            result.push('_');
        }
    }
    if result.is_empty() {
        result.push_str("_unnamed");
    }
    result
}

/// Check whether trust-codegen dispatch should be active for the current run.
///
/// Returns `true` when:
/// 1. The `trust_cg` feature is compiled in (checked at compile time by cfg gate).
/// 2. `TY_trust_cg=1` environment variable is set.
/// 3. The `native` feature is compiled into `tla-trust_cg` (always true when
///    `tla-check/trust_cg` feature is enabled).
pub(in crate::check) fn should_use_trust_cg(structurally_vetoed: bool) -> bool {
    TrustCgNativeCache::is_enabled(structurally_vetoed) && TrustCgNativeCache::is_available()
}

/// First trust_cg-backed adapter for the backend-agnostic compiled BFS step trait.
///
/// This milestone reuses separately compiled trust-codegen action and invariant
/// function pointers. It does not yet generate a fused trust-codegen BFS function; the
/// existing Rust compiled-BFS loop calls this adapter once per parent state.
pub(in crate::check) struct TrustCgCompiledBfsStep {
    action_fns: Vec<NativeNextStateFn>,
    /// Owner libraries for the raw fn pointers. Publication happens once at
    /// construction (H7 hoist in [`Self::from_cache_with_state_len`]); the
    /// handles are retained here as keep-alives so the mapped native code
    /// outlives every stored fn pointer.
    #[allow(dead_code)]
    action_libraries: Vec<Option<tla_trust_cg::NativeLibrary>>,
    #[allow(dead_code)]
    action_symbol_names: Vec<Option<String>>,
    invariant_fns: Vec<NativeInvariantFn>,
    /// Keep-alive owners for `invariant_fns`; see `action_libraries`.
    #[allow(dead_code)]
    invariant_libraries: Vec<Option<tla_trust_cg::NativeLibrary>>,
    #[allow(dead_code)]
    invariant_symbol_names: Vec<Option<String>>,
    state_len: usize,
}

impl TrustCgCompiledBfsStep {
    /// Build a per-parent trust-codegen compiled BFS step from an existing native cache.
    ///
    /// Returns `None` unless every requested action key and every configured
    /// invariant resolves to a native trust-codegen function. That all-or-nothing guard
    /// is what keeps missing invariant coverage from entering compiled BFS.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(in crate::check) fn from_cache(
        cache: &TrustCgNativeCache,
        action_keys: &[String],
        invariant_count: usize,
    ) -> Option<Self> {
        Self::from_cache_with_state_len(
            cache,
            action_keys,
            invariant_count,
            cache.state_var_count(),
        )
    }

    pub(in crate::check) fn from_cache_with_state_len(
        cache: &TrustCgNativeCache,
        action_keys: &[String],
        invariant_count: usize,
        state_len: usize,
    ) -> Option<Self> {
        let action_fns = cache.resolve_actions_ordered(action_keys)?;
        let action_libraries = cache.resolve_action_owner_libraries_ordered(action_keys)?;
        let action_symbol_names = cache.resolve_action_symbol_names_ordered(action_keys)?;
        let invariant_fns = cache.resolve_invariants_ordered(invariant_count)?;
        let invariant_libraries =
            cache.resolve_invariant_owner_libraries_ordered(invariant_count)?;
        let invariant_symbol_names =
            cache.resolve_invariant_symbol_names_ordered(invariant_count)?;

        // Publication hoist (H7): `ensure_published_symbol_ptr` /
        // `ensure_published_executable` are idempotent registrations, so run
        // them ONCE at step construction instead of once per parent state /
        // per successor in the hot loops below. A publication failure makes
        // the step unavailable (`None`), routing checking to the fallback
        // paths. `ensure_jit_execute_mode` is deliberately NOT hoisted: it is
        // a global W^X toggle that mid-BFS compilations can flip, so it is
        // re-asserted (cheaply) per call in `run_step_scoped`.
        for (idx, library) in action_libraries.iter().enumerate() {
            let Some(library) = library.as_ref() else {
                continue;
            };
            let action_fn = action_fns.get(idx)?;
            if let Some(symbol_name) = action_symbol_names.get(idx).and_then(Option::as_deref) {
                library
                    .ensure_published_symbol_ptr(
                        symbol_name,
                        *action_fn as *const () as *mut std::ffi::c_void,
                    )
                    .ok()?;
            } else {
                library.ensure_published_executable().ok()?;
            }
        }
        for (idx, library) in invariant_libraries.iter().enumerate() {
            let Some(library) = library.as_ref() else {
                continue;
            };
            let invariant_fn = invariant_fns.get(idx)?;
            if let Some(symbol_name) = invariant_symbol_names.get(idx).and_then(Option::as_deref) {
                library
                    .ensure_published_symbol_ptr(
                        symbol_name,
                        *invariant_fn as *const () as *mut std::ffi::c_void,
                    )
                    .ok()?;
            } else {
                library.ensure_published_executable().ok()?;
            }
        }

        Some(Self {
            action_fns,
            action_libraries,
            action_symbol_names,
            invariant_fns,
            invariant_libraries,
            invariant_symbol_names,
            state_len,
        })
    }

    pub(in crate::check) fn state_len(&self) -> usize {
        self.state_len
    }

    fn run_step_scoped<'a>(
        &self,
        state: &[i64],
        scratch: &'a mut super::bfs::compiled_step_trait::CompiledBfsStepScratch,
    ) -> Result<super::bfs::compiled_step_trait::CompiledStepOutput<'a>, BfsStepError> {
        if state.len() != self.state_len {
            return Err(BfsStepError::RuntimeError);
        }
        let state_len_u32 =
            u32::try_from(self.state_len).map_err(|_| BfsStepError::RuntimeError)?;

        let mut successor_count = 0usize;
        let mut generated_count = 0u32;
        let mut invariant_ok = true;
        let mut failed_invariant_idx = None;
        let mut failed_successor_idx = None;
        scratch.clear();

        for (action_idx, action_fn) in self.action_fns.iter().enumerate() {
            let mut out = JitCallOut::default();
            let successor_start = scratch.append_successor_template(state)?;

            // SAFETY: Function pointers come from `TrustCgNativeCache` after
            // native compilation with the next-state ABI. Buffers are valid for
            // `state_len` i64 slots and `out` is caller-allocated. Symbol
            // publication was hoisted to step construction (H7:
            // `from_cache_with_state_len` — idempotent registration); only the
            // global W^X execute-mode toggle is re-asserted per call because
            // mid-BFS compilations can flip it.
            unsafe {
                let state_out = scratch.successor_mut(successor_start, self.state_len)?;
                tla_trust_cg::ensure_jit_execute_mode();
                action_fn(
                    &mut out,
                    state.as_ptr(),
                    state_out.as_mut_ptr(),
                    state_len_u32,
                );
            }

            match out.status {
                tla_jit_abi::JitStatus::Ok => {
                    if !decode_strict_trust_cg_boolean(
                        out.value,
                        &format!("compiled BFS action index {action_idx}"),
                    )
                    .map_err(|()| BfsStepError::RuntimeError)?
                    {
                        scratch.truncate_slots(successor_start);
                        continue;
                    }
                    generated_count = generated_count
                        .checked_add(1)
                        .ok_or(BfsStepError::RuntimeError)?;
                    let current_successor_idx = successor_count;
                    let state_out = scratch.successor(successor_start, self.state_len)?;

                    for (idx, invariant_fn) in self.invariant_fns.iter().enumerate() {
                        let mut inv_out = JitCallOut::default();
                        // SAFETY: Invariant function pointers come from
                        // `TrustCgNativeCache` with the invariant ABI. The
                        // successor buffer is valid for `state_len` i64 slots.
                        // Symbol publication was hoisted to step construction
                        // (H7); only the global W^X execute-mode toggle is
                        // re-asserted per call.
                        tla_trust_cg::ensure_jit_execute_mode();
                        unsafe {
                            invariant_fn(&mut inv_out, state_out.as_ptr(), state_len_u32);
                        }

                        match inv_out.status {
                            tla_jit_abi::JitStatus::Ok => {
                                if decode_strict_trust_cg_boolean(
                                    inv_out.value,
                                    &format!("compiled BFS invariant index {idx}"),
                                )
                                .map_err(|()| BfsStepError::RuntimeError)?
                                {
                                    continue;
                                }
                                invariant_ok = false;
                                failed_invariant_idx = Some(
                                    u32::try_from(idx).map_err(|_| BfsStepError::RuntimeError)?,
                                );
                                failed_successor_idx = Some(
                                    u32::try_from(current_successor_idx)
                                        .map_err(|_| BfsStepError::RuntimeError)?,
                                );
                                break;
                            }
                            tla_jit_abi::JitStatus::RuntimeError => {
                                return Err(BfsStepError::RuntimeError);
                            }
                            _ => return Err(BfsStepError::RuntimeError),
                        }
                    }

                    successor_count += 1;

                    if !invariant_ok {
                        break;
                    }
                }
                tla_jit_abi::JitStatus::RuntimeError => return Err(BfsStepError::RuntimeError),
                _ => return Err(BfsStepError::RuntimeError),
            }
        }

        let output = scratch.output_ref(
            self.state_len,
            successor_count,
            generated_count,
            invariant_ok,
            failed_invariant_idx,
            failed_successor_idx,
        )?;
        Ok(
            super::bfs::compiled_step_trait::CompiledStepOutput::from_borrowed(
                output,
                self.state_len,
            ),
        )
    }
}

impl super::bfs::compiled_step_trait::CompiledBfsStep for TrustCgCompiledBfsStep {
    fn state_len(&self) -> usize {
        self.state_len
    }

    fn preserves_state_graph_successor_edges(&self) -> bool {
        true
    }

    fn step_flat(&self, state: &[i64]) -> Result<FlatBfsStepOutput, BfsStepError> {
        let mut scratch =
            super::bfs::compiled_step_trait::CompiledBfsStepScratch::new(self.state_len);
        self.run_step_scoped(state, &mut scratch)
            .map(|output| output.to_owned_flat())
    }

    fn step_flat_scoped<'a>(
        &self,
        state: &[i64],
        scratch: &'a mut super::bfs::compiled_step_trait::CompiledBfsStepScratch,
    ) -> Result<super::bfs::compiled_step_trait::CompiledStepOutput<'a>, BfsStepError> {
        self.run_step_scoped(state, scratch)
    }
}

/// Route B: hard cap (successors per parent) for the adaptive loop-kernel
/// successor-arena growth in the native fused level path. A level that still
/// overflows at this width propagates `BufferOverflow` and the checker falls
/// back to the per-parent/interpreter path — sound (never a truncated
/// successor set), exactly like the host prototype's sink slot cap.
const TRUST_CG_LOOP_LEVEL_MAX_SUCCESSORS_PER_PARENT: usize = 1 << 16;

/// Route B: initial (and floor) successors-per-parent width for loop-kernel
/// levels. Real per-parent yield is runtime-dependent and typically FAR below
/// the parents x action_count worst case the single-successor sizing assumes —
/// that preallocation dominated peak RSS (PaxosCommit: ~580MB at the widest
/// level, 1.30x TLC). Starting narrow is sound and self-correcting: the
/// grow-and-retry loop doubles on overflow (each retry is a complete level
/// re-run over a cleared arena), early levels are small so the few doubling
/// re-runs are nearly free, and the learned width persists across levels via
/// `loop_successors_per_parent` — wide levels never re-run.
const TRUST_CG_LOOP_LEVEL_INITIAL_SUCCESSORS_PER_PARENT: usize = 4;

/// Slack added to the learned per-parent successor width when predictively
/// sizing the successor arena for the NON-loop (specialized single-successor)
/// native fused path. Each of the `action_count` specialized actions fires at
/// most one successor per parent, so `action_count` is a proven-sufficient
/// per-parent ceiling; flooring the arena at `parents x action_count` (the old
/// behavior) over-provisioned peak RSS badly for message-passing specs whose
/// real per-parent yield is far below that ceiling (MCLamportMutex: 27 actions
/// but ~3.5 successors/parent, a 7.6x over-reservation of a ~89-slot flat
/// state — the single dominant term in peak heap). Instead we learn the yield
/// from the previous level's actual committed fill and add this slack so the
/// next (typically wider) level rarely needs a grow-and-retry. Mispredicts are
/// SOUND, never truncating: the native level returns `BufferOverflow` with
/// nothing committed (every invocation is a complete re-run over a cleared
/// arena), and the grow-and-retry loop widens the arena up to the proven
/// `parents x action_count` ceiling, which cannot overflow.
const TRUST_CG_NONLOOP_LEVEL_SUCCESSOR_WIDTH_SLACK: usize = 2;

/// trust_cg-backed adapter for the backend-agnostic compiled BFS level trait.
///
/// This currently wraps `tla_trust_cg::TrustCgBfsLevelPrototype`: action and
/// invariant calls are native trust-codegen function pointers, while the parent
/// frontier loop is still Rust. That makes the level path available to
/// `compiled_bfs_loop.rs` without pretending that trust-codegen has generated a native
/// fused parent-loop function yet.
pub(in crate::check) struct TrustCgCompiledBfsLevel {
    implementation: TrustCgCompiledBfsLevelImplementation,
    state_len: usize,
    native_fused_loop: bool,
    native_fused_state_constraint_count: usize,
    native_fused_invariant_count: usize,
    regular_invariants_checked_by_backend: bool,
    state_graph_successors_complete: bool,
}

// `Native` carries the active per-level native state and is the production
// variant; boxing it to match the lighter `Prototype`/`MockNative` cases would
// add indirection on the hot level-stepping path, so the size gap is accepted.
#[allow(clippy::large_enum_variant)]
enum TrustCgCompiledBfsLevelImplementation {
    Prototype {
        prototype: Mutex<tla_trust_cg::CompiledBfsLevel>,
        // Successor-arena pool shared across levels. Mirrors the `Native`
        // variant: the prototype level previously allocated (and dropped) a
        // fresh `TrustCgSuccessorArena` every level, churning the allocator on
        // large frontiers. Recycling the arena through this pool reuses the
        // backing allocation between levels (cleared on return, so no stale
        // successor/handle aliasing).
        successor_arena_pool: super::bfs::compiled_step_trait::TrustCgSuccessorArenaPool,
    },
    Native {
        level: Mutex<tla_trust_cg::TrustCgBfsLevelNative>,
        action_count: usize,
        /// Route B: true when the compiled level contains at least one
        /// multi-successor record-set kernel (`NextStateLoopFn` sink ABI).
        /// Enables the adaptive successor-arena growth retry on
        /// `BufferOverflow` — a loop kernel's per-parent successor count is
        /// runtime data, so `parent_count * action_count` is not a capacity
        /// upper bound for such levels.
        has_loop_actions: bool,
        /// Shared learned successors-per-parent width. Despite the legacy
        /// `loop_` field name, the non-loop specialized path now owns the
        /// width update: it starts at
        /// [`TRUST_CG_LOOP_LEVEL_INITIAL_SUCCESSORS_PER_PARENT`] and, after a
        /// successful level, stores `ceil(total_new / parents) + 2`, clamped
        /// to `action_count`. An underprediction grows only the invocation's
        /// local arena capacity before a complete retry; it does not mutate
        /// this atomic until that retry succeeds. Loop-kernel capacity is
        /// predicted separately from `loop_prev_level_new` below.
        loop_successors_per_parent: std::sync::atomic::AtomicUsize,
        /// Route B: the previous level's actual committed successor count
        /// (`total_new`), used to size the next level's arena predictively —
        /// most levels yield FAR less than parents x learned-width, and the
        /// per-parent worst-case allocation dominated peak RSS. 0 = no
        /// history yet (fall back to parents x initial width).
        loop_prev_level_new: std::sync::atomic::AtomicUsize,
        successor_arena_pool: super::bfs::compiled_step_trait::TrustCgSuccessorArenaPool,
        callout_selftest: Mutex<Option<TrustCgNativeCalloutSelftest>>,
    },
    #[cfg(test)]
    MockNative {
        entrypoint: tla_trust_cg::TrustCgFusedLevelFn,
        action_count: usize,
        scratch: Mutex<Vec<i64>>,
    },
}

impl TrustCgCompiledBfsLevel {
    /// Build a full-frontier trust-codegen level adapter from an existing native cache.
    ///
    /// Returns `None` unless every requested action key resolves to native
    /// trust-codegen code. The production native path is action-only: the generated
    /// parent loop emits successors and the Rust BFS loop checks invariants
    /// after flat dedup. The prototype fallback still requires compiled
    /// invariants because it performs invariant checks inside the prototype
    /// level object.
    #[allow(dead_code)] // JIT batch-compile / trust-cg-cache machinery, currently unwired (TY_JIT off by default, #4035)
    pub(in crate::check) fn from_cache(
        cache: &TrustCgNativeCache,
        action_keys: &[String],
        invariant_names: &[String],
        state_constraint_names: &[String],
        expected_states: usize,
        native_fused_state_len: Option<usize>,
    ) -> Option<Self> {
        Self::from_cache_with_native_fused_action_pre_call_pc_guards(
            cache,
            action_keys,
            invariant_names,
            state_constraint_names,
            &[],
            expected_states,
            native_fused_state_len,
            &[],
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::check) fn from_cache_with_native_fused_action_pre_call_pc_guards(
        cache: &TrustCgNativeCache,
        action_keys: &[String],
        invariant_names: &[String],
        state_constraint_names: &[String],
        implied_action_names: &[String],
        expected_states: usize,
        native_fused_state_len: Option<usize>,
        native_fused_action_pre_call_pc_guards: &[Option<tla_trust_cg::NativeBfsPreCallPcGuard>],
        // Size-gate: when `false`, the (unconstrained) native fused level is
        // built in action-only mode even though `invariant_names` compiled to
        // native entries in `cache`. The BFS loop then checks invariants per
        // successor in the interpreter (`regular_invariants_checked_by_backend`
        // is `false` because `invariant_names` is non-empty), exactly the
        // default action-only path. A caller sets this `false` below the
        // `TY_FUSED_INVARIANT_MIN_STATES` floor to skip the (large) invariant
        // fusion compile, and rebuilds with `true` once the run is large enough.
        // Ignored for state-constrained runs (they always fuse — the gate is
        // never applied to them).
        fuse_invariants: bool,
    ) -> Option<Self> {
        // Action-property (implied-action) predicates are only admitted into the
        // fused native level when every name resolves to a native entry; the
        // per-branch `resolve_native_implied_actions_ordered` checks below fail
        // closed otherwise. The eligibility decision (all terms native-capable +
        // flat-primary-safe layout) is made upstream by
        // `implied_actions_require_interpreter_eval`. Native implied-action
        // checking requires the native fused parent loop, so a non-empty
        // implied-action set without `native_fused_state_len` is not eligible.
        if !implied_action_names.is_empty() && native_fused_state_len.is_none() {
            eprintln!(
                "[trust_cg] native fused CompiledBfsLevel not eligible: implied actions require the native fused parent loop (no native_fused_state_len)"
            );
            return None;
        }

        if native_fused_state_len.is_some() {
            if let Some((residual_key, expansion_count)) = action_keys.iter().find_map(|key| {
                let expansions = cache.inner_exists_expansion_keys(key);
                (!expansions.is_empty()).then_some((key, expansions.len()))
            }) {
                eprintln!(
                    "[trust-cg] native fused CompiledBfsLevel not eligible: action key \
                     '{residual_key}' has {expansion_count} inner EXISTS expansion key(s); \
                     expanded action keys are required"
                );
                return None;
            }
        }

        // Any-ABI resolution: multi-successor record-set kernels
        // (`next_state_loop_fns`) resolve to `TrustCgCompiledAction::new_loop`
        // entries that the prototype parent loop dispatches through the
        // `NextStateLoopSink` convention. Previously this used the strict
        // single-ABI `resolve_actions_ordered`, which silently returned `None`
        // for any loop action and prevented the level from ever building.
        let actions = cache.resolve_compiled_actions_ordered_any_abi(action_keys)?;

        // Route B: the generated native fused parent loop drives sink-ABI
        // (`NextStateLoopFn`) actions through the native sink commit loop
        // (`append_loop_action_blocks`) — but ONLY in the constraint-free,
        // implied-action-free configurations (action-only and
        // invariant-checking modes). The loop commit path does not run the
        // per-successor state-constraint / implied-action predicate blocks, and
        // the generated module keeps a fail-closed fallback branch for that
        // combination, so a native level built with them would fall back on
        // every level. Gate accordingly:
        // - loop + implied actions: fail closed to the interpreter (unchanged —
        //   implied-action checking only exists in the native fused loop, and
        //   the prototype would silently skip it);
        // - loop + state constraints: skip the native attempt and use the host
        //   prototype sink dispatch (unchanged behavior);
        // - loop, no constraints, no implied actions: attempt the native fused
        //   level. Native invariant entries (when all resolve) run per sink
        //   successor inside the commit loop exactly like the single-successor
        //   path; otherwise the action-only native mode is used and the BFS
        //   loop's per-successor Rust invariant check applies, identical to
        //   non-loop specs.
        let native_fused_attempt_state_len = if cache.any_action_is_loop(action_keys) {
            if !implied_action_names.is_empty() {
                eprintln!(
                    "[trust-cg] CompiledBfsLevel not eligible: implied actions require the native fused parent loop, which fails closed for record-set loop actions combined with implied actions"
                );
                return None;
            }
            if !state_constraint_names.is_empty() {
                if native_fused_state_len.is_some() {
                    eprintln!(
                        "[trust-cg] native fused CompiledBfsLevel skipped: record-set loop action(s) present with state constraints; the native sink commit loop does not run constraint blocks — using host prototype sink dispatch"
                    );
                }
                None
            } else {
                native_fused_state_len
            }
        } else {
            native_fused_state_len
        };

        if let Some(native_fused_state_len) = native_fused_attempt_state_len {
            if let Some(mut native_actions) = cache.resolve_native_actions_ordered(action_keys) {
                TrustCgNativeCalloutSelftest::log_fused_build_without_sample(
                    native_fused_state_len,
                );
                if !native_fused_action_pre_call_pc_guards.is_empty() {
                    if native_fused_action_pre_call_pc_guards.len() == native_actions.len() {
                        for (action, guard) in native_actions
                            .iter_mut()
                            .zip(native_fused_action_pre_call_pc_guards)
                        {
                            if let Some(guard) = guard {
                                action.set_pre_call_pc_guard(*guard);
                            }
                        }
                    } else {
                        eprintln!(
                            "[trust-cg] native fused pc pre-call guards ignored: guard count {} does not match action count {}",
                            native_fused_action_pre_call_pc_guards.len(),
                            native_actions.len(),
                        );
                    }
                }

                if !state_constraint_names.is_empty() {
                    let Some(native_state_constraints) =
                        cache.resolve_native_state_constraints_ordered(state_constraint_names)
                    else {
                        let missing = state_constraint_names
                            .iter()
                            .enumerate()
                            .filter_map(|(idx, name)| {
                                cache
                                    .native_state_constraint_entries
                                    .get(idx)
                                    .and_then(|entry| entry.as_ref())
                                    .is_none()
                                    .then_some(name.as_str())
                            })
                            .collect::<Vec<_>>();
                        let first_missing = missing.first().copied().unwrap_or("<unknown>");
                        eprintln!(
                            "[trust-cg] native fused CompiledBfsLevel not eligible: {}/{} state constraints missing native entries (first missing: {first_missing})",
                            missing.len(),
                            state_constraint_names.len(),
                        );
                        return None;
                    };

                    let native_implied_actions =
                        cache.resolve_native_implied_actions_ordered(implied_action_names);
                    let implied_actions_checked_by_backend =
                        native_implied_actions.is_some() || implied_action_names.is_empty();
                    if !implied_action_names.is_empty() && !implied_actions_checked_by_backend {
                        eprintln!(
                            "[trust_cg] state-constrained native fused CompiledBfsLevel not eligible: not all implied actions have native entries"
                        );
                        return None;
                    }
                    let native_implied_actions = native_implied_actions.unwrap_or_default();

                    let native_invariants =
                        cache.resolve_native_invariants_ordered(invariant_names);
                    let regular_invariants_checked_by_backend = native_invariants.is_some();
                    if !invariant_names.is_empty() && !regular_invariants_checked_by_backend {
                        eprintln!(
                            "[trust-cg] state-constrained native fused CompiledBfsLevel using Rust invariant fallback: not all regular invariants have native entries"
                        );
                    }
                    let native_invariants = native_invariants.unwrap_or_default();
                    let callout_selftest =
                        TrustCgNativeCalloutSelftest::from_cache_if_enabled_or_required(
                            cache,
                            &native_actions,
                            &native_state_constraints,
                            &native_invariants,
                            true,
                        );

                    let native_level_start = std::time::Instant::now();
                    let native_level_result =
                        tla_trust_cg::compile_bfs_level_native_with_state_constraints_and_implied_actions(
                            native_fused_state_len,
                            &native_actions,
                            &native_state_constraints,
                            &native_implied_actions,
                            &native_invariants,
                            cache.opt_level,
                        );
                    if trust_cg_setup_timing_enabled() {
                        eprintln!(
                            "[trust_cg-timing] compile_bfs_level_native_with_state_constraints_ms={} actions={} state_constraints={} invariants={} state_len={}",
                            native_level_start.elapsed().as_millis(),
                            native_actions.len(),
                            native_state_constraints.len(),
                            native_implied_actions.len() + native_invariants.len(),
                            native_fused_state_len,
                        );
                    }
                    match native_level_result {
                        Ok(native_level)
                            if native_level.state_constraint_count()
                                == state_constraint_names.len() =>
                        {
                            return Self::from_native_with_callout_selftest(
                                native_level,
                                action_keys.len(),
                                regular_invariants_checked_by_backend,
                                callout_selftest,
                            );
                        }
                        Ok(native_level) => {
                            eprintln!(
                                "[trust-cg] state-constrained native fused CompiledBfsLevel rejected: backend reported {}/{} active state constraints",
                                native_level.state_constraint_count(),
                                state_constraint_names.len(),
                            );
                            return None;
                        }
                        Err(err) => {
                            eprintln!(
                                "[trust-cg] state-constrained native fused CompiledBfsLevel unavailable: {err}"
                            );
                            return None;
                        }
                    }
                }

                let fused_native_invariants = if fuse_invariants {
                    cache.resolve_native_invariants_ordered(invariant_names)
                } else {
                    // Size-gate: force action-only even though the invariants
                    // compiled. `invariant_names` stays non-empty so the
                    // action-only fallback below reports
                    // `regular_invariants_checked_by_backend = false` and the
                    // BFS loop runs the per-successor interpreter invariant check.
                    None
                };
                if let Some(native_invariants) = fused_native_invariants {
                    let callout_selftest =
                        TrustCgNativeCalloutSelftest::from_cache_if_enabled_or_required(
                            cache,
                            &native_actions,
                            &[],
                            &native_invariants,
                            false,
                        );
                    let native_level_start = std::time::Instant::now();
                    let native_implied_actions =
                        cache.resolve_native_implied_actions_ordered(implied_action_names);
                    if !implied_action_names.is_empty() && native_implied_actions.is_none() {
                        eprintln!(
                            "[trust_cg] native fused CompiledBfsLevel not eligible: not all implied actions have native entries"
                        );
                        return None;
                    }
                    let native_implied_actions = native_implied_actions.unwrap_or_default();
                    let native_level_result =
                        tla_trust_cg::compile_bfs_level_native_with_state_constraints_and_implied_actions(
                        native_fused_state_len,
                        &native_actions,
                            &[],
                            &native_implied_actions,
                        &native_invariants,
                        cache.opt_level,
                    );
                    if trust_cg_setup_timing_enabled() {
                        eprintln!(
                            "[trust_cg-timing] compile_bfs_level_native_ms={} actions={} invariants={} state_len={}",
                            native_level_start.elapsed().as_millis(),
                            native_actions.len(),
                            native_implied_actions.len() + native_invariants.len(),
                            native_fused_state_len,
                        );
                    }
                    match native_level_result {
                        Ok(native_level) => {
                            return Self::from_native_with_callout_selftest(
                                native_level,
                                action_keys.len(),
                                true,
                                callout_selftest,
                            );
                        }
                        Err(err) => {
                            eprintln!(
                                "[trust-cg] invariant-checking native fused CompiledBfsLevel unavailable, falling back to action-only native/prototype: {err}"
                            );
                        }
                    }
                } else if !invariant_names.is_empty() {
                    if fuse_invariants {
                        eprintln!(
                            "[trust-cg] native fused CompiledBfsLevel using action-only fallback: not all invariants have native entries"
                        );
                    } else {
                        eprintln!(
                            "[trust-cg] native fused CompiledBfsLevel built action-only: invariant fusion size-gated below TY_FUSED_INVARIANT_MIN_STATES; invariants checked per successor by the interpreter"
                        );
                    }
                }

                let callout_selftest =
                    TrustCgNativeCalloutSelftest::from_cache_if_enabled_or_required(
                        cache,
                        &native_actions,
                        &[],
                        &[],
                        false,
                    );
                let native_level_start = std::time::Instant::now();
                let native_implied_actions =
                    cache.resolve_native_implied_actions_ordered(implied_action_names);
                if !implied_action_names.is_empty() && native_implied_actions.is_none() {
                    eprintln!(
                        "[trust_cg] action-only native fused CompiledBfsLevel not eligible: not all implied actions have native entries"
                    );
                    return None;
                }
                let native_implied_actions = native_implied_actions.unwrap_or_default();
                let native_level_result =
                    tla_trust_cg::compile_bfs_level_native_with_state_constraints_and_implied_actions(
                        native_fused_state_len,
                        &native_actions,
                        &[],
                        &native_implied_actions,
                        &[],
                        cache.opt_level,
                    );
                if trust_cg_setup_timing_enabled() {
                    eprintln!(
                        "[trust_cg-timing] compile_bfs_level_native_actions_only_ms={} actions={} state_len={}",
                        native_level_start.elapsed().as_millis(),
                        native_actions.len(),
                        native_fused_state_len,
                    );
                }
                match native_level_result {
                    Ok(native_level) => {
                        return Self::from_native_with_callout_selftest(
                            native_level,
                            action_keys.len(),
                            invariant_names.is_empty(),
                            callout_selftest,
                        );
                    }
                    Err(err) => {
                        eprintln!(
                            "[trust-cg] action-only native fused CompiledBfsLevel unavailable, falling back to prototype: {err}"
                        );
                    }
                }
            }
        }

        let prototype_state_len = native_fused_state_len.unwrap_or_else(|| cache.state_var_count());
        let (invariants, regular_invariants_checked_by_backend): (
            Vec<tla_trust_cg::TrustCgCompiledInvariant>,
            bool,
        ) = match cache.resolve_invariants_ordered(invariant_names.len()) {
            Some(invariant_fns) => {
                let invariant_libraries =
                    cache.resolve_invariant_owner_libraries_ordered(invariant_names.len())?;
                let invariant_symbol_names =
                    cache.resolve_invariant_symbol_names_ordered(invariant_names.len())?;
                let invariants = invariant_names
                    .iter()
                    .cloned()
                    .enumerate()
                    .zip(
                        invariant_fns
                            .into_iter()
                            .zip(invariant_libraries)
                            .zip(invariant_symbol_names),
                    )
                    .map(|((idx, name), ((func, library), symbol_name))| {
                        let invariant = tla_trust_cg::TrustCgCompiledInvariant::new(
                            tla_trust_cg::InvariantDescriptor {
                                name,
                                invariant_idx: idx as u32,
                            },
                            func,
                        );
                        match (library, symbol_name) {
                            (Some(library), Some(symbol_name)) => {
                                invariant.with_owner_library_and_symbol(library, symbol_name)
                            }
                            (Some(library), None) => invariant.with_owner_library(library),
                            _ => invariant,
                        }
                    })
                    .collect();
                (invariants, true)
            }
            None if cache.any_action_is_loop(action_keys) => {
                // Route B prototype relaxation: mirror the native action-only
                // fallback. The compiled BFS loop checks regular invariants in
                // Rust per successor whenever
                // `regular_invariants_checked_by_backend` is false
                // (`fused_successor_needs_rust_regular_invariant_check`), so
                // building the level with no compiled invariants is sound.
                // Restricted to loop-action levels so legacy (non-loop)
                // prototype admission stays byte-identical.
                eprintln!(
                    "[trust-cg] prototype CompiledBfsLevel using Rust invariant fallback: not all regular invariants have native entries"
                );
                (Vec::new(), false)
            }
            None => return None,
        };

        let prototype = tla_trust_cg::CompiledBfsLevel::new(
            prototype_state_len,
            actions,
            invariants,
            expected_states,
        )
        .ok()?;
        let capabilities = prototype.capabilities();

        Some(Self {
            implementation: TrustCgCompiledBfsLevelImplementation::Prototype {
                prototype: Mutex::new(prototype),
                successor_arena_pool: Arc::new(Mutex::new(Some(
                    tla_trust_cg::TrustCgSuccessorArena::new(prototype_state_len),
                ))),
            },
            state_len: prototype_state_len,
            native_fused_loop: capabilities.native_fused_loop,
            native_fused_state_constraint_count: 0,
            native_fused_invariant_count: 0,
            regular_invariants_checked_by_backend,
            state_graph_successors_complete: !capabilities.local_dedup,
        })
    }

    /// Wrap a native trust-codegen fused parent-loop object once the trust-codegen compiler
    /// surface can produce one.
    #[allow(dead_code)]
    pub(in crate::check) fn from_native(
        native_level: tla_trust_cg::TrustCgBfsLevelNative,
        action_count: usize,
        regular_invariants_checked_by_backend: bool,
    ) -> Option<Self> {
        Self::from_native_with_callout_selftest(
            native_level,
            action_count,
            regular_invariants_checked_by_backend,
            None,
        )
    }

    fn from_native_with_callout_selftest(
        native_level: tla_trust_cg::TrustCgBfsLevelNative,
        action_count: usize,
        regular_invariants_checked_by_backend: bool,
        callout_selftest: Option<TrustCgNativeCalloutSelftest>,
    ) -> Option<Self> {
        let capabilities = native_level.capabilities();
        if !capabilities.native_fused_loop {
            return None;
        }
        let state_len = native_level.state_len();
        let native_fused_state_constraint_count = native_level.state_constraint_count();
        let native_fused_invariant_count = native_level.invariant_count();
        let has_loop_actions = native_level.metadata().has_loop_actions;

        Some(Self {
            state_len,
            implementation: TrustCgCompiledBfsLevelImplementation::Native {
                level: Mutex::new(native_level),
                action_count,
                has_loop_actions,
                loop_successors_per_parent: std::sync::atomic::AtomicUsize::new(
                    TRUST_CG_LOOP_LEVEL_INITIAL_SUCCESSORS_PER_PARENT,
                ),
                loop_prev_level_new: std::sync::atomic::AtomicUsize::new(0),
                successor_arena_pool: Arc::new(Mutex::new(Some(
                    tla_trust_cg::TrustCgSuccessorArena::new(state_len),
                ))),
                callout_selftest: Mutex::new(callout_selftest),
            },
            native_fused_loop: true,
            native_fused_state_constraint_count,
            native_fused_invariant_count,
            regular_invariants_checked_by_backend,
            state_graph_successors_complete: !capabilities.local_dedup,
        })
    }

    #[cfg(test)]
    fn test_native_successors_per_parent_width(&self) -> Option<usize> {
        match &self.implementation {
            TrustCgCompiledBfsLevelImplementation::Native {
                loop_successors_per_parent,
                ..
            } => Some(loop_successors_per_parent.load(std::sync::atomic::Ordering::Relaxed)),
            _ => None,
        }
    }

    #[cfg(test)]
    fn from_mock_native_fn(
        state_len: usize,
        action_count: usize,
        entrypoint: tla_trust_cg::TrustCgFusedLevelFn,
    ) -> Self {
        Self::from_mock_native_fn_with_metadata(state_len, action_count, entrypoint, 0, false)
    }

    #[cfg(test)]
    fn from_mock_native_fn_with_metadata(
        state_len: usize,
        action_count: usize,
        entrypoint: tla_trust_cg::TrustCgFusedLevelFn,
        native_fused_invariant_count: usize,
        regular_invariants_checked_by_backend: bool,
    ) -> Self {
        Self::from_mock_native_fn_with_counts(
            state_len,
            action_count,
            entrypoint,
            0,
            native_fused_invariant_count,
            regular_invariants_checked_by_backend,
        )
    }

    #[cfg(test)]
    fn from_mock_native_fn_with_counts(
        state_len: usize,
        action_count: usize,
        entrypoint: tla_trust_cg::TrustCgFusedLevelFn,
        native_fused_state_constraint_count: usize,
        native_fused_invariant_count: usize,
        regular_invariants_checked_by_backend: bool,
    ) -> Self {
        let scratch_len = state_len
            .max(1)
            .checked_add(64)
            .expect("mock native fused level scratch layout");
        Self {
            state_len,
            implementation: TrustCgCompiledBfsLevelImplementation::MockNative {
                entrypoint,
                action_count,
                scratch: Mutex::new(vec![0; scratch_len]),
            },
            native_fused_loop: true,
            native_fused_state_constraint_count,
            native_fused_invariant_count,
            regular_invariants_checked_by_backend,
            state_graph_successors_complete: true,
        }
    }

    pub(in crate::check) fn state_len(&self) -> usize {
        self.state_len
    }

    pub(in crate::check) fn is_native_fused_loop(&self) -> bool {
        self.native_fused_loop
    }

    pub(in crate::check) fn native_fused_invariant_count(&self) -> usize {
        self.native_fused_invariant_count
    }

    pub(in crate::check) fn native_fused_state_constraint_count(&self) -> usize {
        self.native_fused_state_constraint_count
    }

    pub(in crate::check) fn native_fused_mode(&self) -> &'static str {
        if !self.native_fused_loop {
            "prototype"
        } else if self.native_fused_state_constraint_count > 0 {
            "state_constraint_checking"
        } else if self.native_fused_invariant_count == 0 {
            "action_only"
        } else {
            "invariant_checking"
        }
    }

    pub(in crate::check) fn loop_kind_label(&self) -> &'static str {
        match self.native_fused_mode() {
            "state_constraint_checking" => "state-constrained native fused Trust-CG parent loop",
            "action_only" => "action-only native fused Trust-CG parent loop",
            "invariant_checking" => "invariant-checking native fused Trust-CG parent loop",
            _ => "prototype Rust parent loop over Trust-CG action/invariant pointers",
        }
    }

    pub(in crate::check) fn loop_kind_telemetry(&self) -> &'static str {
        match self.native_fused_mode() {
            "action_only" | "invariant_checking" | "state_constraint_checking" => {
                "native_fused_trust_cg_parent_loop"
            }
            _ => "prototype_rust_parent_loop_over_trust_cg_action_invariant_pointers",
        }
    }

    pub(in crate::check) fn native_fused_state_constraints_checked_by_backend(
        &self,
        expected_count: usize,
    ) -> bool {
        expected_count == 0
            || (self.native_fused_loop
                && self.native_fused_state_constraint_count == expected_count)
    }

    pub(in crate::check) fn native_fused_regular_invariants_checked_by_backend(&self) -> bool {
        self.native_fused_loop && self.regular_invariants_checked_by_backend
    }

    pub(in crate::check) fn native_fused_local_dedup(&self) -> bool {
        self.native_fused_loop && !self.state_graph_successors_complete
    }

    fn map_fused_level_error(error: tla_trust_cg::TrustCgBfsLevelError) -> Option<BfsStepError> {
        match error {
            tla_trust_cg::TrustCgBfsLevelError::FallbackNeeded => {
                eprintln!(
                    "[trust-cg] CompiledBfsLevel requested interpreter fallback; using non-fused fallback path"
                );
                None
            }
            tla_trust_cg::TrustCgBfsLevelError::BufferOverflow { partial_count } => {
                Some(BfsStepError::BufferOverflow { partial_count })
            }
            other => {
                eprintln!("[lazy-union-debug] fused level error detail: {other:?}");
                Some(BfsStepError::RuntimeError)
            }
        }
    }

    fn map_fused_level_error_for_level(
        &self,
        error: tla_trust_cg::TrustCgBfsLevelError,
    ) -> Option<BfsStepError> {
        if self.native_fused_loop && self.native_fused_state_constraint_count > 0 {
            eprintln!(
                "[trust-cg] state-constrained native fused CompiledBfsLevel failed closed: {error}"
            );
            return Some(BfsStepError::FatalRuntimeError);
        }
        Self::map_fused_level_error(error)
    }

    #[allow(dead_code)] // JIT batch-compile / trust-cg-cache machinery, currently unwired (TY_JIT off by default, #4035)
    fn map_error(error: tla_trust_cg::TrustCgBfsLevelError) -> BfsStepError {
        Self::map_fused_level_error(error).unwrap_or(BfsStepError::RuntimeError)
    }

    #[allow(dead_code)] // JIT batch-compile / trust-cg-cache machinery, currently unwired (TY_JIT off by default, #4035)
    fn level_result_from_trust_cg(
        successors: tla_trust_cg::TrustCgSuccessorArena,
        outcome: tla_trust_cg::TrustCgBfsLevelOutcome,
        regular_invariants_checked_by_backend: bool,
        state_graph_successors_complete: bool,
    ) -> super::bfs::compiled_step_trait::CompiledLevelResult {
        let (invariant_ok, failed_parent_idx, failed_invariant_idx, failed_successor_idx) =
            match outcome.invariant {
                tla_trust_cg::TrustCgInvariantStatus::Passed => (true, None, None, None),
                tla_trust_cg::TrustCgInvariantStatus::Failed {
                    parent_index,
                    invariant_index,
                    successor_index,
                } => (
                    false,
                    Some(parent_index as usize),
                    Some(invariant_index),
                    Some(successor_index as usize),
                ),
            };

        super::bfs::compiled_step_trait::CompiledLevelResult::from_trust_cg_successor_arena_with_failed_successor_idx(
            successors,
            outcome.parents_processed,
            outcome.total_generated,
            outcome.total_new,
            invariant_ok,
            failed_parent_idx,
            failed_invariant_idx,
            None,
            failed_successor_idx,
            regular_invariants_checked_by_backend,
        )
        .with_state_graph_successors_complete(state_graph_successors_complete)
    }

    fn level_result_from_reusable_trust_cg(
        successors: tla_trust_cg::TrustCgSuccessorArena,
        pool: super::bfs::compiled_step_trait::TrustCgSuccessorArenaPool,
        outcome: tla_trust_cg::TrustCgBfsLevelOutcome,
        regular_invariants_checked_by_backend: bool,
        state_graph_successors_complete: bool,
    ) -> super::bfs::compiled_step_trait::CompiledLevelResult {
        let (invariant_ok, failed_parent_idx, failed_invariant_idx, failed_successor_idx) =
            match outcome.invariant {
                tla_trust_cg::TrustCgInvariantStatus::Passed => (true, None, None, None),
                tla_trust_cg::TrustCgInvariantStatus::Failed {
                    parent_index,
                    invariant_index,
                    successor_index,
                } => (
                    false,
                    Some(parent_index as usize),
                    Some(invariant_index),
                    Some(successor_index as usize),
                ),
            };

        super::bfs::compiled_step_trait::CompiledLevelResult::from_reusable_trust_cg_successor_arena_with_failed_successor_idx(
            successors,
            pool,
            outcome.parents_processed,
            outcome.total_generated,
            outcome.total_new,
            invariant_ok,
            failed_parent_idx,
            failed_invariant_idx,
            None,
            failed_successor_idx,
            regular_invariants_checked_by_backend,
        )
        .with_state_graph_successors_complete(state_graph_successors_complete)
    }

    fn recycle_successor_arena(
        pool: &super::bfs::compiled_step_trait::TrustCgSuccessorArenaPool,
        mut successors: tla_trust_cg::TrustCgSuccessorArena,
    ) {
        successors.clear();
        if let Ok(mut pooled) = pool.lock() {
            if pooled.is_none() {
                *pooled = Some(successors);
            }
        }
    }

    fn maybe_run_native_callout_selftest(
        callout_selftest: &Mutex<Option<TrustCgNativeCalloutSelftest>>,
        arena: &[i64],
        parent_count: usize,
        state_len: usize,
    ) -> Result<(), BfsStepError> {
        let selftest = match callout_selftest.lock() {
            Ok(mut guard) => guard.take(),
            Err(_) => return Err(BfsStepError::RuntimeError),
        };
        let Some(selftest) = selftest else {
            return Ok(());
        };

        let fail_closed = selftest.fail_closed;
        match selftest.run_on_first_parent(arena, parent_count, state_len) {
            Ok(()) => {
                eprintln!("[trust_cg-selftest] native fused callout selftest complete");
                Ok(())
            }
            Err(reason) => {
                eprintln!("[trust_cg-selftest] native fused callout selftest failed: {reason}");
                if fail_closed {
                    eprintln!(
                        "[trust_cg-selftest] failing closed because {TRUST_CG_NATIVE_CALLOUT_SELFTEST_FAIL_CLOSED_ENV}=1 or {TRUST_CG_NATIVE_CALLOUT_SELFTEST_ENV}=strict/fail_closed"
                    );
                    Err(BfsStepError::FatalRuntimeError)
                } else {
                    Ok(())
                }
            }
        }
    }

    #[cfg(test)]
    fn run_mock_native(
        &self,
        entrypoint: tla_trust_cg::TrustCgFusedLevelFn,
        action_count: usize,
        scratch: &mut [i64],
        arena: &[i64],
        parent_count: usize,
    ) -> Option<Result<super::bfs::compiled_step_trait::CompiledLevelResult, BfsStepError>> {
        let parent_abi = match tla_trust_cg::TrustCgBfsParentArenaAbi::new(
            arena,
            parent_count,
            self.state_len,
            scratch,
        ) {
            Ok(parent_abi) => parent_abi,
            Err(error) => return Some(Err(Self::map_error(error))),
        };
        let mut successors = tla_trust_cg::TrustCgSuccessorArena::with_capacity(
            self.state_len,
            parent_count.saturating_mul(action_count),
        );
        let mut successor_abi =
            match successors.prepare_abi(parent_count.saturating_mul(action_count)) {
                Ok(successor_abi) => successor_abi,
                Err(error) => return Some(Err(Self::map_error(error))),
            };
        let returned_status = unsafe { entrypoint(&parent_abi, &mut successor_abi) };
        if returned_status != successor_abi.status {
            return Some(Err(BfsStepError::RuntimeError));
        }
        let outcome = match unsafe { successors.commit_abi(&successor_abi) } {
            Ok(outcome) => outcome,
            Err(error) => return self.map_fused_level_error_for_level(error).map(Err),
        };
        Some(Ok(Self::level_result_from_trust_cg(
            successors,
            outcome,
            self.regular_invariants_checked_by_backend,
            self.state_graph_successors_complete,
        )))
    }
}

impl super::bfs::compiled_step_trait::CompiledBfsLevel for TrustCgCompiledBfsLevel {
    fn has_fused_level(&self) -> bool {
        true
    }

    fn has_native_fused_level(&self) -> bool {
        self.native_fused_loop
    }

    fn fused_level_state_len(&self) -> Option<usize> {
        Some(self.state_len)
    }

    fn native_fused_state_constraint_count(&self) -> usize {
        self.native_fused_state_constraint_count
    }

    fn native_fused_invariant_count(&self) -> usize {
        self.native_fused_invariant_count
    }

    fn native_fused_state_constraints_checked_by_backend(&self, expected_count: usize) -> bool {
        self.native_fused_state_constraints_checked_by_backend(expected_count)
    }

    fn native_fused_regular_invariants_checked_by_backend(&self) -> bool {
        self.native_fused_regular_invariants_checked_by_backend()
    }

    fn native_fused_has_loop_actions(&self) -> bool {
        matches!(
            &self.implementation,
            TrustCgCompiledBfsLevelImplementation::Native {
                has_loop_actions: true,
                ..
            }
        )
    }

    fn skip_global_pre_seen_lookup(&self) -> bool {
        self.native_fused_loop
            && self.native_fused_invariant_count > 0
            && self.regular_invariants_checked_by_backend
    }

    fn preflight_fused_arena(
        &self,
        arena: &[i64],
        parent_count: usize,
    ) -> Result<(), BfsStepError> {
        match &self.implementation {
            TrustCgCompiledBfsLevelImplementation::Native {
                callout_selftest, ..
            } => Self::maybe_run_native_callout_selftest(
                callout_selftest,
                arena,
                parent_count,
                self.state_len,
            ),
            _ => Ok(()),
        }
    }

    fn run_level_fused_arena(
        &self,
        arena: &[i64],
        parent_count: usize,
    ) -> Option<Result<super::bfs::compiled_step_trait::CompiledLevelResult, BfsStepError>> {
        match &self.implementation {
            TrustCgCompiledBfsLevelImplementation::Prototype {
                prototype,
                successor_arena_pool,
            } => {
                let mut prototype = match prototype.lock() {
                    Ok(guard) => guard,
                    Err(_) => return Some(Err(BfsStepError::RuntimeError)),
                };
                let pool = successor_arena_pool.clone();
                let successors_from_pool = match pool.lock() {
                    Ok(mut pooled) => pooled.take(),
                    Err(_) => return Some(Err(BfsStepError::RuntimeError)),
                };
                let mut successors = successors_from_pool
                    .unwrap_or_else(|| tla_trust_cg::TrustCgSuccessorArena::new(self.state_len));
                // Clear-before-reuse soundness guard, identical to the native
                // path: a recycled prototype arena must arrive empty so it can
                // never alias stale prior-level successor/handle data.
                // `run_level_arena` re-clears, so this is a defensive check.
                debug_assert_eq!(
                    successors.successor_count(),
                    0,
                    "pooled prototype trust-cg successor arena reused with stale successors"
                );
                debug_assert!(
                    successors.states_flat().is_empty()
                        && successors.parent_indices().is_empty()
                        && successors.successor_fingerprints().is_empty(),
                    "pooled prototype trust-cg successor arena reused with stale sidecars"
                );

                let outcome = match prototype.run_level_arena(arena, parent_count, &mut successors)
                {
                    Ok(outcome) => outcome,
                    Err(error) => {
                        Self::recycle_successor_arena(&pool, successors);
                        return Self::map_fused_level_error(error).map(Err);
                    }
                };

                Some(Ok(Self::level_result_from_reusable_trust_cg(
                    successors,
                    pool,
                    outcome,
                    self.regular_invariants_checked_by_backend,
                    self.state_graph_successors_complete,
                )))
            }
            TrustCgCompiledBfsLevelImplementation::Native {
                level,
                action_count,
                has_loop_actions,
                loop_successors_per_parent,
                loop_prev_level_new,
                successor_arena_pool,
                callout_selftest,
            } => {
                if let Err(error) = Self::maybe_run_native_callout_selftest(
                    callout_selftest,
                    arena,
                    parent_count,
                    self.state_len,
                ) {
                    return Some(Err(error));
                }

                let mut level = match level.lock() {
                    Ok(guard) => guard,
                    Err(_) => return Some(Err(BfsStepError::RuntimeError)),
                };
                let pool = successor_arena_pool.clone();
                let successors_from_pool = match pool.lock() {
                    Ok(mut pooled) => pooled.take(),
                    Err(_) => return Some(Err(BfsStepError::RuntimeError)),
                };
                let mut successors = successors_from_pool
                    .unwrap_or_else(|| tla_trust_cg::TrustCgSuccessorArena::new(self.state_len));
                // Clear-before-reuse soundness guard for the pooled arena. Both
                // recycle paths (`recycle_successor_arena` and
                // `ReusableTrustCgSuccessorArena::drop`) clear before returning
                // to the pool, so a recycled arena must arrive empty. Prove it
                // here at the reuse boundary BEFORE `run_level_arena_with_capacity`
                // writes new successors, so a pooled arena can never alias stale
                // successor/handle data from a prior level (the MCLamportMutex
                // aliasing class). `run_level_arena_with_capacity` also re-clears,
                // so this is a defensive debug-only check, not the only barrier.
                debug_assert_eq!(
                    successors.successor_count(),
                    0,
                    "pooled trust-cg successor arena reused with stale successors"
                );
                debug_assert!(
                    successors.states_flat().is_empty(),
                    "pooled trust-cg successor arena reused with stale successor states"
                );
                debug_assert!(
                    successors.parent_indices().is_empty(),
                    "pooled trust-cg successor arena reused with stale parent indexes"
                );
                debug_assert!(
                    successors.successor_fingerprints().is_empty(),
                    "pooled trust-cg successor arena reused with stale fingerprints"
                );
                let base_successors_per_parent = (*action_count).max(1);
                let successors_per_parent = if *has_loop_actions {
                    // Narrow start (see TRUST_CG_LOOP_LEVEL_INITIAL_SUCCESSORS_
                    // PER_PARENT): the learned width converges upward via the
                    // sound grow-and-retry below instead of flooring at the
                    // parents x action_count worst case that dominated RSS.
                    loop_successors_per_parent
                        .load(std::sync::atomic::Ordering::Relaxed)
                        .max(TRUST_CG_LOOP_LEVEL_INITIAL_SUCCESSORS_PER_PARENT)
                } else {
                    // Non-loop specialized path: learn the per-parent successor
                    // width from the previous level (stored in the same atomic)
                    // rather than flooring at the proven `action_count` ceiling.
                    // Clamped to that ceiling so it can never exceed the sound
                    // upper bound. See TRUST_CG_NONLOOP_LEVEL_SUCCESSOR_WIDTH_SLACK.
                    loop_successors_per_parent
                        .load(std::sync::atomic::Ordering::Relaxed)
                        .clamp(1, base_successors_per_parent)
                };
                // Loop-kernel levels: size the arena from the PREVIOUS level's
                // actual committed yield (x2 headroom) instead of the
                // parents x width worst case — the dominant term in peak RSS.
                // Floored at parents x initial width; capped at the width-model
                // bound so the learned width still bounds growth. Mispredicts
                // are sound: overflow -> grow-and-retry below (complete level
                // re-run over a cleared arena).
                //
                // The per-parent ceiling differs by architecture: the loop-kernel
                // width-model bound for record-set actions, or the PROVEN
                // `action_count` for the specialized single-successor path (each
                // specialized action fires at most one successor per parent, so
                // `parents x action_count` can never overflow — it is the sound
                // termination bound for the grow-and-retry loop below).
                let hard_cap = if *has_loop_actions {
                    parent_count.saturating_mul(TRUST_CG_LOOP_LEVEL_MAX_SUCCESSORS_PER_PARENT)
                } else {
                    parent_count.saturating_mul(base_successors_per_parent)
                };
                let mut capacity_records = if *has_loop_actions {
                    // Predictive sizing: 2x the previous level's ACTUAL
                    // committed yield (levels grow smoothly, so this rarely
                    // misses), floored at parents x initial width for the
                    // first/tiny levels. Mispredicts are cheap AND sound: the
                    // retry below doubles capacity only (a complete native
                    // level re-run costs a fraction of a second), so peak RSS
                    // tracks ~2x the real yield instead of the parents x width
                    // worst case that dominated memory.
                    let prev_new = loop_prev_level_new.load(std::sync::atomic::Ordering::Relaxed);
                    let floor = parent_count
                        .saturating_mul(TRUST_CG_LOOP_LEVEL_INITIAL_SUCCESSORS_PER_PARENT);
                    prev_new.saturating_mul(2).max(floor).min(hard_cap)
                } else {
                    // Non-loop predictive sizing: parents x learned width, capped
                    // at the proven `parents x action_count` ceiling. The learned
                    // width tracks the observed per-parent yield (updated on the
                    // success arm below), so the arena is sized to the real
                    // successor rate instead of the worst case.
                    parent_count
                        .saturating_mul(successors_per_parent)
                        .min(hard_cap)
                };
                let outcome = loop {
                    match level.run_level_arena_with_capacity(
                        arena,
                        parent_count,
                        capacity_records,
                        &mut successors,
                    ) {
                        Ok(outcome) => {
                            if *has_loop_actions {
                                loop_successors_per_parent.store(
                                    successors_per_parent,
                                    std::sync::atomic::Ordering::Relaxed,
                                );
                                loop_prev_level_new.store(
                                    usize::try_from(outcome.total_new).unwrap_or(usize::MAX),
                                    std::sync::atomic::Ordering::Relaxed,
                                );
                            } else if parent_count > 0 {
                                // Learn the per-parent successor width from this
                                // level's ACTUAL committed fill for the next
                                // level's predictive sizing. `+slack` absorbs
                                // level-to-level rate variation so wider levels
                                // rarely need a grow-and-retry; clamped to the
                                // proven `action_count` ceiling.
                                let committed =
                                    usize::try_from(outcome.total_new).unwrap_or(usize::MAX);
                                let observed = committed.div_ceil(parent_count);
                                let learned = observed
                                    .saturating_add(TRUST_CG_NONLOOP_LEVEL_SUCCESSOR_WIDTH_SLACK)
                                    .clamp(1, base_successors_per_parent);
                                loop_successors_per_parent
                                    .store(learned, std::sync::atomic::Ordering::Relaxed);
                            }
                            break outcome;
                        }
                        // Adaptive arena growth (BOTH architectures). The
                        // predictive initial sizing above can legitimately
                        // undersize a level: a record-set loop kernel emits a
                        // runtime-dependent successor count, and the non-loop
                        // specialized path is now sized from the learned yield
                        // rather than the `parents x action_count` worst case.
                        // Each native invocation is a complete level re-run over
                        // a cleared arena and a reset local-dedup set
                        // (`run_level_arena_with_capacity` clears on entry and on
                        // every error), and the native code checks capacity
                        // before every successor commit, so an overflow commits
                        // NOTHING — the partial arena tail is dead — and growing
                        // and retrying preserves the exact successor set. The
                        // learned width persists across levels via
                        // `loop_successors_per_parent`. `hard_cap` is a proven
                        // ceiling for both paths (`parents x action_count` for
                        // the specialized path can never overflow), so this loop
                        // terminates; beyond it the overflow propagates and the
                        // checker falls back (also sound, never truncated).
                        Err(tla_trust_cg::TrustCgBfsLevelError::BufferOverflow { .. })
                            if capacity_records < hard_cap =>
                        {
                            let grown_capacity = capacity_records.saturating_mul(2).min(hard_cap);
                            eprintln!(
                                "[trust-cg] native fused level overflowed the successor arena; growing capacity {}→{} records and re-running the level (sound: complete level re-run)",
                                capacity_records, grown_capacity,
                            );
                            capacity_records = grown_capacity;
                            continue;
                        }
                        Err(error) => {
                            Self::recycle_successor_arena(&pool, successors);
                            return self.map_fused_level_error_for_level(error).map(Err);
                        }
                    }
                };

                if std::env::var_os("TY_DEBUG_SUBBATCH").is_some() {
                    eprintln!(
                        "[subbatch-sizing] parents={parent_count} capacity_records={capacity_records} \
                         total_new={} total_generated={} loop_actions={}",
                        outcome.total_new, outcome.total_generated, has_loop_actions,
                    );
                }

                Some(Ok(Self::level_result_from_reusable_trust_cg(
                    successors,
                    pool,
                    outcome,
                    self.regular_invariants_checked_by_backend,
                    self.state_graph_successors_complete,
                )))
            }
            #[cfg(test)]
            TrustCgCompiledBfsLevelImplementation::MockNative {
                entrypoint,
                action_count,
                scratch,
            } => {
                let mut scratch = match scratch.lock() {
                    Ok(guard) => guard,
                    Err(_) => return Some(Err(BfsStepError::RuntimeError)),
                };
                self.run_mock_native(
                    *entrypoint,
                    *action_count,
                    &mut scratch,
                    arena,
                    parent_count,
                )
            }
        }
    }
}

#[cfg(test)]
mod tests;
